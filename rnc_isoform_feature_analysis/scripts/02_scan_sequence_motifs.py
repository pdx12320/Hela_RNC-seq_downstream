#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import add_common_cli_args, kozak_context_score, read_config, scan_polya, scan_uorfs, setup_logger


def parse_mirna_table(path: Path, tid_col: str = "transcript_id_base") -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if tid_col not in df.columns:
        for cand in ["transcript_id", "target_id", "mRNA"]:
            if cand in df.columns:
                df[tid_col] = df[cand].astype(str).map(lambda x: x.split(".")[0])
                break
    if tid_col not in df.columns:
        return pd.DataFrame(columns=[tid_col, "miRNA_site_count", "unique_miRNA_count", "conserved_site_count", "strongest_binding_energy"])

    if "miRNA" not in df.columns:
        for cand in ["mirna", "miRNA_id", "query_id"]:
            if cand in df.columns:
                df["miRNA"] = df[cand]
                break
        if "miRNA" not in df.columns:
            df["miRNA"] = "NA"

    energy_col = None
    for c in ["energy", "binding_energy", "MFE", "score"]:
        if c in df.columns:
            energy_col = c
            break

    out = []
    for tid, g in df.groupby(tid_col):
        d = {
            tid_col: tid,
            "miRNA_site_count": len(g),
            "unique_miRNA_count": g["miRNA"].nunique(dropna=True),
            "conserved_site_count": g["conserved"].sum() if "conserved" in g.columns else np.nan,
            "strongest_binding_energy": g[energy_col].min() if energy_col else np.nan,
        }
        out.append(d)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="Scan Kozak/uORF/polyA and integrate optional miRNA predictions")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("02_scan_sequence_motifs", Path(args.log) if args.log else None)

    table = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "transcript_basic_features.tsv", sep="\t")
    seq_dir = Path(cfg["outputs"]["sequences_dir"])

    def read_fa_to_dict(fp: Path):
        d = {}
        cur = None
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    cur = line[1:].split()[0]
                    d[cur] = ""
                else:
                    d[cur] += line.upper()
        return d

    tx = read_fa_to_dict(seq_dir / "transcript.fa")
    five = read_fa_to_dict(seq_dir / "five_utr.fa")
    three = read_fa_to_dict(seq_dir / "three_utr.fa")

    motif_rows = []
    for _, r in table.iterrows():
        tid = r["transcript_id_base"]
        tx_seq = tx.get(tid, "")
        five_seq = five.get(tid, "")
        three_seq = three.get(tid, "")

        start_idx = len(five_seq) if tx_seq else -1
        ctx, sc, st = kozak_context_score(tx_seq, start_idx)
        u = scan_uorfs(five_seq)
        p = scan_polya(three_seq)

        motif_rows.append({"transcript_id_base": tid, "kozak_context": ctx, "kozak_score": sc, "kozak_strength": st, **u, **p})

    motif_df = pd.DataFrame(motif_rows)

    mirna_df = None
    mirna_input = cfg["inputs"].get("mirna_prediction")
    if mirna_input and Path(mirna_input).exists():
        mirna_df = parse_mirna_table(Path(mirna_input))
        logger.info("Parsed miRNA prediction from %s", mirna_input)
    else:
        mirna_df = pd.DataFrame({
            "transcript_id_base": table["transcript_id_base"],
            "miRNA_site_count": np.nan,
            "unique_miRNA_count": np.nan,
            "conserved_site_count": np.nan,
            "strongest_binding_energy": np.nan,
        })
        logger.info("No miRNA prediction file supplied, fill NA")

    out = table.merge(motif_df, on="transcript_id_base", how="left").merge(mirna_df, on="transcript_id_base", how="left")
    out_path = Path(cfg["outputs"]["tables_dir"]) / "transcript_features_motifs.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    logger.info("Motif feature table written: %s", out_path)


if __name__ == "__main__":
    main()
