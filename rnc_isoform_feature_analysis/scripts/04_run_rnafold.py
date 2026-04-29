#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import add_common_cli_args, read_config, run_rnafold, setup_logger, which


def read_fasta_dict(fp: Path):
    d, cur = {}, None
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].split()[0]
                d[cur] = ""
            else:
                d[cur] += line
    return d


def main():
    parser = argparse.ArgumentParser(description="Run RNAfold (optional) for UTR and CDS-start windows")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("04_run_rnafold", Path(args.log) if args.log else None)

    df = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "transcript_features_orf_nmd.tsv", sep="\t")
    seq_dir = Path(cfg["outputs"]["sequences_dir"])
    tx = read_fasta_dict(seq_dir / "transcript.fa")
    five = read_fasta_dict(seq_dir / "five_utr.fa")
    three = read_fasta_dict(seq_dir / "three_utr.fa")

    rnafold_bin = cfg["params"].get("rnafold_bin", "RNAfold")
    installed = which(rnafold_bin) is not None
    max_len = int(cfg["params"].get("rnafold_max_len", 1500))
    flank = int(cfg["params"].get("cds_start_flank", 70))

    rows = []
    for _, r in df.iterrows():
        tid = r["transcript_id_base"]
        five_seq = five.get(tid, "")
        tx_seq = tx.get(tid, "")
        three_seq = three.get(tid, "")

        cds_start = len(five_seq)
        start_window = tx_seq[max(0, cds_start - flank): min(len(tx_seq), cds_start + flank)] if tx_seq else ""

        def clip(s: str):
            if len(s) <= max_len:
                return s
            center = len(s) // 2
            half = max_len // 2
            return s[max(0, center - half): center + half]

        if installed:
            five_mfe = run_rnafold(clip(five_seq), rnafold_bin)
            start_mfe = run_rnafold(clip(start_window), rnafold_bin)
            three_mfe = run_rnafold(clip(three_seq), rnafold_bin)
        else:
            five_mfe = start_mfe = three_mfe = np.nan

        rows.append({"transcript_id_base": tid, "five_utr_mfe": five_mfe, "cds_start_window_mfe": start_mfe, "three_utr_mfe": three_mfe})

    mfe_df = pd.DataFrame(rows)
    out = df.merge(mfe_df, on="transcript_id_base", how="left")
    out_path = Path(cfg["outputs"]["tables_dir"]) / "transcript_features.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    if not installed:
        logger.warning("RNAfold not found. MFE columns filled with NA. Install ViennaRNA and set params.rnafold_bin.")
    logger.info("Final transcript feature table written: %s", out_path)


if __name__ == "__main__":
    main()
