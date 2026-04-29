#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from utils import (
    extract_polyA_features,
    get_context_around,
    kozak_score_and_strength,
    load_config,
    scan_uorfs,
    setup_logger,
)


def parse_mirna_table(path: str, tx_col_candidates=None, mirna_col_candidates=None):
    if tx_col_candidates is None:
        tx_col_candidates = ["transcript_id", "target_id", "sequence_id", "mRNA"]
    if mirna_col_candidates is None:
        mirna_col_candidates = ["miRNA", "mirna", "miRNA_id", "query_id"]
    df = pd.read_csv(path, sep="\t")
    tx_col = next((c for c in tx_col_candidates if c in df.columns), None)
    mir_col = next((c for c in mirna_col_candidates if c in df.columns), None)
    if tx_col is None:
        return pd.DataFrame(columns=["transcript_id", "miRNA_site_count", "unique_miRNA_count", "conserved_site_count", "strongest_binding_energy"])
    if mir_col is None:
        df["_mirna"] = "NA"
        mir_col = "_mirna"

    agg = df.groupby(tx_col).agg(
        miRNA_site_count=(tx_col, "size"),
        unique_miRNA_count=(mir_col, lambda x: x.nunique()),
    ).reset_index().rename(columns={tx_col: "transcript_id"})

    if "conserved_site_count" in df.columns:
        cons = df.groupby(tx_col)["conserved_site_count"].sum().reset_index().rename(columns={tx_col: "transcript_id"})
        agg = agg.merge(cons, on="transcript_id", how="left")
    else:
        agg["conserved_site_count"] = np.nan

    energy_col = next((c for c in ["energy", "binding_energy", "mfe"] if c in df.columns), None)
    if energy_col:
        en = df.groupby(tx_col)[energy_col].min().reset_index().rename(columns={tx_col: "transcript_id", energy_col: "strongest_binding_energy"})
        agg = agg.merge(en, on="transcript_id", how="left")
    else:
        agg["strongest_binding_energy"] = np.nan

    return agg


def main():
    ap = argparse.ArgumentParser(description="Scan sequence motifs: Kozak, uORF, polyA, miRNA summary")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("02_motif", os.path.join(cfg["output"]["tables_dir"], "02_scan_sequence_motifs.log"))

    df = pd.read_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step1.tsv"), sep="\t")

    out_rows = []
    for _, r in df.iterrows():
        tx_seq = str(r.get("transcript_sequence", "") or "")
        utr5 = str(r.get("five_utr_sequence", "") or "")
        utr3 = str(r.get("three_utr_sequence", "") or "")
        start_pos = r.get("start_codon_pos_in_transcript", np.nan)

        if pd.notna(start_pos):
            context = get_context_around(tx_seq, int(start_pos), left=6, right=2)
            k_score, k_strength = kozak_score_and_strength(context)
        else:
            context, k_score, k_strength = "NA", 0, "NA"

        uorfs = scan_uorfs(utr5)
        longest_nt = max([(b - a) for a, b in uorfs], default=0)
        longest_aa = longest_nt // 3 if longest_nt else 0
        uorf_strong = 0
        coord_text = []
        for a, b in uorfs:
            uctx = get_context_around(utr5, a, left=6, right=2)
            _, us = kozak_score_and_strength(uctx)
            if us == "strong":
                uorf_strong += 1
            coord_text.append(f"{a + 1}-{b}")

        poly = extract_polyA_features(utr3)

        out_rows.append(
            {
                "transcript_id": r["transcript_id"],
                "uORF_count": len(uorfs),
                "longest_uORF_length_nt": longest_nt,
                "longest_uORF_length_aa": longest_aa,
                "uORF_with_strong_kozak_count": uorf_strong,
                "uORF_coordinates_in_5utr": ";".join(coord_text),
                "kozak_context": context,
                "kozak_score": k_score,
                "kozak_strength": k_strength,
                **poly,
            }
        )

    motif_df = pd.DataFrame(out_rows)

    mirna_path = cfg["input"].get("mirna_tsv")
    if mirna_path and os.path.exists(mirna_path):
        mirna_df = parse_mirna_table(mirna_path)
        logger.info("Loaded miRNA table: %s", mirna_path)
    else:
        mirna_df = pd.DataFrame({"transcript_id": df["transcript_id"].astype(str).unique()})
        mirna_df["miRNA_site_count"] = np.nan
        mirna_df["unique_miRNA_count"] = np.nan
        mirna_df["conserved_site_count"] = np.nan
        mirna_df["strongest_binding_energy"] = np.nan
        logger.info("No miRNA table provided. Filled NA.")

    merged = df.merge(motif_df, on="transcript_id", how="left")
    merged = merged.merge(mirna_df, on="transcript_id", how="left")
    merged.to_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step2.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
