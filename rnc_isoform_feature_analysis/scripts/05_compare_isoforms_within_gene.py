#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from utils import add_common_cli_args, read_config, setup_logger


def cmp_change(a, b):
    if pd.isna(a) or pd.isna(b):
        return "NA"
    return f"{b}->{a}" if a != b else "same"


def main():
    parser = argparse.ArgumentParser(description="Within-gene isoform pairwise comparison and statistics")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("05_compare_isoforms_within_gene", Path(args.log) if args.log else None)

    df = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "transcript_features.tsv", sep="\t")

    pair_rows = []
    for gene, g in df.groupby("gene_name"):
        if len(g) < 2:
            continue
        ref = g["reference_transcript"].iloc[0]
        ref_row = g[g["transcript_id_base"] == ref]
        if ref_row.empty:
            ref_row = g.iloc[[0]]
            ref = ref_row.iloc[0]["transcript_id_base"]
        rr = ref_row.iloc[0]

        for _, qr in g.iterrows():
            pair_rows.append(
                {
                    "gene_name": gene,
                    "ref_transcript": ref,
                    "query_transcript": qr["transcript_id_base"],
                    "delta_log2FC": qr["log2FC"] - rr["log2FC"],
                    "delta_Delta_IF": qr["Delta_IF"] - rr["Delta_IF"],
                    "delta_5utr_length": qr["five_utr_length"] - rr["five_utr_length"],
                    "delta_cds_length": qr["cds_length"] - rr["cds_length"],
                    "delta_3utr_length": qr["three_utr_length"] - rr["three_utr_length"],
                    "orf_changed": qr.get("orf_changed_vs_reference", np.nan),
                    "domain_changed": qr.get("domain_changed_vs_reference", np.nan),
                    "delta_uORF_count": qr["uORF_count"] - rr["uORF_count"],
                    "kozak_strength_change": cmp_change(qr["kozak_strength"], rr["kozak_strength"]),
                    "delta_miRNA_site_count": qr.get("miRNA_site_count", np.nan) - rr.get("miRNA_site_count", np.nan),
                    "polyA_signal_change": cmp_change(qr["nearest_polyA_signal_to_3end"], rr["nearest_polyA_signal_to_3end"]),
                    "delta_5utr_mfe": qr.get("five_utr_mfe", np.nan) - rr.get("five_utr_mfe", np.nan),
                    "delta_cds_start_window_mfe": qr.get("cds_start_window_mfe", np.nan) - rr.get("cds_start_window_mfe", np.nan),
                    "delta_3utr_mfe": qr.get("three_utr_mfe", np.nan) - rr.get("three_utr_mfe", np.nan),
                    "nmd_likelihood_change": cmp_change(qr.get("nmd_likelihood", "NA"), rr.get("nmd_likelihood", "NA")),
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    pair_fp = Path(cfg["outputs"]["tables_dir"]) / "isoform_pairwise_comparison.tsv"
    pair_df.to_csv(pair_fp, sep="\t", index=False)

    stats_rows = []
    numeric_features = ["five_utr_length", "cds_length", "three_utr_length", "uORF_count", "miRNA_site_count", "five_utr_mfe", "cds_start_window_mfe", "three_utr_mfe"]
    for feat in numeric_features:
        d = df[["Delta_IF", feat]].dropna()
        if len(d) >= 3:
            rho, p = spearmanr(d["Delta_IF"], d[feat])
            stats_rows.append({"test": "spearman", "feature": feat, "statistic": rho, "pvalue": p, "n": len(d)})

    kz = [x["Delta_IF"].dropna().values for _, x in df.groupby("kozak_strength") if len(x["Delta_IF"].dropna()) > 0]
    if len(kz) >= 2:
        stat, p = kruskal(*kz)
        stats_rows.append({"test": "kruskal", "feature": "Delta_IF~kozak_strength", "statistic": stat, "pvalue": p, "n": sum(len(x) for x in kz)})

    if "orf_changed_vs_reference" in df.columns:
        g1 = df[df["orf_changed_vs_reference"] == True]["Delta_IF"].dropna()
        g0 = df[df["orf_changed_vs_reference"] == False]["Delta_IF"].dropna()
        if len(g1) > 0 and len(g0) > 0:
            stat, p = mannwhitneyu(g1, g0, alternative="two-sided")
            stats_rows.append({"test": "wilcoxon_rank_sum", "feature": "Delta_IF~orf_changed", "statistic": stat, "pvalue": p, "n": len(g1) + len(g0)})

    g_high = df[df["nmd_likelihood"] == "high"]["Delta_IF"].dropna()
    g_low = df[df["nmd_likelihood"] == "low"]["Delta_IF"].dropna()
    if len(g_high) > 0 and len(g_low) > 0:
        stat, p = mannwhitneyu(g_high, g_low, alternative="two-sided")
        stats_rows.append({"test": "wilcoxon_rank_sum", "feature": "Delta_IF~nmd_high_vs_low", "statistic": stat, "pvalue": p, "n": len(g_high) + len(g_low)})

    stats_df = pd.DataFrame(stats_rows)
    stats_fp = Path(cfg["outputs"]["tables_dir"]) / "statistics_summary.tsv"
    stats_df.to_csv(stats_fp, sep="\t", index=False)

    logger.info("Pairwise comparison written: %s", pair_fp)
    logger.info("Statistics summary written: %s", stats_fp)


if __name__ == "__main__":
    main()
