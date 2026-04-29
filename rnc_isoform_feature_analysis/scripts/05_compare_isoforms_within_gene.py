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


def safe_threshold_name(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def get_ddif_threshold(args, cfg) -> float:
    if getattr(args, "ddif_threshold", None) is not None:
        return float(args.ddif_threshold)
    return float(cfg.get("params", {}).get("ddif_threshold", 0.2))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def add_pairwise_statistics(pair_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []

    if pair_df.empty or len(pair_df) < 3:
        return pd.DataFrame(
            [
                {
                    "analysis_scope": "candidate_pairwise",
                    "ddif_threshold": threshold,
                    "n_pairs_used": len(pair_df),
                    "feature": "NA",
                    "test": "NA",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "insufficient candidate pairs",
                }
            ]
        )

    numeric_delta_features = [
        "delta_5utr_length",
        "delta_cds_length",
        "delta_3utr_length",
        "delta_uORF_count",
        "delta_miRNA_site_count",
        "delta_5utr_mfe",
        "delta_cds_start_window_mfe",
        "delta_3utr_mfe",
    ]

    pair_df = pair_df.copy()
    pair_df["delta_Delta_IF"] = to_num(pair_df["delta_Delta_IF"])

    for feat in numeric_delta_features:
        if feat not in pair_df.columns:
            continue

        pair_df[feat] = to_num(pair_df[feat])
        d = pair_df[["delta_Delta_IF", feat]].dropna()

        if len(d) >= 3 and d[feat].nunique() > 1:
            rho, p = spearmanr(d["delta_Delta_IF"], d[feat])
            rows.append(
                {
                    "analysis_scope": "candidate_pairwise",
                    "ddif_threshold": threshold,
                    "n_pairs_used": len(d),
                    "feature": f"delta_Delta_IF~{feat}",
                    "test": "spearman",
                    "statistic": rho,
                    "p_value": p,
                    "note": "",
                }
            )
        else:
            rows.append(
                {
                    "analysis_scope": "candidate_pairwise",
                    "ddif_threshold": threshold,
                    "n_pairs_used": len(d),
                    "feature": f"delta_Delta_IF~{feat}",
                    "test": "spearman",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "insufficient data or no variation",
                }
            )

    group_features = [
        "orf_changed",
        "domain_changed",
        "kozak_strength_change",
        "nmd_likelihood_change",
        "polyA_signal_change",
    ]

    for feat in group_features:
        if feat not in pair_df.columns:
            continue

        tmp = pair_df[["delta_Delta_IF", feat]].dropna()
        tmp = tmp[tmp[feat].astype(str) != "NA"]

        groups = [
            g["delta_Delta_IF"].dropna().values
            for _, g in tmp.groupby(feat)
            if len(g["delta_Delta_IF"].dropna()) > 0
        ]

        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                rows.append(
                    {
                        "analysis_scope": "candidate_pairwise",
                        "ddif_threshold": threshold,
                        "n_pairs_used": sum(len(x) for x in groups),
                        "feature": f"delta_Delta_IF~{feat}",
                        "test": "kruskal",
                        "statistic": stat,
                        "p_value": p,
                        "note": "",
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "analysis_scope": "candidate_pairwise",
                        "ddif_threshold": threshold,
                        "n_pairs_used": len(tmp),
                        "feature": f"delta_Delta_IF~{feat}",
                        "test": "kruskal",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "note": str(e),
                    }
                )
        else:
            rows.append(
                {
                    "analysis_scope": "candidate_pairwise",
                    "ddif_threshold": threshold,
                    "n_pairs_used": len(tmp),
                    "feature": f"delta_Delta_IF~{feat}",
                    "test": "kruskal",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "fewer than two groups",
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Within-gene isoform pairwise comparison filtered by abs(delta_Delta_IF)"
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--ddif-threshold",
        type=float,
        default=None,
        help="Keep candidate isoform pairs with abs(delta_Delta_IF) > threshold. "
             "Default: config.yaml params.ddif_threshold, fallback 0.2.",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger(
        "05_compare_isoforms_within_gene",
        Path(args.log) if args.log else None,
    )

    threshold = get_ddif_threshold(args, cfg)

    tables_dir = Path(cfg["outputs"]["tables_dir"])
    df = pd.read_csv(tables_dir / "transcript_features.tsv", sep="\t")

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
                    "kozak_strength_change": cmp_change(
                        qr["kozak_strength"],
                        rr["kozak_strength"],
                    ),
                    "delta_miRNA_site_count": qr.get("miRNA_site_count", np.nan)
                    - rr.get("miRNA_site_count", np.nan),
                    "polyA_signal_change": cmp_change(
                        qr["nearest_polyA_signal_to_3end"],
                        rr["nearest_polyA_signal_to_3end"],
                    ),
                    "delta_5utr_mfe": qr.get("five_utr_mfe", np.nan)
                    - rr.get("five_utr_mfe", np.nan),
                    "delta_cds_start_window_mfe": qr.get("cds_start_window_mfe", np.nan)
                    - rr.get("cds_start_window_mfe", np.nan),
                    "delta_3utr_mfe": qr.get("three_utr_mfe", np.nan)
                    - rr.get("three_utr_mfe", np.nan),
                    "nmd_likelihood_change": cmp_change(
                        qr.get("nmd_likelihood", "NA"),
                        rr.get("nmd_likelihood", "NA"),
                    ),
                }
            )

    pair_df = pd.DataFrame(pair_rows)

    if pair_df.empty:
        logger.warning("No pairwise isoform comparisons generated.")
        pair_df.to_csv(tables_dir / "isoform_pairwise_comparison.tsv", sep="\t", index=False)
        return

    pair_df["delta_Delta_IF"] = to_num(pair_df["delta_Delta_IF"])
    pair_df["abs_delta_Delta_IF"] = pair_df["delta_Delta_IF"].abs()

    all_fp = tables_dir / "isoform_pairwise_comparison_all.tsv"
    pair_df.to_csv(all_fp, sep="\t", index=False)

    candidate_df = pair_df[
        (pair_df["ref_transcript"].astype(str) != pair_df["query_transcript"].astype(str))
        & pair_df["delta_Delta_IF"].notna()
        & (pair_df["abs_delta_Delta_IF"] > threshold)
    ].copy()

    candidate_df = candidate_df.sort_values("abs_delta_Delta_IF", ascending=False)

    threshold_tag = safe_threshold_name(threshold)

    filtered_fp = tables_dir / f"isoform_pairwise_comparison.filtered.ddif_gt_{threshold_tag}.tsv"
    candidate_df.to_csv(filtered_fp, sep="\t", index=False)

    # 关键：为了让 06 默认读到筛选后的结果，这里覆盖原来的默认文件名
    pair_fp = tables_dir / "isoform_pairwise_comparison.tsv"
    candidate_df.to_csv(pair_fp, sep="\t", index=False)

    stats_df = add_pairwise_statistics(candidate_df, threshold)
    stats_fp = tables_dir / "statistics_summary.tsv"
    stats_df.to_csv(stats_fp, sep="\t", index=False)

    logger.info("All pairwise comparison written: %s", all_fp)
    logger.info("Filtered candidate comparison written: %s", filtered_fp)
    logger.info("Default pairwise comparison overwritten with filtered candidates: %s", pair_fp)
    logger.info("Statistics summary based on filtered candidates written: %s", stats_fp)
    logger.info("Threshold: abs(delta_Delta_IF) > %.4g", threshold)
    logger.info("Total pairwise rows: %d", len(pair_df))
    logger.info(
        "Non-self pairwise rows: %d",
        (pair_df["ref_transcript"].astype(str) != pair_df["query_transcript"].astype(str)).sum(),
    )
    logger.info("Candidate pairwise rows retained: %d", len(candidate_df))
    logger.info(
        "Candidate genes retained: %d",
        candidate_df["gene_name"].nunique() if not candidate_df.empty else 0,
    )


if __name__ == "__main__":
    main()
