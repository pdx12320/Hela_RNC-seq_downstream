#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from utils import add_common_cli_args, read_config, setup_logger


def safe_threshold_name(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def get_ddif_threshold(args, cfg) -> float:
    if getattr(args, "ddif_threshold", None) is not None:
        return float(args.ddif_threshold)
    return float(cfg.get("params", {}).get("ddif_threshold", 0.2))


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def value(row, col, default=np.nan):
    if col in row.index:
        return row[col]
    return default


def delta_num(high_row, low_row, col):
    hv = pd.to_numeric(value(high_row, col), errors="coerce")
    lv = pd.to_numeric(value(low_row, col), errors="coerce")
    if pd.isna(hv) or pd.isna(lv):
        return np.nan
    return hv - lv


def change_str(high_row, low_row, col):
    hv = value(high_row, col, np.nan)
    lv = value(low_row, col, np.nan)

    if pd.isna(hv) or pd.isna(lv):
        return "NA"

    if str(hv) == str(lv):
        return "same"

    return f"{lv}->{hv}"


def is_different(high_row, low_row, col):
    hv = value(high_row, col, np.nan)
    lv = value(low_row, col, np.nan)

    if pd.isna(hv) or pd.isna(lv):
        return np.nan

    return str(hv) != str(lv)


def build_high_vs_low_table(df: pd.DataFrame, logger) -> pd.DataFrame:
    rows = []

    df = df.copy()

    if "gene_name" not in df.columns:
        raise ValueError("Missing required column: gene_name")

    if "transcript_id_base" not in df.columns:
        raise ValueError("Missing required column: transcript_id_base")

    if "Delta_IF" not in df.columns:
        raise ValueError("Missing required column: Delta_IF")

    df["Delta_IF_num"] = pd.to_numeric(df["Delta_IF"], errors="coerce")

    usable = df.dropna(subset=["Delta_IF_num"]).copy()

    logger.info("Total transcripts: %d", len(df))
    logger.info("Transcripts with valid Delta_IF: %d", len(usable))

    for gene, g in usable.groupby("gene_name"):
        g = g.copy()

        # 关键：必须是同一基因内至少 2 个 isoform
        if g["transcript_id_base"].nunique() < 2:
            continue

        g = g.sort_values("Delta_IF_num", ascending=True)

        low_row = g.iloc[0]
        high_row = g.iloc[-1]

        low_tid = str(low_row["transcript_id_base"])
        high_tid = str(high_row["transcript_id_base"])

        if high_tid == low_tid:
            continue

        low_delta_if = float(low_row["Delta_IF_num"])
        high_delta_if = float(high_row["Delta_IF_num"])
        delta_delta_if = high_delta_if - low_delta_if

        rows.append(
            {
                "gene_name": gene,
                "n_isoforms_in_gene": g["transcript_id_base"].nunique(),

                "low_efficiency_transcript": low_tid,
                "high_efficiency_transcript": high_tid,

                "low_Delta_IF": low_delta_if,
                "high_Delta_IF": high_delta_if,
                "delta_Delta_IF": delta_delta_if,
                "abs_delta_Delta_IF": abs(delta_delta_if),

                "low_log2FC": value(low_row, "log2FC"),
                "high_log2FC": value(high_row, "log2FC"),
                "delta_log2FC": delta_num(high_row, low_row, "log2FC"),

                "low_IF_Total": value(low_row, "IF_Total"),
                "high_IF_Total": value(high_row, "IF_Total"),
                "delta_IF_Total": delta_num(high_row, low_row, "IF_Total"),

                "low_IF_Ribo": value(low_row, "IF_Ribo"),
                "high_IF_Ribo": value(high_row, "IF_Ribo"),
                "delta_IF_Ribo": delta_num(high_row, low_row, "IF_Ribo"),

                "low_five_utr_length": value(low_row, "five_utr_length"),
                "high_five_utr_length": value(high_row, "five_utr_length"),
                "delta_5utr_length": delta_num(high_row, low_row, "five_utr_length"),

                "low_cds_length": value(low_row, "cds_length"),
                "high_cds_length": value(high_row, "cds_length"),
                "delta_cds_length": delta_num(high_row, low_row, "cds_length"),

                "low_three_utr_length": value(low_row, "three_utr_length"),
                "high_three_utr_length": value(high_row, "three_utr_length"),
                "delta_3utr_length": delta_num(high_row, low_row, "three_utr_length"),

                "low_transcript_length": value(low_row, "transcript_length"),
                "high_transcript_length": value(high_row, "transcript_length"),
                "delta_transcript_length": delta_num(high_row, low_row, "transcript_length"),

                "low_protein_length": value(low_row, "protein_length"),
                "high_protein_length": value(high_row, "protein_length"),
                "delta_protein_length": delta_num(high_row, low_row, "protein_length"),

                "low_uORF_count": value(low_row, "uORF_count"),
                "high_uORF_count": value(high_row, "uORF_count"),
                "delta_uORF_count": delta_num(high_row, low_row, "uORF_count"),

                "low_kozak_strength": value(low_row, "kozak_strength"),
                "high_kozak_strength": value(high_row, "kozak_strength"),
                "kozak_strength_change": change_str(high_row, low_row, "kozak_strength"),

                "low_miRNA_site_count": value(low_row, "miRNA_site_count"),
                "high_miRNA_site_count": value(high_row, "miRNA_site_count"),
                "delta_miRNA_site_count": delta_num(high_row, low_row, "miRNA_site_count"),

                "low_polyA_signal_count": value(low_row, "polyA_signal_count"),
                "high_polyA_signal_count": value(high_row, "polyA_signal_count"),
                "delta_polyA_signal_count": delta_num(high_row, low_row, "polyA_signal_count"),

                "low_five_utr_mfe": value(low_row, "five_utr_mfe"),
                "high_five_utr_mfe": value(high_row, "five_utr_mfe"),
                "delta_5utr_mfe": delta_num(high_row, low_row, "five_utr_mfe"),

                "low_cds_start_window_mfe": value(low_row, "cds_start_window_mfe"),
                "high_cds_start_window_mfe": value(high_row, "cds_start_window_mfe"),
                "delta_cds_start_window_mfe": delta_num(high_row, low_row, "cds_start_window_mfe"),

                "low_three_utr_mfe": value(low_row, "three_utr_mfe"),
                "high_three_utr_mfe": value(high_row, "three_utr_mfe"),
                "delta_3utr_mfe": delta_num(high_row, low_row, "three_utr_mfe"),

                "low_nmd_likelihood": value(low_row, "nmd_likelihood"),
                "high_nmd_likelihood": value(high_row, "nmd_likelihood"),
                "nmd_likelihood_change": change_str(high_row, low_row, "nmd_likelihood"),

                # high vs low 的 ORF/domain 变化。优先用更直接的长度变化近似。
                "orf_changed_high_vs_low": (
                    delta_num(high_row, low_row, "cds_length") != 0
                    or delta_num(high_row, low_row, "protein_length") != 0
                ),

                "domain_changed_high_vs_low": is_different(
                    high_row,
                    low_row,
                    "domain_changed_vs_reference",
                ),

                "high_reference_transcript": value(high_row, "reference_transcript"),
                "low_reference_transcript": value(low_row, "reference_transcript"),
            }
        )

    return pd.DataFrame(rows)


def add_high_low_statistics(comp: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []

    if comp.empty or len(comp) < 3:
        return pd.DataFrame(
            [
                {
                    "analysis_scope": "high_vs_low_within_gene",
                    "ddif_threshold": threshold,
                    "n_genes_used": len(comp),
                    "feature": "NA",
                    "test": "NA",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "insufficient genes after filtering",
                }
            ]
        )

    comp = comp.copy()
    comp["delta_Delta_IF"] = to_num(comp["delta_Delta_IF"])

    numeric_features = [
        "delta_5utr_length",
        "delta_cds_length",
        "delta_3utr_length",
        "delta_transcript_length",
        "delta_protein_length",
        "delta_uORF_count",
        "delta_miRNA_site_count",
        "delta_polyA_signal_count",
        "delta_5utr_mfe",
        "delta_cds_start_window_mfe",
        "delta_3utr_mfe",
    ]

    for feat in numeric_features:
        if feat not in comp.columns:
            continue

        comp[feat] = to_num(comp[feat])
        d = comp[["delta_Delta_IF", feat]].dropna()

        if len(d) >= 3 and d[feat].nunique() > 1:
            rho, p = spearmanr(d["delta_Delta_IF"], d[feat])
            rows.append(
                {
                    "analysis_scope": "high_vs_low_within_gene",
                    "ddif_threshold": threshold,
                    "n_genes_used": len(d),
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
                    "analysis_scope": "high_vs_low_within_gene",
                    "ddif_threshold": threshold,
                    "n_genes_used": len(d),
                    "feature": f"delta_Delta_IF~{feat}",
                    "test": "spearman",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "insufficient data or no variation",
                }
            )

    group_features = [
        "kozak_strength_change",
        "nmd_likelihood_change",
        "orf_changed_high_vs_low",
        "domain_changed_high_vs_low",
    ]

    for feat in group_features:
        if feat not in comp.columns:
            continue

        tmp = comp[["delta_Delta_IF", feat]].dropna()
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
                        "analysis_scope": "high_vs_low_within_gene",
                        "ddif_threshold": threshold,
                        "n_genes_used": sum(len(x) for x in groups),
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
                        "analysis_scope": "high_vs_low_within_gene",
                        "ddif_threshold": threshold,
                        "n_genes_used": len(tmp),
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
                    "analysis_scope": "high_vs_low_within_gene",
                    "ddif_threshold": threshold,
                    "n_genes_used": len(tmp),
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
        description="Compare high-efficiency vs low-efficiency isoforms within each gene"
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--ddif-threshold",
        type=float,
        default=None,
        help=(
            "Keep genes whose high-low Delta_IF difference is greater than this threshold. "
            "Default: config.yaml params.ddif_threshold, fallback 0.2."
        ),
    )

    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger(
        "05_compare_high_low_isoforms_within_gene",
        Path(args.log) if args.log else None,
    )

    threshold = get_ddif_threshold(args, cfg)

    tables_dir = Path(cfg["outputs"]["tables_dir"])
    input_fp = tables_dir / "transcript_features.tsv"

    df = pd.read_csv(input_fp, sep="\t")

    comp_all = build_high_vs_low_table(df, logger)

    all_fp = tables_dir / "high_vs_low_isoform_comparison_all_genes.tsv"
    comp_all.to_csv(all_fp, sep="\t", index=False)

    if comp_all.empty:
        logger.warning("No genes with at least two valid isoforms were found.")
        return

    comp_filtered = comp_all[
        comp_all["delta_Delta_IF"].notna()
        & (comp_all["delta_Delta_IF"] > threshold)
    ].copy()

    comp_filtered = comp_filtered.sort_values(
        "delta_Delta_IF",
        ascending=False,
    )

    threshold_tag = safe_threshold_name(threshold)

    filtered_fp = tables_dir / f"high_vs_low_isoform_comparison.filtered.ddif_gt_{threshold_tag}.tsv"
    comp_filtered.to_csv(filtered_fp, sep="\t", index=False)

    # 为了兼容原来的 06 脚本命名，也把 high-vs-low 的结果保存成默认 pairwise 文件
    pair_fp = tables_dir / "isoform_pairwise_comparison.tsv"
    comp_filtered.to_csv(pair_fp, sep="\t", index=False)

    stats_df = add_high_low_statistics(comp_filtered, threshold)
    stats_fp = tables_dir / "statistics_summary.tsv"
    stats_df.to_csv(stats_fp, sep="\t", index=False)

    logger.info("Input transcript feature table: %s", input_fp)
    logger.info("All high-vs-low gene comparison written: %s", all_fp)
    logger.info("Filtered high-vs-low comparison written: %s", filtered_fp)
    logger.info("Default isoform_pairwise_comparison.tsv overwritten with high-vs-low results: %s", pair_fp)
    logger.info("Statistics summary written: %s", stats_fp)

    logger.info("Threshold: high_Delta_IF - low_Delta_IF > %.4g", threshold)
    logger.info("Genes with >=2 valid isoforms: %d", len(comp_all))
    logger.info("Genes retained after threshold: %d", len(comp_filtered))

    if not comp_filtered.empty:
        logger.info(
            "Top candidate: %s, high=%s, low=%s, delta_Delta_IF=%.4f",
            comp_filtered.iloc[0]["gene_name"],
            comp_filtered.iloc[0]["high_efficiency_transcript"],
            comp_filtered.iloc[0]["low_efficiency_transcript"],
            comp_filtered.iloc[0]["delta_Delta_IF"],
        )


if __name__ == "__main__":
    main()
