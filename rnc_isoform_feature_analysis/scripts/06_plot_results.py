#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import add_common_cli_args, read_config, setup_logger

sns.set(style="whitegrid")


def safe_threshold_name(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def get_ddif_threshold(args, cfg) -> float:
    if getattr(args, "ddif_threshold", None) is not None:
        return float(args.ddif_threshold)
    return float(cfg.get("params", {}).get("ddif_threshold", 0.2))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def scatter_plot(df, x, y, out, title, threshold):
    if x not in df.columns or y not in df.columns:
        return

    d = df[[x, y, "gene_name"]].copy()
    d[x] = to_num(d[x])
    d[y] = to_num(d[y])
    d = d.dropna()

    if d.empty:
        return

    plt.figure(figsize=(5.8, 4.5))
    sns.scatterplot(data=d, x=x, y=y, s=28, alpha=0.75)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)

    plt.title(f"{title}\nHigh vs low isoform genes: ΔΔIF > {threshold}")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def box_plot(df, x, y, out, title, threshold):
    if x not in df.columns or y not in df.columns:
        return

    d = df[[x, y]].copy()
    d[y] = to_num(d[y])
    d = d.dropna()
    d = d[d[x].astype(str) != "NA"]

    if d.empty or d[x].nunique() < 2:
        return

    plt.figure(figsize=(6, 4.5))
    sns.boxplot(data=d, x=x, y=y)
    sns.stripplot(data=d, x=x, y=y, color="black", size=3, alpha=0.45)

    plt.xticks(rotation=35, ha="right")
    plt.title(f"{title}\nHigh vs low isoform genes: ΔΔIF > {threshold}")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-vs-low isoform plots within genes"
    )
    add_common_cli_args(parser)

    parser.add_argument(
        "--ddif-threshold",
        type=float,
        default=None,
        help=(
            "Plot genes whose high_Delta_IF - low_Delta_IF > threshold. "
            "Default: config.yaml params.ddif_threshold, fallback 0.2."
        ),
    )

    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("06_plot_results", Path(args.log) if args.log else None)

    threshold = get_ddif_threshold(args, cfg)
    threshold_tag = safe_threshold_name(threshold)

    tables_dir = Path(cfg["outputs"]["tables_dir"])
    fig_dir = Path(cfg["outputs"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 优先读取 high-vs-low 的明确输出
    high_low_path = tables_dir / f"high_vs_low_isoform_comparison.filtered.ddif_gt_{threshold_tag}.tsv"

    # 如果不存在，则读取兼容文件名
    fallback_path = tables_dir / "isoform_pairwise_comparison.tsv"

    if high_low_path.exists():
        comp_path = high_low_path
    elif fallback_path.exists():
        comp_path = fallback_path
    else:
        logger.warning("No high-vs-low comparison table found.")
        logger.warning("Expected: %s", high_low_path)
        logger.warning("Fallback: %s", fallback_path)
        return

    comp = pd.read_csv(comp_path, sep="\t")

    if comp.empty:
        logger.warning("High-vs-low comparison table is empty. No plots generated.")
        return

    required = {
        "gene_name",
        "low_efficiency_transcript",
        "high_efficiency_transcript",
        "delta_Delta_IF",
    }

    missing = required - set(comp.columns)
    if missing:
        logger.warning("This table does not look like high-vs-low output.")
        logger.warning("Missing required columns: %s", sorted(missing))
        logger.warning("Please rerun the updated 05_compare_isoforms_within_gene.py first.")
        return

    comp["delta_Delta_IF"] = to_num(comp["delta_Delta_IF"])
    comp["abs_delta_Delta_IF"] = comp["delta_Delta_IF"].abs()

    # high-low 逻辑里 delta_Delta_IF 应该是 high - low，所以用 > threshold，不再用 abs
    candidate = comp[
        comp["delta_Delta_IF"].notna()
        & (comp["delta_Delta_IF"] > threshold)
    ].copy()

    candidate = candidate.sort_values("delta_Delta_IF", ascending=False)

    candidate_out = tables_dir / f"plot_high_vs_low_candidates.ddif_gt_{threshold_tag}.tsv"
    candidate.to_csv(candidate_out, sep="\t", index=False)

    logger.info("====== High-vs-low candidate plotting ======")
    logger.info("Input table: %s", comp_path)
    logger.info("Threshold: high_Delta_IF - low_Delta_IF > %.4g", threshold)
    logger.info("Input genes: %d", len(comp))
    logger.info("Candidate genes used for plotting: %d", len(candidate))
    logger.info("Candidate table used for plotting written: %s", candidate_out)
    logger.info("===========================================")

    if candidate.empty:
        logger.warning("No high-vs-low candidate genes after filtering. No plots generated.")
        return

    # 1. high-low ΔΔIF dotplot
    top = candidate.head(200).copy()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=top,
        x=np.arange(len(top)),
        y="delta_Delta_IF",
        hue="orf_changed_high_vs_low" if "orf_changed_high_vs_low" in top.columns else None,
        s=34,
        alpha=0.85,
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Candidate genes ranked by ΔΔIF")
    plt.ylabel("ΔΔIF = high_Delta_IF - low_Delta_IF")
    plt.title(
        f"High-efficiency vs low-efficiency isoforms within genes\n"
        f"ΔΔIF > {threshold}, top {len(top)} genes"
    )
    plt.tight_layout()
    plt.savefig(fig_dir / f"high_vs_low_delta_Delta_IF_dotplot.ddif_gt_{threshold_tag}.png", dpi=180)
    plt.close()

    # 2. numeric feature differences
    scatter_plot(
        candidate,
        "delta_5utr_length",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_delta5UTR_length_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "Δ5'UTR length vs ΔΔIF",
        threshold,
    )

    scatter_plot(
        candidate,
        "delta_cds_length",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_deltaCDS_length_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "ΔCDS length vs ΔΔIF",
        threshold,
    )

    scatter_plot(
        candidate,
        "delta_3utr_length",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_delta3UTR_length_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "Δ3'UTR length vs ΔΔIF",
        threshold,
    )

    scatter_plot(
        candidate,
        "delta_uORF_count",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_delta_uORF_count_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "ΔuORF count vs ΔΔIF",
        threshold,
    )

    # 只画你现在真正计算过的 5'UTR RNAfold
    scatter_plot(
        candidate,
        "delta_5utr_mfe",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_delta5UTR_MFE_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "Δ5'UTR MFE vs ΔΔIF",
        threshold,
    )

    # 3. categorical feature changes
    box_plot(
        candidate,
        "kozak_strength_change",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_delta_Delta_IF_by_Kozak_change.ddif_gt_{threshold_tag}.png",
        "ΔΔIF by Kozak strength change",
        threshold,
    )

    box_plot(
        candidate,
        "nmd_likelihood_change",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_delta_Delta_IF_by_NMD_change.ddif_gt_{threshold_tag}.png",
        "ΔΔIF by NMD likelihood change",
        threshold,
    )

    box_plot(
        candidate,
        "orf_changed_high_vs_low",
        "delta_Delta_IF",
        fig_dir / f"high_vs_low_delta_Delta_IF_by_ORF_changed.ddif_gt_{threshold_tag}.png",
        "ΔΔIF by ORF changed",
        threshold,
    )

    # 4. feature change summary barplot
    summary = {}

    if "orf_changed_high_vs_low" in candidate.columns:
        summary["ORF changed"] = (
            candidate["orf_changed_high_vs_low"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .sum()
        )

    if "kozak_strength_change" in candidate.columns:
        summary["Kozak changed"] = (
            candidate["kozak_strength_change"].astype(str).ne("same")
            & candidate["kozak_strength_change"].astype(str).ne("NA")
        ).sum()

    if "nmd_likelihood_change" in candidate.columns:
        summary["NMD changed"] = (
            candidate["nmd_likelihood_change"].astype(str).ne("same")
            & candidate["nmd_likelihood_change"].astype(str).ne("NA")
        ).sum()

    if "delta_5utr_mfe" in candidate.columns:
        valid_mfe = pd.to_numeric(candidate["delta_5utr_mfe"], errors="coerce").notna().sum()
        summary["5'UTR MFE available"] = valid_mfe

    if summary:
        summary_df = pd.DataFrame(
            {
                "feature_change": list(summary.keys()),
                "count": list(summary.values()),
            }
        )

        plt.figure(figsize=(6, 4.5))
        sns.barplot(data=summary_df, x="feature_change", y="count")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Number of candidate genes")
        plt.title(f"Feature changes in high-vs-low isoform candidates\nΔΔIF > {threshold}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"high_vs_low_feature_change_barplot.ddif_gt_{threshold_tag}.png", dpi=180)
        plt.close()

    logger.info("High-vs-low candidate figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
