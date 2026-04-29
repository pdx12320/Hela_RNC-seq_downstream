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


def candidate_scatter_plot(pair_df, x, y, out, title, threshold):
    if x not in pair_df.columns or y not in pair_df.columns:
        return

    d = pair_df[[x, y, "gene_name"]].copy()
    d[x] = to_num(d[x])
    d[y] = to_num(d[y])
    d = d.dropna()

    if d.empty:
        return

    plt.figure(figsize=(5.8, 4.5))
    sns.scatterplot(
        data=d,
        x=x,
        y=y,
        s=28,
        alpha=0.75,
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)

    plt.title(f"{title}\nCandidate pairs: |ΔΔIF| > {threshold}")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def candidate_boxplot(pair_df, x, y, out, title, threshold):
    if x not in pair_df.columns or y not in pair_df.columns:
        return

    d = pair_df[[x, y]].copy()
    d[y] = to_num(d[y])
    d = d.dropna()
    d = d[d[x].astype(str) != "NA"]

    if d.empty or d[x].nunique() < 2:
        return

    plt.figure(figsize=(6, 4.5))
    sns.boxplot(data=d, x=x, y=y)
    sns.stripplot(data=d, x=x, y=y, color="black", size=3, alpha=0.45)

    plt.xticks(rotation=35, ha="right")
    plt.title(f"{title}\nCandidate pairs: |ΔΔIF| > {threshold}")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate isoform-pair plots filtered by abs(delta_Delta_IF)"
    )
    add_common_cli_args(parser)

    parser.add_argument(
        "--ddif-threshold",
        type=float,
        default=None,
        help=(
            "Plot candidate isoform pairs with abs(delta_Delta_IF) > threshold. "
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

    pair_path = tables_dir / "isoform_pairwise_comparison.tsv"

    if not pair_path.exists():
        logger.warning("Pairwise comparison file not found: %s", pair_path)
        return

    pair = pd.read_csv(pair_path, sep="\t")

    if pair.empty:
        logger.warning("Pairwise comparison table is empty. No candidate plots generated.")
        return

    required = {"gene_name", "ref_transcript", "query_transcript", "delta_Delta_IF"}
    missing = required - set(pair.columns)

    if missing:
        logger.warning("Missing required columns in pairwise table: %s", sorted(missing))
        return

    pair["delta_Delta_IF"] = to_num(pair["delta_Delta_IF"])
    pair["abs_delta_Delta_IF"] = pair["delta_Delta_IF"].abs()

    # 关键：这里再次筛一遍，防止 05 没有正确筛选
    candidate = pair[
        (pair["ref_transcript"].astype(str) != pair["query_transcript"].astype(str))
        & pair["delta_Delta_IF"].notna()
        & (pair["abs_delta_Delta_IF"] > threshold)
    ].copy()

    candidate = candidate.sort_values("abs_delta_Delta_IF", ascending=False)

    candidate_out = tables_dir / f"plot_candidates.ddif_gt_{threshold_tag}.tsv"
    candidate.to_csv(candidate_out, sep="\t", index=False)

    logger.info("====== Candidate plotting ======")
    logger.info("Threshold: abs(delta_Delta_IF) > %.4g", threshold)
    logger.info("Input pairwise rows: %d", len(pair))
    logger.info("Candidate pairwise rows used for plotting: %d", len(candidate))
    logger.info(
        "Candidate genes used for plotting: %d",
        candidate["gene_name"].nunique() if not candidate.empty else 0,
    )
    logger.info("Candidate table used for plotting written: %s", candidate_out)
    logger.info("===============================")

    if candidate.empty:
        logger.warning("No candidate pairs after filtering. No plots generated.")
        return

    # 1. candidate ΔΔIF dotplot
    top = candidate.head(200).copy()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=top,
        x=np.arange(len(top)),
        y="delta_Delta_IF",
        hue="orf_changed" if "orf_changed" in top.columns else None,
        s=34,
        alpha=0.85,
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Candidate isoform pairs ranked by |ΔΔIF|")
    plt.ylabel("ΔΔIF")
    plt.title(
        f"Candidate isoform ΔΔIF dotplot\n"
        f"|ΔΔIF| > {threshold}, top {len(top)} pairs"
    )
    plt.tight_layout()
    plt.savefig(fig_dir / f"candidate_delta_Delta_IF_dotplot.ddif_gt_{threshold_tag}.png", dpi=180)
    plt.close()

    # 2. delta feature vs delta_Delta_IF
    candidate_scatter_plot(
        candidate,
        "delta_5utr_length",
        "delta_Delta_IF",
        fig_dir / f"candidate_delta5UTR_length_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "Δ5'UTR length vs ΔΔIF",
        threshold,
    )

    candidate_scatter_plot(
        candidate,
        "delta_cds_length",
        "delta_Delta_IF",
        fig_dir / f"candidate_deltaCDS_length_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "ΔCDS length vs ΔΔIF",
        threshold,
    )

    candidate_scatter_plot(
        candidate,
        "delta_3utr_length",
        "delta_Delta_IF",
        fig_dir / f"candidate_delta3UTR_length_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "Δ3'UTR length vs ΔΔIF",
        threshold,
    )

    candidate_scatter_plot(
        candidate,
        "delta_uORF_count",
        "delta_Delta_IF",
        fig_dir / f"candidate_delta_uORF_count_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "ΔuORF count vs ΔΔIF",
        threshold,
    )

    # 你现在只算了 5'UTR RNAfold，所以只画 delta_5utr_mfe
    candidate_scatter_plot(
        candidate,
        "delta_5utr_mfe",
        "delta_Delta_IF",
        fig_dir / f"candidate_delta5UTR_MFE_vs_delta_Delta_IF.ddif_gt_{threshold_tag}.png",
        "Δ5'UTR MFE vs ΔΔIF",
        threshold,
    )

    # 3. categorical feature change plots
    candidate_boxplot(
        candidate,
        "orf_changed",
        "delta_Delta_IF",
        fig_dir / f"candidate_delta_Delta_IF_by_ORF_changed.ddif_gt_{threshold_tag}.png",
        "ΔΔIF by ORF changed",
        threshold,
    )

    candidate_boxplot(
        candidate,
        "kozak_strength_change",
        "delta_Delta_IF",
        fig_dir / f"candidate_delta_Delta_IF_by_Kozak_change.ddif_gt_{threshold_tag}.png",
        "ΔΔIF by Kozak strength change",
        threshold,
    )

    candidate_boxplot(
        candidate,
        "nmd_likelihood_change",
        "delta_Delta_IF",
        fig_dir / f"candidate_delta_Delta_IF_by_NMD_change.ddif_gt_{threshold_tag}.png",
        "ΔΔIF by NMD likelihood change",
        threshold,
    )

    # 4. feature change summary barplot
    summary = {}

    if "orf_changed" in candidate.columns:
        summary["ORF changed"] = (
            candidate["orf_changed"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
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

    if "polyA_signal_change" in candidate.columns:
        summary["polyA changed"] = (
            candidate["polyA_signal_change"].astype(str).ne("same")
            & candidate["polyA_signal_change"].astype(str).ne("NA")
        ).sum()

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
        plt.ylabel("Number of candidate pairs")
        plt.title(f"Feature changes among candidate isoform pairs\n|ΔΔIF| > {threshold}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"candidate_feature_change_barplot.ddif_gt_{threshold_tag}.png", dpi=180)
        plt.close()

    logger.info("Candidate figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
