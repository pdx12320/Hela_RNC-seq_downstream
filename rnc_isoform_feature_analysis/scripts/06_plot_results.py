#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import add_common_cli_args, read_config, setup_logger

sns.set(style="whitegrid")


def scatter_plot(df, x, y, out, title):
    d = df[[x, y]].dropna()
    if d.empty:
        return
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=d, x=x, y=y, s=18, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for transcript features")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("06_plot_results", Path(args.log) if args.log else None)

    df = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "transcript_features.tsv", sep="\t")
    pair = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "isoform_pairwise_comparison.tsv", sep="\t")
    fig_dir = Path(cfg["outputs"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    scatter_plot(df, "five_utr_length", "Delta_IF", fig_dir / "DeltaIF_vs_5UTR_length.png", "Delta_IF vs 5'UTR length")
    scatter_plot(df, "cds_length", "Delta_IF", fig_dir / "DeltaIF_vs_CDS_length.png", "Delta_IF vs CDS length")
    scatter_plot(df, "three_utr_length", "Delta_IF", fig_dir / "DeltaIF_vs_3UTR_length.png", "Delta_IF vs 3'UTR length")
    scatter_plot(df, "uORF_count", "Delta_IF", fig_dir / "DeltaIF_vs_uORF_count.png", "Delta_IF vs uORF count")
    scatter_plot(df, "five_utr_mfe", "Delta_IF", fig_dir / "DeltaIF_vs_5UTR_MFE.png", "Delta_IF vs 5'UTR MFE")

    if not df[["kozak_strength", "Delta_IF"]].dropna().empty:
        plt.figure(figsize=(5, 4))
        sns.boxplot(data=df, x="kozak_strength", y="Delta_IF", order=["weak", "moderate", "strong"])
        plt.title("Delta_IF vs Kozak strength")
        plt.tight_layout()
        plt.savefig(fig_dir / "DeltaIF_vs_Kozak_strength.png", dpi=150)
        plt.close()

    if not pair.empty:
        top = pair.sort_values("delta_Delta_IF", key=lambda x: x.abs(), ascending=False).head(200)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=top, x="gene_name", y="delta_Delta_IF", hue="orf_changed", s=28)
        plt.xticks([], [])
        plt.title("Within-gene isoform Delta_IF differences (top 200)")
        plt.tight_layout()
        plt.savefig(fig_dir / "within_gene_isoform_deltaIF_dotplot.png", dpi=150)
        plt.close()

    logger.info("Figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
