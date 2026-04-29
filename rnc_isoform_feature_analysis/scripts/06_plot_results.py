#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import ensure_dirs, load_config, setup_logger


def scatter(df, x, y, out, title: str | None = None):
    d = df[[x, y]].dropna()
    if d.empty:
        return
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=d, x=x, y=y, s=20, alpha=0.7)
    sns.regplot(data=d, x=x, y=y, scatter=False, color="red", line_kws={"linewidth": 1})
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot summary figures")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ddif-threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ddif_threshold = (
        args.ddif_threshold
        if args.ddif_threshold is not None
        else float(cfg.get("params", {}).get("ddif_threshold", 0.0))
    )

    ensure_dirs([cfg["output"]["figures_dir"]])
    logger = setup_logger("06_plot", os.path.join(cfg["output"]["tables_dir"], "06_plot_results.log"))

    title_suffix = f"|ΔΔIF| >= {ddif_threshold:g}"
    figures_dir = Path(cfg["output"]["figures_dir"])

    # Keep original transcript-level figures
    df = pd.read_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features.tsv"), sep="\t")
    pairs = [
        ("five_utr_length", "Delta_IF", "DeltaIF_vs_5UTR_length.png"),
        ("cds_length", "Delta_IF", "DeltaIF_vs_CDS_length.png"),
        ("three_utr_length", "Delta_IF", "DeltaIF_vs_3UTR_length.png"),
        ("uORF_count", "Delta_IF", "DeltaIF_vs_uORF_count.png"),
        ("five_utr_mfe", "Delta_IF", "DeltaIF_vs_5UTR_MFE.png"),
        ("cds_start_window_mfe", "Delta_IF", "DeltaIF_vs_CDS_start_MFE.png"),
        ("three_utr_mfe", "Delta_IF", "DeltaIF_vs_3UTR_MFE.png"),
    ]
    for x, y, fn in pairs:
        if x in df.columns and y in df.columns:
            scatter(df, x, y, figures_dir / fn)

    if "kozak_strength" in df.columns:
        plt.figure(figsize=(5, 4))
        sns.boxplot(data=df, x="kozak_strength", y="Delta_IF", order=["weak", "moderate", "strong"])
        plt.tight_layout()
        plt.savefig(figures_dir / "DeltaIF_vs_Kozak_strength.png", dpi=200)
        plt.close()

    # Candidate pair-focused plotting
    pair_path = Path(cfg["output"]["tables_dir"]) / "isoform_pairwise_comparison.tsv"
    pair = pd.read_csv(pair_path, sep="\t") if pair_path.exists() else pd.DataFrame()
    if pair.empty:
        logger.warning("No candidate isoform pairs after filtering (|ΔΔIF| >= %s). Skip candidate plots.", ddif_threshold)
        return

    pair["delta_Delta_IF"] = pd.to_numeric(pair.get("delta_Delta_IF"), errors="coerce")
    pair = pair[pair["delta_Delta_IF"].notna()].copy()
    if pair.empty:
        logger.warning("Candidate pair table has no numeric delta_Delta_IF values. Skip candidate plots.")
        return

    # a) candidate delta_Delta_IF dotplot
    top_genes = pair.groupby("gene_name")["query_transcript"].nunique().sort_values(ascending=False).head(30).index
    d = pair[pair["gene_name"].isin(top_genes)].copy()
    if not d.empty:
        plt.figure(figsize=(10, 5))
        sns.stripplot(data=d, x="gene_name", y="delta_Delta_IF", size=4, alpha=0.7)
        plt.xticks(rotation=90)
        plt.title(f"Candidate isoform pairs ({title_suffix})")
        plt.tight_layout()
        plt.savefig(figures_dir / "candidate_delta_Delta_IF_dotplot.png", dpi=200)
        plt.close()

    # b) candidate delta features vs delta_Delta_IF
    feature_cols = [c for c in ["delta_5utr_length", "delta_cds_length", "delta_3utr_length", "delta_uORF_count"] if c in pair.columns]
    if feature_cols:
        tmp = pair[["delta_Delta_IF"] + feature_cols].copy()
        for col in feature_cols:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        long_df = tmp.melt(id_vars="delta_Delta_IF", value_vars=feature_cols, var_name="feature", value_name="delta_feature")
        long_df = long_df.dropna()
        if not long_df.empty:
            g = sns.FacetGrid(long_df, col="feature", col_wrap=2, height=3.2, sharey=False)
            g.map_dataframe(sns.scatterplot, x="delta_feature", y="delta_Delta_IF", s=15, alpha=0.7)
            g.map_dataframe(sns.regplot, x="delta_feature", y="delta_Delta_IF", scatter=False, color="red", line_kws={"linewidth": 1})
            g.fig.suptitle(f"Candidate delta features vs delta_Delta_IF ({title_suffix})", y=1.02)
            g.tight_layout()
            g.savefig(figures_dir / "candidate_delta_features_vs_delta_Delta_IF.png", dpi=200)
            plt.close(g.fig)

    # c) candidate feature change barplot
    frac_rows = []
    if "orf_changed" in pair.columns:
        frac_rows.append({"feature_change": "ORF changed", "fraction": pair["orf_changed"].astype(str).str.lower().isin(["true", "1", "yes"]).mean()})
    if "nmd_likelihood_change" in pair.columns:
        frac_rows.append({"feature_change": "NMD changed", "fraction": pair["nmd_likelihood_change"].astype(str).str.split("->").apply(lambda x: len(x) == 2 and x[0] != x[1]).mean()})
    if "kozak_strength_change" in pair.columns:
        frac_rows.append({"feature_change": "Kozak changed", "fraction": pair["kozak_strength_change"].astype(str).str.split("->").apply(lambda x: len(x) == 2 and x[0] != x[1]).mean()})

    if frac_rows:
        frac_df = pd.DataFrame(frac_rows)
        plt.figure(figsize=(6, 4))
        sns.barplot(data=frac_df, x="feature_change", y="fraction")
        plt.ylim(0, 1)
        plt.title(f"Candidate feature-change proportion ({title_suffix})")
        plt.tight_layout()
        plt.savefig(figures_dir / "candidate_feature_change_barplot.png", dpi=200)
        plt.close()

    # keep old file for compatibility, now derived from candidate pairs
    if not d.empty:
        plt.figure(figsize=(10, 5))
        sns.stripplot(data=d, x="gene_name", y="delta_Delta_IF", size=4, alpha=0.7)
        plt.xticks(rotation=90)
        plt.title(f"Within-gene delta_Delta_IF ({title_suffix})")
        plt.tight_layout()
        plt.savefig(figures_dir / "within_gene_isoform_deltaIF_dotplot.png", dpi=200)
        plt.close()

    logger.info("Figures saved to %s", cfg["output"]["figures_dir"])


if __name__ == "__main__":
    main()
