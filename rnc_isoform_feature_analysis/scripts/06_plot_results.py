#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, wilcoxon

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


def p_to_star(p) -> str:
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def fmt_p(p) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def is_true_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def make_change_count_summary(candidate: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many candidate genes show feature changed vs unchanged.

    p value:
    binomial test comparing changed vs unchanged counts under p=0.5.
    This is a count-enrichment test, not a mechanistic causal test.
    """
    rows = []

    numeric_change_features = [
        ("delta_5utr_length", "5'UTR length changed"),
        ("delta_cds_length", "CDS length changed"),
        ("delta_3utr_length", "3'UTR length changed"),
        ("delta_uORF_count", "uORF count changed"),
        ("delta_5utr_mfe", "5'UTR MFE changed"),
    ]

    for col, label in numeric_change_features:
        if col not in candidate.columns:
            continue

        x = to_num(candidate[col]).dropna()
        if len(x) == 0:
            continue

        changed = x != 0
        n_changed = int(changed.sum())
        n_unchanged = int((~changed).sum())
        n_valid = int(len(x))

        p = np.nan
        if n_valid >= 5:
            p = binomtest(
                k=min(n_changed, n_unchanged),
                n=n_valid,
                p=0.5,
                alternative="two-sided",
            ).pvalue

        rows.append(
            {
                "feature": label,
                "changed": n_changed,
                "unchanged": n_unchanged,
                "n_valid": n_valid,
                "percent_changed": n_changed / n_valid * 100,
                "binomial_p_changed_vs_unchanged": p,
                "significance": p_to_star(p),
            }
        )

    if "orf_changed_high_vs_low" in candidate.columns:
        s = candidate["orf_changed_high_vs_low"].dropna()
        if len(s) > 0:
            changed = is_true_series(s)
            n_changed = int(changed.sum())
            n_unchanged = int((~changed).sum())
            n_valid = int(len(s))

            p = np.nan
            if n_valid >= 5:
                p = binomtest(
                    k=min(n_changed, n_unchanged),
                    n=n_valid,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue

            rows.append(
                {
                    "feature": "ORF changed",
                    "changed": n_changed,
                    "unchanged": n_unchanged,
                    "n_valid": n_valid,
                    "percent_changed": n_changed / n_valid * 100,
                    "binomial_p_changed_vs_unchanged": p,
                    "significance": p_to_star(p),
                }
            )

    categorical_change_features = [
        ("kozak_strength_change", "Kozak changed"),
        ("nmd_likelihood_change", "NMD changed"),
    ]

    for col, label in categorical_change_features:
        if col not in candidate.columns:
            continue

        s = candidate[col].dropna().astype(str)
        s = s[(s != "NA") & (s != "nan")]
        if len(s) == 0:
            continue

        changed = s != "same"
        n_changed = int(changed.sum())
        n_unchanged = int((~changed).sum())
        n_valid = int(len(s))

        p = np.nan
        if n_valid >= 5:
            p = binomtest(
                k=min(n_changed, n_unchanged),
                n=n_valid,
                p=0.5,
                alternative="two-sided",
            ).pvalue

        rows.append(
            {
                "feature": label,
                "changed": n_changed,
                "unchanged": n_unchanged,
                "n_valid": n_valid,
                "percent_changed": n_changed / n_valid * 100,
                "binomial_p_changed_vs_unchanged": p,
                "significance": p_to_star(p),
            }
        )

    return pd.DataFrame(rows)


def make_direction_summary(candidate: pd.DataFrame) -> pd.DataFrame:
    """
    For numeric high-low delta features:
    delta = high isoform value - low isoform value

    negative: high < low
    zero: same
    positive: high > low

    p values:
    - Wilcoxon signed-rank test against 0
    - sign test by binomial test among non-zero deltas
    """
    rows = []

    features = [
        ("delta_5utr_length", "5'UTR length"),
        ("delta_cds_length", "CDS length"),
        ("delta_3utr_length", "3'UTR length"),
        ("delta_uORF_count", "uORF count"),
        ("delta_5utr_mfe", "5'UTR MFE"),
    ]

    for col, label in features:
        if col not in candidate.columns:
            continue

        x = to_num(candidate[col]).dropna()
        if len(x) == 0:
            continue

        n_negative = int((x < 0).sum())
        n_zero = int((x == 0).sum())
        n_positive = int((x > 0).sum())
        n_valid = int(len(x))

        median_delta = float(x.median())
        mean_delta = float(x.mean())

        x_nonzero = x[x != 0]

        wilcoxon_p_two_sided = np.nan
        wilcoxon_p_less = np.nan
        wilcoxon_p_greater = np.nan

        if len(x_nonzero) >= 5:
            try:
                wilcoxon_p_two_sided = wilcoxon(
                    x_nonzero,
                    alternative="two-sided",
                ).pvalue
                wilcoxon_p_less = wilcoxon(
                    x_nonzero,
                    alternative="less",
                ).pvalue
                wilcoxon_p_greater = wilcoxon(
                    x_nonzero,
                    alternative="greater",
                ).pvalue
            except Exception:
                pass

        sign_test_p = np.nan
        if n_negative + n_positive >= 5:
            sign_test_p = binomtest(
                k=min(n_negative, n_positive),
                n=n_negative + n_positive,
                p=0.5,
                alternative="two-sided",
            ).pvalue

        if median_delta < 0:
            direction = "high < low"
            directional_p = wilcoxon_p_less
        elif median_delta > 0:
            direction = "high > low"
            directional_p = wilcoxon_p_greater
        else:
            direction = "no median shift"
            directional_p = wilcoxon_p_two_sided

        rows.append(
            {
                "feature": label,
                "column": col,
                "n_valid": n_valid,
                "n_high_lt_low": n_negative,
                "n_same": n_zero,
                "n_high_gt_low": n_positive,
                "median_delta": median_delta,
                "mean_delta": mean_delta,
                "direction": direction,
                "wilcoxon_p_two_sided": wilcoxon_p_two_sided,
                "wilcoxon_p_less": wilcoxon_p_less,
                "wilcoxon_p_greater": wilcoxon_p_greater,
                "directional_wilcoxon_p": directional_p,
                "sign_test_p": sign_test_p,
                "directional_significance": p_to_star(directional_p),
                "sign_test_significance": p_to_star(sign_test_p),
            }
        )

    return pd.DataFrame(rows)


def plot_change_counts(summary_df: pd.DataFrame, out: Path, threshold: float):
    if summary_df.empty:
        return

    plot_df = summary_df[["feature", "changed", "unchanged"]].copy()
    plot_df = plot_df.set_index("feature")

    ax = plot_df[["changed", "unchanged"]].plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
    )

    ax.set_ylabel("Number of candidate genes")
    ax.set_xlabel("")
    ax.set_title(
        f"Feature changed vs unchanged in high-vs-low isoform candidates\n"
        f"ΔΔIF > {threshold}"
    )
    plt.xticks(rotation=35, ha="right")

    max_total = max(plot_df.sum(axis=1).max(), 1)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        total = row["n_valid"]
        p = row["binomial_p_changed_vs_unchanged"]
        star = row["significance"]

        ax.text(
            i,
            total + max_total * 0.03,
            f"n={int(total)}\np={fmt_p(p)}\n{star}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max_total * 1.25)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_direction_counts(direction_df: pd.DataFrame, out: Path, threshold: float):
    if direction_df.empty:
        return

    plot_df = direction_df[
        ["feature", "n_high_lt_low", "n_same", "n_high_gt_low"]
    ].copy()

    plot_df = plot_df.set_index("feature")
    plot_df = plot_df.rename(
        columns={
            "n_high_lt_low": "high < low",
            "n_same": "same",
            "n_high_gt_low": "high > low",
        }
    )

    ax = plot_df[["high < low", "same", "high > low"]].plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
    )

    ax.set_ylabel("Number of candidate genes")
    ax.set_xlabel("")
    ax.set_title(
        f"Direction of feature differences: high isoform - low isoform\n"
        f"ΔΔIF > {threshold}"
    )
    plt.xticks(rotation=35, ha="right")

    max_total = max(plot_df.sum(axis=1).max(), 1)

    for i, (_, row) in enumerate(direction_df.iterrows()):
        total = row["n_valid"]
        p = row["directional_wilcoxon_p"]
        star = row["directional_significance"]
        median_delta = row["median_delta"]

        ax.text(
            i,
            total + max_total * 0.03,
            f"median={median_delta:.1f}\np={fmt_p(p)}\n{star}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max_total * 1.28)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_significance_bar(direction_df: pd.DataFrame, out: Path, threshold: float):
    """
    Barplot of -log10 directional Wilcoxon p values.
    This is a statistical summary plot, not a scatter plot.
    """
    if direction_df.empty:
        return

    d = direction_df[["feature", "directional_wilcoxon_p", "direction"]].copy()
    d["directional_wilcoxon_p"] = to_num(d["directional_wilcoxon_p"])
    d = d.dropna()

    if d.empty:
        return

    d["minus_log10_p"] = -np.log10(d["directional_wilcoxon_p"].clip(lower=1e-300))

    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(data=d, x="feature", y="minus_log10_p")

    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.set_ylabel("-log10 directional Wilcoxon p")
    ax.set_xlabel("")
    ax.set_title(
        f"Significance of high-vs-low feature shifts\n"
        f"ΔΔIF > {threshold}"
    )

    plt.xticks(rotation=35, ha="right")

    for i, (_, row) in enumerate(d.iterrows()):
        p = row["directional_wilcoxon_p"]
        ax.text(
            i,
            row["minus_log10_p"] + 0.05,
            f"{p_to_star(p)}\n{row['direction']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate count/statistical summary plots for high-vs-low isoform analysis"
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

    high_low_path = tables_dir / f"high_vs_low_isoform_comparison.filtered.ddif_gt_{threshold_tag}.tsv"
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

    candidate = comp[
        comp["delta_Delta_IF"].notna()
        & (comp["delta_Delta_IF"] > threshold)
    ].copy()

    candidate = candidate.sort_values("delta_Delta_IF", ascending=False)

    if candidate.empty:
        logger.warning("No high-vs-low candidate genes after filtering. No plots generated.")
        return

    candidate_out = tables_dir / f"plot_high_vs_low_count_candidates.ddif_gt_{threshold_tag}.tsv"
    candidate.to_csv(candidate_out, sep="\t", index=False)

    change_summary = make_change_count_summary(candidate)
    change_summary_fp = tables_dir / f"high_vs_low_feature_change_count_summary.ddif_gt_{threshold_tag}.tsv"
    change_summary.to_csv(change_summary_fp, sep="\t", index=False)

    direction_summary = make_direction_summary(candidate)
    direction_summary_fp = tables_dir / f"high_vs_low_direction_count_significance_summary.ddif_gt_{threshold_tag}.tsv"
    direction_summary.to_csv(direction_summary_fp, sep="\t", index=False)

    logger.info("====== High-vs-low count/stat plotting ======")
    logger.info("Input table: %s", comp_path)
    logger.info("Threshold: high_Delta_IF - low_Delta_IF > %.4g", threshold)
    logger.info("Candidate genes used for plotting: %d", len(candidate))
    logger.info("Candidate table written: %s", candidate_out)
    logger.info("Change count summary written: %s", change_summary_fp)
    logger.info("Direction/significance summary written: %s", direction_summary_fp)
    logger.info("============================================")

    plot_change_counts(
        change_summary,
        fig_dir / f"high_vs_low_feature_changed_counts_with_significance.ddif_gt_{threshold_tag}.png",
        threshold,
    )

    plot_direction_counts(
        direction_summary,
        fig_dir / f"high_vs_low_feature_direction_counts_with_significance.ddif_gt_{threshold_tag}.png",
        threshold,
    )

    plot_significance_bar(
        direction_summary,
        fig_dir / f"high_vs_low_feature_shift_significance.ddif_gt_{threshold_tag}.png",
        threshold,
    )

    logger.info("Count/statistical figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
