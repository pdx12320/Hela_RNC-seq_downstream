#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from utils import load_config, setup_logger


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_delta(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return a - b


def sanitize_threshold_for_filename(threshold: float) -> str:
    txt = f"{threshold:g}"
    return txt.replace("-", "neg").replace(".", "p")


def compare_group(g: pd.DataFrame) -> pd.DataFrame:
    ref_id = g["reference_transcript"].iloc[0]
    ref = g[g["transcript_id"] == ref_id]
    if ref.empty:
        ref = g.iloc[[0]]
    ref = ref.iloc[0]

    rows = []
    for _, r in g.iterrows():
        rows.append(
            {
                "gene_name": r.get("gene_name", "NA"),
                "ref_transcript": ref_id,
                "query_transcript": r["transcript_id"],
                "delta_log2FC": safe_delta(r.get("log2FC", np.nan), ref.get("log2FC", np.nan)),
                "delta_Delta_IF": safe_delta(r.get("Delta_IF", np.nan), ref.get("Delta_IF", np.nan)),
                "delta_5utr_length": safe_delta(r.get("five_utr_length", np.nan), ref.get("five_utr_length", np.nan)),
                "delta_cds_length": safe_delta(r.get("cds_length", np.nan), ref.get("cds_length", np.nan)),
                "delta_3utr_length": safe_delta(r.get("three_utr_length", np.nan), ref.get("three_utr_length", np.nan)),
                "orf_changed": r.get("orf_changed_vs_reference", np.nan),
                "domain_changed": r.get("domain_changed_vs_reference", np.nan),
                "delta_uORF_count": safe_delta(r.get("uORF_count", np.nan), ref.get("uORF_count", np.nan)),
                "kozak_strength_change": f"{ref.get('kozak_strength', 'NA')}->{r.get('kozak_strength', 'NA')}",
                "delta_miRNA_site_count": safe_delta(r.get("miRNA_site_count", np.nan), ref.get("miRNA_site_count", np.nan)),
                "polyA_signal_change": f"{ref.get('nearest_polyA_signal_to_3end', 'NA')}->{r.get('nearest_polyA_signal_to_3end', 'NA')}",
                "delta_5utr_mfe": safe_delta(r.get("five_utr_mfe", np.nan), ref.get("five_utr_mfe", np.nan)),
                "delta_cds_start_window_mfe": safe_delta(r.get("cds_start_window_mfe", np.nan), ref.get("cds_start_window_mfe", np.nan)),
                "delta_3utr_mfe": safe_delta(r.get("three_utr_mfe", np.nan), ref.get("three_utr_mfe", np.nan)),
                "nmd_likelihood_change": f"{ref.get('nmd_likelihood', 'NA')}->{r.get('nmd_likelihood', 'NA')}",
            }
        )
    return pd.DataFrame(rows)


def add_stat_row(stats_rows, scope, threshold, n_pairs, feature, test, statistic, p_value, note):
    stats_rows.append(
        {
            "analysis_scope": scope,
            "ddif_threshold": threshold,
            "n_pairs_used": n_pairs,
            "feature": feature,
            "test": test,
            "statistic": statistic,
            "p_value": p_value,
            "note": note,
        }
    )


def summarize_stats_all(df: pd.DataFrame) -> pd.DataFrame:
    stats_rows = []
    numeric_features = [
        "transcript_length",
        "five_utr_length",
        "cds_length",
        "three_utr_length",
        "uORF_count",
        "miRNA_site_count",
        "five_utr_mfe",
        "cds_start_window_mfe",
        "three_utr_mfe",
    ]
    for f in numeric_features:
        if f in df.columns:
            x = to_num(df[f])
            y = to_num(df["Delta_IF"])
            m = x.notna() & y.notna()
            if m.sum() >= 3:
                rho, p = spearmanr(x[m], y[m])
                add_stat_row(stats_rows, "transcript_all", np.nan, int(m.sum()), f"Delta_IF~{f}", "spearman", rho, p, "")
            else:
                add_stat_row(stats_rows, "transcript_all", np.nan, int(m.sum()), f"Delta_IF~{f}", "spearman", np.nan, np.nan, "insufficient points")

    return pd.DataFrame(stats_rows)


def _bool_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    return s.isin(["true", "1", "yes"])


def summarize_candidate_pair_stats(candidate_pair_df: pd.DataFrame, ddif_threshold: float) -> pd.DataFrame:
    stats_rows = []
    n_pairs = len(candidate_pair_df)

    numeric_features = [
        "delta_5utr_length",
        "delta_cds_length",
        "delta_3utr_length",
        "delta_uORF_count",
        "delta_miRNA_site_count",
        "delta_5utr_mfe",
        "delta_cds_start_window_mfe",
        "delta_3utr_mfe",
    ]

    if n_pairs < 3:
        for feat in numeric_features:
            if feat in candidate_pair_df.columns:
                add_stat_row(
                    stats_rows,
                    "candidate_pair",
                    ddif_threshold,
                    n_pairs,
                    f"delta_Delta_IF~{feat}",
                    "spearman",
                    np.nan,
                    np.nan,
                    "insufficient candidate pairs",
                )
        add_stat_row(stats_rows, "candidate_pair", ddif_threshold, n_pairs, "orf_changed", "wilcoxon_rank_sum", np.nan, np.nan, "insufficient candidate pairs")
        add_stat_row(stats_rows, "candidate_pair", ddif_threshold, n_pairs, "domain_changed", "wilcoxon_rank_sum", np.nan, np.nan, "insufficient candidate pairs")
        add_stat_row(stats_rows, "candidate_pair", ddif_threshold, n_pairs, "nmd_likelihood_change", "kruskal", np.nan, np.nan, "insufficient candidate pairs")
        add_stat_row(stats_rows, "candidate_pair", ddif_threshold, n_pairs, "kozak_strength_change", "kruskal", np.nan, np.nan, "insufficient candidate pairs")
        return pd.DataFrame(stats_rows)

    y = to_num(candidate_pair_df["delta_Delta_IF"])
    for feat in numeric_features:
        if feat not in candidate_pair_df.columns:
            continue
        x = to_num(candidate_pair_df[feat])
        m = x.notna() & y.notna()
        if m.sum() >= 3:
            rho, p = spearmanr(x[m], y[m])
            add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(m.sum()), f"delta_Delta_IF~{feat}", "spearman", rho, p, "")
        else:
            add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(m.sum()), f"delta_Delta_IF~{feat}", "spearman", np.nan, np.nan, "insufficient valid pairs")

    if "orf_changed" in candidate_pair_df.columns:
        tmask = _bool_mask(candidate_pair_df["orf_changed"])
        a = y[tmask & y.notna()]
        b = y[(~tmask) & y.notna()]
        if len(a) > 0 and len(b) > 0:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(len(a) + len(b)), "delta_Delta_IF~orf_changed", "wilcoxon_rank_sum", stat, p, "")
        else:
            add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(len(a) + len(b)), "delta_Delta_IF~orf_changed", "wilcoxon_rank_sum", np.nan, np.nan, "insufficient groups")

    if "domain_changed" in candidate_pair_df.columns:
        valid = candidate_pair_df["domain_changed"].notna()
        tmask = _bool_mask(candidate_pair_df.loc[valid, "domain_changed"])
        yy = y[valid]
        a = yy[tmask & yy.notna()]
        b = yy[(~tmask) & yy.notna()]
        if len(a) > 0 and len(b) > 0:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(len(a) + len(b)), "delta_Delta_IF~domain_changed", "wilcoxon_rank_sum", stat, p, "")
        else:
            add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(len(a) + len(b)), "delta_Delta_IF~domain_changed", "wilcoxon_rank_sum", np.nan, np.nan, "insufficient groups")

    for col, fname in [("nmd_likelihood_change", "delta_Delta_IF~nmd_likelihood_change"), ("kozak_strength_change", "delta_Delta_IF~kozak_strength_change")]:
        if col in candidate_pair_df.columns:
            tmp = pd.DataFrame({"group": candidate_pair_df[col].astype(str), "y": y}).dropna()
            groups = [g["y"].values for _, g in tmp.groupby("group") if len(g) >= 2]
            if len(groups) >= 2:
                stat, p = kruskal(*groups)
                add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(len(tmp)), fname, "kruskal", stat, p, "")
            else:
                add_stat_row(stats_rows, "candidate_pair", ddif_threshold, int(len(tmp)), fname, "kruskal", np.nan, np.nan, "insufficient groups")

    return pd.DataFrame(stats_rows)


def main():
    ap = argparse.ArgumentParser(description="Isoform comparison and statistics")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ddif-threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ddif_threshold = (
        args.ddif_threshold
        if args.ddif_threshold is not None
        else float(cfg.get("params", {}).get("ddif_threshold", 0.0))
    )

    logger = setup_logger("05_compare", os.path.join(cfg["output"]["tables_dir"], "05_compare_isoforms_within_gene.log"))
    df = pd.read_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step4.tsv"), sep="\t")

    pairwise_all = pd.concat([compare_group(g) for _, g in df.groupby("gene_name", dropna=False)], ignore_index=True)
    pairwise_all["delta_Delta_IF"] = to_num(pairwise_all["delta_Delta_IF"])
    pairwise_all["abs_delta_Delta_IF"] = pairwise_all["delta_Delta_IF"].abs()

    tables_dir = Path(cfg["output"]["tables_dir"])
    all_path = tables_dir / "isoform_pairwise_comparison_all.tsv"
    pairwise_all.to_csv(all_path, sep="\t", index=False)

    non_self = pairwise_all[pairwise_all["ref_transcript"].astype(str) != pairwise_all["query_transcript"].astype(str)].copy()
    non_self = non_self[non_self["delta_Delta_IF"].notna()].copy()
    candidate = non_self[non_self["abs_delta_Delta_IF"] >= float(ddif_threshold)].copy()
    candidate = candidate.sort_values("abs_delta_Delta_IF", ascending=False)

    thresh_tag = sanitize_threshold_for_filename(ddif_threshold)
    candidate_backup = tables_dir / f"isoform_pairwise_comparison.filtered.ddif_ge_{thresh_tag}.tsv"
    candidate_current = tables_dir / "isoform_pairwise_comparison.tsv"
    candidate.to_csv(candidate_backup, sep="\t", index=False)
    candidate.to_csv(candidate_current, sep="\t", index=False)

    final_cols = [
        "gene_name", "transcript_id", "M_mean", "R_mean", "log2FC", "IF_Total", "IF_Ribo", "Delta_IF",
        "transcript_length", "exon_count", "five_utr_length", "cds_length", "three_utr_length", "protein_length",
        "uORF_count", "kozak_context", "kozak_score", "kozak_strength", "miRNA_site_count", "polyA_signal_count",
        "nearest_polyA_signal_distance_to_3end", "five_utr_mfe", "cds_start_window_mfe", "three_utr_mfe", "nmd_likelihood",
        "reference_transcript", "orf_changed_vs_reference", "protein_changed_vs_reference", "domain_changed_vs_reference",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    final_df = df[final_cols].copy()
    final_df.to_csv(tables_dir / "transcript_features.tsv", sep="\t", index=False)

    stats_all_df = summarize_stats_all(df)
    stats_all_df.to_csv(tables_dir / "statistics_summary_all.tsv", sep="\t", index=False)

    stats_candidate_df = summarize_candidate_pair_stats(candidate, ddif_threshold)
    stats_candidate_df.to_csv(tables_dir / "statistics_summary.tsv", sep="\t", index=False)

    logger.info("total pairs: %d", len(pairwise_all))
    logger.info("non-self pairs: %d", len(non_self))
    logger.info("threshold: %s", ddif_threshold)
    logger.info("candidate pairs retained: %d", len(candidate))
    logger.info("candidate genes retained: %d", candidate["gene_name"].nunique())
    logger.info("Saved final transcript features (%d rows)", len(final_df))


if __name__ == "__main__":
    main()
