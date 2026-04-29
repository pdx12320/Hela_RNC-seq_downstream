#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import add_common_cli_args, read_config, run_rnafold, setup_logger, which


def read_fasta_dict(fp: Path):
    d, cur = {}, None
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].split()[0]
                d[cur] = ""
            else:
                d[cur] += line
    return d


def safe_threshold_name(x: float) -> str:
    """
    Make threshold safe for Windows filenames.
    Example:
    0.2 -> 0p2
    0.05 -> 0p05
    """
    return str(x).replace("-", "m").replace(".", "p")


def get_ddif_threshold(args, cfg) -> float:
    """
    Priority:
    1. terminal --ddif-threshold
    2. config.yaml params.ddif_threshold
    3. 0.0
    """
    if getattr(args, "ddif_threshold", None) is not None:
        return float(args.ddif_threshold)
    return float(cfg.get("params", {}).get("ddif_threshold", 0.0))


def choose_fallback_reference(df: pd.DataFrame) -> pd.Series:
    """
    Fallback reference transcript selection if reference_transcript column is missing.
    Priority:
    1. longest CDS
    2. highest M_mean
    """
    work = df.copy()

    work["cds_length_num"] = pd.to_numeric(work.get("cds_length", np.nan), errors="coerce")
    work["M_mean_num"] = pd.to_numeric(work.get("M_mean", np.nan), errors="coerce")

    refs = {}

    for gene, g in work.groupby("gene_name", dropna=False):
        g2 = g.copy()
        g2["cds_length_num"] = g2["cds_length_num"].fillna(-1)
        g2["M_mean_num"] = g2["M_mean_num"].fillna(-1)

        g2 = g2.sort_values(
            ["cds_length_num", "M_mean_num"],
            ascending=[False, False],
        )

        refs[gene] = str(g2.iloc[0]["transcript_id_base"])

    return work["gene_name"].map(refs)


def select_transcripts_for_rnafold(
    df: pd.DataFrame,
    threshold: float,
    logger,
    tables_dir: Path,
) -> set[str]:
    """
    If threshold <= 0:
        run RNAfold for all transcripts.

    If threshold > 0:
        compute delta_Delta_IF = query Delta_IF - reference Delta_IF,
        then select candidate pairs with abs(delta_Delta_IF) >= threshold.

    Both query transcript and reference transcript are selected for RNAfold.
    Non-selected transcripts will keep MFE as NA.
    """

    if "transcript_id_base" not in df.columns:
        raise ValueError("Missing required column: transcript_id_base")

    all_tids = set(df["transcript_id_base"].astype(str))

    if threshold <= 0:
        logger.info(
            "ddif_threshold <= 0. RNAfold will run for all transcripts: %d",
            len(all_tids),
        )
        return all_tids

    required_cols = {"gene_name", "transcript_id_base", "Delta_IF"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning(
            "Cannot prefilter RNAfold because required columns are missing: %s. "
            "All MFE columns will be NA.",
            sorted(missing),
        )
        return set()

    work = df.copy()
    work["transcript_id_base"] = work["transcript_id_base"].astype(str)
    work["Delta_IF_num"] = pd.to_numeric(work["Delta_IF"], errors="coerce")

    if "reference_transcript" in work.columns:
        work["reference_transcript"] = work["reference_transcript"].astype(str)
    else:
        logger.warning(
            "Column reference_transcript not found. "
            "Fallback reference will be selected by longest CDS, then highest M_mean."
        )
        work["reference_transcript"] = choose_fallback_reference(work).astype(str)

    # If reference transcript has version number, strip version.
    work["reference_transcript"] = work["reference_transcript"].str.replace(
        r"\.\d+$",
        "",
        regex=True,
    )

    delta_map = work.set_index("transcript_id_base")["Delta_IF_num"].to_dict()
    work["ref_Delta_IF"] = work["reference_transcript"].map(delta_map)

    work["delta_Delta_IF"] = work["Delta_IF_num"] - work["ref_Delta_IF"]
    work["abs_delta_Delta_IF"] = work["delta_Delta_IF"].abs()

    candidate_pairs = work[
        (work["transcript_id_base"] != work["reference_transcript"])
        & work["delta_Delta_IF"].notna()
        & (work["abs_delta_Delta_IF"] >= threshold)
    ].copy()

    selected = set(candidate_pairs["transcript_id_base"].astype(str))
    selected.update(candidate_pairs["reference_transcript"].astype(str))
    selected = selected & all_tids

    threshold_tag = safe_threshold_name(threshold)
    out_path = tables_dir / f"rnafold_candidate_transcripts.ddif_ge_{threshold_tag}.tsv"

    output_cols = [
        "gene_name",
        "reference_transcript",
        "transcript_id_base",
        "Delta_IF_num",
        "ref_Delta_IF",
        "delta_Delta_IF",
        "abs_delta_Delta_IF",
    ]

    if candidate_pairs.empty:
        pd.DataFrame(columns=output_cols).to_csv(out_path, sep="\t", index=False)
        logger.warning(
            "No candidate isoform pairs found with abs(delta_Delta_IF) >= %.4g. "
            "All MFE columns will be NA.",
            threshold,
        )
        return set()

    candidate_pairs = candidate_pairs.sort_values(
        "abs_delta_Delta_IF",
        ascending=False,
    )

    candidate_pairs[output_cols].to_csv(out_path, sep="\t", index=False)

    logger.info("ddif_threshold: %.4g", threshold)
    logger.info("Total transcripts: %d", len(all_tids))
    logger.info("Candidate pairs retained: %d", len(candidate_pairs))
    logger.info("Candidate genes retained: %d", candidate_pairs["gene_name"].nunique())
    logger.info(
        "Transcripts selected for RNAfold, including reference transcripts: %d",
        len(selected),
    )
    logger.info("RNAfold candidate transcript list written: %s", out_path)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Run RNAfold for selected candidate UTR and CDS-start windows"
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--ddif-threshold",
        type=float,
        default=None,
        help=(
            "Only run RNAfold for transcripts involved in isoform pairs with "
            "abs(delta_Delta_IF) >= threshold. "
            "If omitted, use params.ddif_threshold in config.yaml. "
            "Fallback default is 0.0, meaning run RNAfold for all transcripts."
        ),
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("04_run_rnafold", Path(args.log) if args.log else None)

    tables_dir = Path(cfg["outputs"]["tables_dir"])
    seq_dir = Path(cfg["outputs"]["sequences_dir"])

    input_table = tables_dir / "transcript_features_orf_nmd.tsv"
    df = pd.read_csv(input_table, sep="\t")

    tx = read_fasta_dict(seq_dir / "transcript.fa")
    five = read_fasta_dict(seq_dir / "five_utr.fa")
    three = read_fasta_dict(seq_dir / "three_utr.fa")

    rnafold_bin = cfg.get("params", {}).get("rnafold_bin", "RNAfold")
    installed = which(rnafold_bin) is not None

    max_len = int(cfg.get("params", {}).get("rnafold_max_len", 1500))
    flank = int(cfg.get("params", {}).get("cds_start_flank", 70))
    ddif_threshold = get_ddif_threshold(args, cfg)

    selected_tids = select_transcripts_for_rnafold(
        df=df,
        threshold=ddif_threshold,
        logger=logger,
        tables_dir=tables_dir,
    )

    def clip(s: str) -> str:
        if not s:
            return ""
        if len(s) <= max_len:
            return s
        center = len(s) // 2
        half = max_len // 2
        return s[max(0, center - half): center + half]

    # Cache: same sequence will be folded only once.
    fold_cache: dict[str, float] = {}

    def fold(seq: str):
        seq = clip(seq)
        if not seq:
            return np.nan
        if seq in fold_cache:
            return fold_cache[seq]

        mfe = run_rnafold(seq, rnafold_bin)
        fold_cache[seq] = mfe
        return mfe

    rows = []

    if not installed:
        logger.warning(
            "RNAfold not found. MFE columns filled with NA. "
            "Install ViennaRNA and set params.rnafold_bin."
        )

    for _, r in df.iterrows():
        tid = str(r["transcript_id_base"])

        # If RNAfold is not installed, or transcript is not selected, keep NA.
        if (not installed) or (tid not in selected_tids):
            rows.append(
                {
                    "transcript_id_base": tid,
                    "five_utr_mfe": np.nan,
                    "cds_start_window_mfe": np.nan,
                    "three_utr_mfe": np.nan,
                }
            )
            continue

        five_seq = five.get(tid, "")
        tx_seq = tx.get(tid, "")
        three_seq = three.get(tid, "")

        cds_start = len(five_seq)

        if tx_seq:
            start_window = tx_seq[
                max(0, cds_start - flank): min(len(tx_seq), cds_start + flank)
            ]
        else:
            start_window = ""

        five_mfe = fold(five_seq)
        start_mfe = fold(start_window)
        three_mfe = fold(three_seq)

        rows.append(
            {
                "transcript_id_base": tid,
                "five_utr_mfe": five_mfe,
                "cds_start_window_mfe": start_mfe,
                "three_utr_mfe": three_mfe,
            }
        )

    mfe_df = pd.DataFrame(rows)

    out = df.merge(mfe_df, on="transcript_id_base", how="left")

    out_path = tables_dir / "transcript_features.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    logger.info("Input table: %s", input_table)
    logger.info("RNAfold installed: %s", installed)
    logger.info("RNAfold binary: %s", rnafold_bin)
    logger.info("rnafold_max_len: %d", max_len)
    logger.info("cds_start_flank: %d", flank)
    logger.info("Unique RNA sequences folded: %d", len(fold_cache))
    logger.info("Final transcript feature table written: %s", out_path)


if __name__ == "__main__":
    main()
