#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import add_common_cli_args, read_config, run_rnafold, setup_logger, which


def read_fasta_dict(fp: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    cur = None

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                cur = line[1:].split()[0]
                d[cur] = ""
            else:
                if cur is not None:
                    d[cur] += line

    return d


def safe_threshold_name(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def get_ddif_threshold(args, cfg) -> float:
    if getattr(args, "ddif_threshold", None) is not None:
        return float(args.ddif_threshold)

    return float(cfg.get("params", {}).get("ddif_threshold", 0.2))


def choose_fallback_reference(df: pd.DataFrame) -> pd.Series:
    """
    如果表里没有 reference_transcript，就临时按：
    1. CDS 最长
    2. M_mean 最高
    给每个 gene 选 reference。
    """
    work = df.copy()

    if "cds_length" in work.columns:
        work["cds_length_num"] = pd.to_numeric(work["cds_length"], errors="coerce")
    else:
        work["cds_length_num"] = np.nan

    if "M_mean" in work.columns:
        work["M_mean_num"] = pd.to_numeric(work["M_mean"], errors="coerce")
    else:
        work["M_mean_num"] = np.nan

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


def select_candidate_transcripts(
    df: pd.DataFrame,
    threshold: float,
    logger,
    tables_dir: Path,
) -> set[str]:
    """
    只选择 abs(delta_Delta_IF) > threshold 的 pair 涉及到的 transcript。

    注意：
    - 这里是严格大于：> threshold
    - query transcript 和 reference transcript 都会加入 RNAfold 计算集合
    """

    required_cols = {"gene_name", "transcript_id_base", "Delta_IF"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for candidate filtering: {sorted(missing)}")

    work = df.copy()

    work["transcript_id_base"] = work["transcript_id_base"].astype(str)
    work["Delta_IF_num"] = pd.to_numeric(work["Delta_IF"], errors="coerce")

    if "reference_transcript" in work.columns:
        work["reference_transcript"] = work["reference_transcript"].astype(str)
    else:
        logger.warning(
            "reference_transcript column not found. "
            "Fallback reference will be selected by longest CDS then highest M_mean."
        )
        work["reference_transcript"] = choose_fallback_reference(work).astype(str)

    # 去掉版本号，避免 ENSTxxx.1 和 ENSTxxx 对不上
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
        & (work["abs_delta_Delta_IF"] > threshold)
    ].copy()

    all_tids = set(work["transcript_id_base"].astype(str))

    selected = set(candidate_pairs["transcript_id_base"].astype(str))
    selected.update(candidate_pairs["reference_transcript"].astype(str))
    selected = selected & all_tids

    threshold_tag = safe_threshold_name(threshold)
    candidate_path = tables_dir / f"rnafold_5utr_candidate_transcripts.ddif_gt_{threshold_tag}.tsv"

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
        pd.DataFrame(columns=output_cols).to_csv(candidate_path, sep="\t", index=False)

        logger.warning("No candidate pairs found with abs(delta_Delta_IF) > %.4g", threshold)
        logger.warning("No transcript will be folded. MFE columns will be NA.")

        return set()

    candidate_pairs = candidate_pairs.sort_values("abs_delta_Delta_IF", ascending=False)
    candidate_pairs[output_cols].to_csv(candidate_path, sep="\t", index=False)

    logger.info("====== RNAfold candidate selection ======")
    logger.info("Threshold: abs(delta_Delta_IF) > %.4g", threshold)
    logger.info("Total transcripts in table: %d", len(all_tids))
    logger.info("Candidate pairs: %d", len(candidate_pairs))
    logger.info("Candidate genes: %d", candidate_pairs["gene_name"].nunique())
    logger.info("Transcripts selected for 5'UTR RNAfold, including references: %d", len(selected))
    logger.info("Candidate transcript table written: %s", candidate_path)
    logger.info("========================================")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Run RNAfold only for 5'UTR of candidate isoforms with abs(delta_Delta_IF) > threshold"
    )

    add_common_cli_args(parser)

    parser.add_argument(
        "--ddif-threshold",
        type=float,
        default=None,
        help=(
            "Only run 5'UTR RNAfold for transcripts involved in isoform pairs with "
            "abs(delta_Delta_IF) > threshold. "
            "Default: config.yaml params.ddif_threshold, fallback 0.2."
        ),
    )

    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("04_run_rnafold", Path(args.log) if args.log else None)

    tables_dir = Path(cfg["outputs"]["tables_dir"])
    seq_dir = Path(cfg["outputs"]["sequences_dir"])

    input_table = tables_dir / "transcript_features_orf_nmd.tsv"
    five_utr_fasta = seq_dir / "five_utr.fa"

    df = pd.read_csv(input_table, sep="\t")

    rnafold_bin = cfg.get("params", {}).get("rnafold_bin", "RNAfold")
    installed = which(rnafold_bin) is not None

    max_len = int(cfg.get("params", {}).get("rnafold_max_len", 500))
    ddif_threshold = get_ddif_threshold(args, cfg)

    selected_tids = select_candidate_transcripts(
        df=df,
        threshold=ddif_threshold,
        logger=logger,
        tables_dir=tables_dir,
    )

    logger.info("RNAfold binary: %s", rnafold_bin)
    logger.info("RNAfold installed: %s", installed)
    logger.info("Only 5'UTR will be folded.")
    logger.info("rnafold_max_len: %d", max_len)
    logger.info("Number of transcripts to run RNAfold: %d", len(selected_tids))

    if not installed:
        logger.warning(
            "RNAfold not found. five_utr_mfe will be NA. "
            "Install ViennaRNA and set params.rnafold_bin."
        )

    five = read_fasta_dict(five_utr_fasta)

    def clip(seq: str) -> str:
        """
        防止超长 5'UTR 太慢。
        默认最多取 500 nt。
        如果超过 max_len，取中间区域。
        """
        if not seq:
            return ""

        if len(seq) <= max_len:
            return seq

        center = len(seq) // 2
        half = max_len // 2

        return seq[max(0, center - half): center + half]

    fold_cache: dict[str, float] = {}

    def fold_5utr(seq: str):
        seq = clip(seq)

        if not seq:
            return np.nan

        if seq in fold_cache:
            return fold_cache[seq]

        mfe = run_rnafold(seq, rnafold_bin)
        fold_cache[seq] = mfe

        return mfe

    rows = []
    processed = 0
    total_to_fold = len(selected_tids)

    logger.info("Starting 5'UTR RNAfold...")

    for _, r in df.iterrows():
        tid = str(r["transcript_id_base"])

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

        five_mfe = fold_5utr(five_seq)

        processed += 1

        if processed == 1 or processed % 10 == 0 or processed == total_to_fold:
            logger.info(
                "RNAfold progress: %d/%d transcripts processed. Unique 5'UTR sequences folded: %d",
                processed,
                total_to_fold,
                len(fold_cache),
            )

        rows.append(
            {
                "transcript_id_base": tid,
                "five_utr_mfe": five_mfe,
                "cds_start_window_mfe": np.nan,
                "three_utr_mfe": np.nan,
            }
        )

    mfe_df = pd.DataFrame(rows)

    out = df.merge(mfe_df, on="transcript_id_base", how="left")

    out_path = tables_dir / "transcript_features.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    logger.info("====== RNAfold finished ======")
    logger.info("Transcripts selected for RNAfold: %d", len(selected_tids))
    logger.info("Transcripts actually processed: %d", processed)
    logger.info("Unique 5'UTR sequences folded: %d", len(fold_cache))
    logger.info("Final transcript feature table written: %s", out_path)
    logger.info("==============================")


if __name__ == "__main__":
    main()
