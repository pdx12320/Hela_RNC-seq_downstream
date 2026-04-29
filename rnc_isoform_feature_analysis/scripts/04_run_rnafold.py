#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, setup_logger, which


def run_rnafold_cached(seq: str, rnafold_bin: str, cache: dict[str, float]) -> float:
    if not seq:
        return np.nan
    seq = seq.replace("T", "U").upper()
    if seq in cache:
        return cache[seq]

    p = subprocess.run([rnafold_bin, "--noPS"], input=seq + "\n", capture_output=True, text=True)
    if p.returncode != 0:
        cache[seq] = np.nan
        return np.nan
    lines = [x.strip() for x in p.stdout.strip().splitlines() if x.strip()]
    if len(lines) < 2:
        cache[seq] = np.nan
        return np.nan
    m = re.search(r"\(([-0-9\.]+)\)", lines[-1])
    if not m:
        cache[seq] = np.nan
        return np.nan
    cache[seq] = float(m.group(1))
    return cache[seq]


def maybe_truncate(seq: str, max_len: int) -> str:
    seq = seq or ""
    if len(seq) <= max_len:
        return seq
    half = max_len // 2
    return seq[:half] + seq[-half:]


def sanitize_threshold_for_filename(threshold: float) -> str:
    txt = f"{threshold:g}"
    return txt.replace("-", "neg").replace(".", "p")


def resolve_ddif_threshold(args_threshold: float | None, cfg: dict) -> float:
    if args_threshold is not None:
        return float(args_threshold)
    return float(cfg.get("params", {}).get("ddif_threshold", 0.0))


def compute_candidate_transcripts(df: pd.DataFrame, ddif_threshold: float, logger) -> tuple[pd.DataFrame, set[str]]:
    needed_cols = {"gene_name", "transcript_id", "reference_transcript", "Delta_IF"}
    if not needed_cols.issubset(set(df.columns)):
        missing = sorted(needed_cols - set(df.columns))
        logger.warning("Missing columns for candidate selection: %s. Fallback to all transcripts.", ",".join(missing))
        return pd.DataFrame(), set(df["transcript_id"].astype(str).tolist())

    tmp = df[["gene_name", "transcript_id", "reference_transcript", "Delta_IF"]].copy()
    tmp["transcript_id"] = tmp["transcript_id"].astype(str)
    tmp["reference_transcript"] = tmp["reference_transcript"].astype(str)
    tmp["Delta_IF"] = pd.to_numeric(tmp["Delta_IF"], errors="coerce")

    ref_df = tmp[["gene_name", "transcript_id", "Delta_IF"]].rename(
        columns={"transcript_id": "ref_transcript", "Delta_IF": "ref_Delta_IF"}
    )
    merged = tmp.merge(ref_df, left_on=["gene_name", "reference_transcript"], right_on=["gene_name", "ref_transcript"], how="left")
    merged["delta_Delta_IF"] = merged["Delta_IF"] - merged["ref_Delta_IF"]
    merged["abs_delta_Delta_IF"] = merged["delta_Delta_IF"].abs()

    candidate_pairs = merged[
        (merged["transcript_id"] != merged["reference_transcript"])
        & merged["delta_Delta_IF"].notna()
        & (merged["abs_delta_Delta_IF"] >= float(ddif_threshold))
    ].copy()

    selected = set(candidate_pairs["transcript_id"].astype(str).tolist())
    selected.update(candidate_pairs["reference_transcript"].astype(str).tolist())

    return candidate_pairs, selected


def main():
    ap = argparse.ArgumentParser(description="Run RNAfold (optional) on UTR/CDS window")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ddif-threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ddif_threshold = resolve_ddif_threshold(args.ddif_threshold, cfg)
    tables_dir = Path(cfg["output"]["tables_dir"])
    logger = setup_logger("04_rnafold", str(tables_dir / "04_run_rnafold.log"))

    step3_orf_nmd = tables_dir / "transcript_features_orf_nmd.tsv"
    step3_default = tables_dir / "transcript_features_step3.tsv"
    if step3_orf_nmd.exists():
        input_path = step3_orf_nmd
    else:
        input_path = step3_default
        logger.warning("Expected %s not found, fallback to %s", step3_orf_nmd, step3_default)

    df = pd.read_csv(input_path, sep="\t")

    df["five_utr_mfe"] = np.nan
    df["cds_start_window_mfe"] = np.nan
    df["three_utr_mfe"] = np.nan

    total_transcripts = len(df)

    if ddif_threshold <= 0:
        selected_transcripts = set(df["transcript_id"].astype(str).tolist())
        candidate_pairs = pd.DataFrame()
    else:
        candidate_pairs, selected_transcripts = compute_candidate_transcripts(df, ddif_threshold, logger)
        threshold_tag = sanitize_threshold_for_filename(ddif_threshold)
        candidate_tx_out = tables_dir / f"rnafold_candidate_transcripts.ddif_ge_{threshold_tag}.tsv"
        pd.DataFrame({"transcript_id": sorted(selected_transcripts)}).to_csv(candidate_tx_out, sep="\t", index=False)

        if len(candidate_pairs) == 0:
            logger.warning("No candidate pair retained at |ΔΔIF| >= %s. All MFE columns remain NA.", ddif_threshold)
            df.to_csv(tables_dir / "transcript_features_step4.tsv", sep="\t", index=False)
            logger.info("ddif_threshold: %s", ddif_threshold)
            logger.info("total transcripts: %d", total_transcripts)
            logger.info("candidate pairs: 0")
            logger.info("candidate genes: 0")
            logger.info("transcripts selected for RNAfold: 0")
            logger.info("unique sequences folded: 0")
            return

    rnafold_bin = cfg["params"].get("rnafold_binary", "RNAfold")
    installed = which(rnafold_bin) is not None
    if not installed:
        logger.warning("RNAfold not found. Fill NA for mfe columns.")
        df.to_csv(tables_dir / "transcript_features_step4.tsv", sep="\t", index=False)
        return

    up = int(cfg["params"].get("cds_start_window_upstream", 70))
    down = int(cfg["params"].get("cds_start_window_downstream", 70))
    max_len = int(cfg["params"].get("rnafold_max_len", 2000))

    fold_cache: dict[str, float] = {}

    for idx, r in df.iterrows():
        tx_id = str(r.get("transcript_id", ""))
        if tx_id not in selected_transcripts:
            continue

        tx = str(r.get("transcript_sequence", "") or "")
        utr5 = maybe_truncate(str(r.get("five_utr_sequence", "") or ""), max_len)
        utr3 = maybe_truncate(str(r.get("three_utr_sequence", "") or ""), max_len)

        sp = r.get("start_codon_pos_in_transcript", np.nan)
        if pd.notna(sp):
            sp = int(sp)
            l = max(0, sp - up)
            rr = min(len(tx), sp + 3 + down)
            win = tx[l:rr]
        else:
            win = ""

        df.at[idx, "five_utr_mfe"] = run_rnafold_cached(utr5, rnafold_bin, fold_cache)
        df.at[idx, "cds_start_window_mfe"] = run_rnafold_cached(win, rnafold_bin, fold_cache)
        df.at[idx, "three_utr_mfe"] = run_rnafold_cached(utr3, rnafold_bin, fold_cache)

    df.to_csv(tables_dir / "transcript_features_step4.tsv", sep="\t", index=False)

    logger.info("ddif_threshold: %s", ddif_threshold)
    logger.info("total transcripts: %d", total_transcripts)
    logger.info("candidate pairs: %d", len(candidate_pairs) if ddif_threshold > 0 else "NA (threshold<=0)")
    logger.info("candidate genes: %d", candidate_pairs["gene_name"].nunique() if ddif_threshold > 0 else "NA (threshold<=0)")
    logger.info("transcripts selected for RNAfold: %d", len(selected_transcripts))
    logger.info("unique sequences folded: %d", len(fold_cache))


if __name__ == "__main__":
    main()
