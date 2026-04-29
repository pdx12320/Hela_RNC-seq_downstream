#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess

import numpy as np
import pandas as pd

from utils import load_config, setup_logger, which


def run_rnafold(seq: str, rnafold_bin: str) -> float:
    if not seq:
        return np.nan
    seq = seq.replace("T", "U").upper()
    p = subprocess.run([rnafold_bin, "--noPS"], input=seq + "\n", capture_output=True, text=True)
    if p.returncode != 0:
        return np.nan
    lines = [x.strip() for x in p.stdout.strip().splitlines() if x.strip()]
    if len(lines) < 2:
        return np.nan
    m = re.search(r"\(([-0-9\.]+)\)", lines[-1])
    if not m:
        return np.nan
    return float(m.group(1))


def maybe_truncate(seq: str, max_len: int) -> str:
    seq = seq or ""
    if len(seq) <= max_len:
        return seq
    half = max_len // 2
    return seq[:half] + seq[-half:]


def main():
    ap = argparse.ArgumentParser(description="Run RNAfold (optional) on UTR/CDS window")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("04_rnafold", os.path.join(cfg["output"]["tables_dir"], "04_run_rnafold.log"))
    df = pd.read_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step3.tsv"), sep="\t")

    rnafold_bin = cfg["params"].get("rnafold_binary", "RNAfold")
    installed = which(rnafold_bin) is not None
    if not installed:
        logger.warning("RNAfold not found. Fill NA for mfe columns.")
        df["five_utr_mfe"] = np.nan
        df["cds_start_window_mfe"] = np.nan
        df["three_utr_mfe"] = np.nan
        df.to_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step4.tsv"), sep="\t", index=False)
        return

    up = int(cfg["params"].get("cds_start_window_upstream", 70))
    down = int(cfg["params"].get("cds_start_window_downstream", 70))
    max_len = int(cfg["params"].get("rnafold_max_len", 2000))

    f5, cwin, f3 = [], [], []
    for _, r in df.iterrows():
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

        f5.append(run_rnafold(utr5, rnafold_bin))
        cwin.append(run_rnafold(win, rnafold_bin))
        f3.append(run_rnafold(utr3, rnafold_bin))

    df["five_utr_mfe"] = f5
    df["cds_start_window_mfe"] = cwin
    df["three_utr_mfe"] = f3
    df.to_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step4.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
