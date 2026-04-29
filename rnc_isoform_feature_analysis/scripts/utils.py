#!/usr/bin/env python3
"""Shared utility functions for RNC isoform analysis pipeline."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from Bio.Seq import Seq


STOP_CODONS = {"TAA", "TAG", "TGA"}


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def transcript_base_id(tx_id: str) -> str:
    return str(tx_id).split(".")[0]


def normalize_transcript_id_column(df: pd.DataFrame, tx_col: str = "transcript_id") -> pd.DataFrame:
    df = df.copy()
    if tx_col not in df.columns:
        raise ValueError(f"Missing required column: {tx_col}")
    if "transcript_id_base" not in df.columns:
        df["transcript_id_base"] = df[tx_col].astype(str).map(transcript_base_id)
    else:
        df["transcript_id_base"] = df["transcript_id_base"].astype(str).map(transcript_base_id)
    df[tx_col] = df[tx_col].astype(str)
    return df


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def run_cmd(cmd: List[str], logger: Optional[logging.Logger] = None, check: bool = True) -> Tuple[int, str, str]:
    if logger:
        logger.info("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.returncode, p.stdout, p.stderr


def which(bin_name: str) -> Optional[str]:
    return shutil.which(bin_name)


def reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def translate_cds(seq: str) -> str:
    if not seq:
        return ""
    seq = seq.upper().replace("U", "T")
    trim = len(seq) - (len(seq) % 3)
    if trim <= 0:
        return ""
    return str(Seq(seq[:trim]).translate(to_stop=False))


def parse_attributes_gtf(attr_text: str) -> Dict[str, str]:
    attrs = {}
    for part in attr_text.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        if " " in part:
            k, v = part.split(" ", 1)
            attrs[k.strip()] = v.strip().strip('"')
        elif "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip().strip('"')
    return attrs


def kozak_score_and_strength(context_11nt: str) -> Tuple[int, str]:
    if not context_11nt or len(context_11nt) != 11:
        return (0, "NA")
    s = context_11nt.upper()
    minus3 = s[3]
    plus4 = s[10]
    score = int(minus3 in {"A", "G"}) + int(plus4 == "G")
    if score == 2:
        strength = "strong"
    elif score == 1:
        strength = "moderate"
    else:
        strength = "weak"
    return score, strength


def scan_uorfs(utr5_seq: str) -> List[Tuple[int, int]]:
    """Return uORF coordinates in 0-based half-open [start,end) on 5'UTR."""
    seq = (utr5_seq or "").upper()
    hits = []
    for i in range(0, len(seq) - 2):
        if seq[i:i + 3] == "ATG":
            frame = i % 3
            for j in range(i + 3, len(seq) - 2, 3):
                if j % 3 != frame:
                    continue
                codon = seq[j:j + 3]
                if codon in STOP_CODONS:
                    hits.append((i, j + 3))
                    break
    return hits


def extract_polyA_features(utr3_seq: str) -> Dict[str, object]:
    seq = (utr3_seq or "").upper()
    canonical = ["AATAAA", "ATTAAA"]
    variant = ["AGTAAA", "TATAAA", "CATAAA", "GATAAA", "AATATA", "AATACA", "AATAGA"]
    motifs = canonical + variant
    found = []
    for m in motifs:
        for mt in re.finditer(m, seq):
            found.append((m, mt.start()))
    found.sort(key=lambda x: x[1])
    if not found:
        return {
            "polyA_signal_count": 0,
            "nearest_polyA_signal_to_3end": "NA",
            "nearest_polyA_signal_distance_to_3end": np.nan,
            "has_canonical_polyA_signal": False,
            "has_variant_polyA_signal": False,
        }

    nearest = min(found, key=lambda x: len(seq) - (x[1] + len(x[0])))
    dist = len(seq) - (nearest[1] + len(nearest[0]))
    return {
        "polyA_signal_count": len(found),
        "nearest_polyA_signal_to_3end": nearest[0],
        "nearest_polyA_signal_distance_to_3end": dist,
        "has_canonical_polyA_signal": any(m in canonical for m, _ in found),
        "has_variant_polyA_signal": any(m in variant for m, _ in found),
    }


def read_table_auto(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def get_context_around(seq: str, start_idx: int, left: int, right: int) -> str:
    """1 codon starts at start_idx. returns length left+3+right with N padding."""
    seq = (seq or "").upper()
    needed_start = start_idx - left
    needed_end = start_idx + 3 + right
    out = []
    for i in range(needed_start, needed_end):
        if i < 0 or i >= len(seq):
            out.append("N")
        else:
            out.append(seq[i])
    return "".join(out)
