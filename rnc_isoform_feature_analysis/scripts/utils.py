#!/usr/bin/env python3
"""Shared utility functions for RNC isoform feature pipeline."""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from Bio.Seq import Seq
from pyfaidx import Fasta

STOP_CODONS = {"TAA", "TAG", "TGA"}
POLYA_CANONICAL = {"AATAAA", "ATTAAA"}
POLYA_VARIANT = {"AGTAAA", "TATAAA", "CATAAA", "GATAAA", "AATATA", "AATACA", "AATAGA"}


@dataclass
class TranscriptModel:
    transcript_id: str
    transcript_id_base: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    transcript_type: str
    exons: List[Tuple[int, int]]
    cds: List[Tuple[int, int]]
    start_codon: List[Tuple[int, int]]
    stop_codon: List[Tuple[int, int]]


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def read_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def transcript_base_id(tid: str) -> str:
    return tid.split(".")[0]


def parse_gtf_attributes(attr: str) -> Dict[str, str]:
    out = {}
    for m in re.finditer(r'(\S+)\s+"([^"]+)"', attr):
        out[m.group(1)] = m.group(2)
    if not out:
        for chunk in attr.strip().split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                out[k] = v
    return out


def load_results_table(results_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    req = [
        "gene_name", "transcript_id", "M_mean", "R_mean", "log2FC", "IF_Total", "IF_Ribo", "Delta_IF"
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"results.csv 缺失列: {missing}")
    if "transcript_id_base" not in df.columns:
        df["transcript_id_base"] = df["transcript_id"].astype(str).map(transcript_base_id)
    else:
        df["transcript_id_base"] = df["transcript_id_base"].astype(str).map(transcript_base_id)
    df["transcript_id"] = df["transcript_id"].astype(str)
    return df


def parse_gtf_models(gtf_path: str | Path, transcript_ids_base: Optional[set] = None) -> Dict[str, TranscriptModel]:
    models: Dict[str, TranscriptModel] = {}
    with open(gtf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts
            if feature not in {"transcript", "exon", "CDS", "start_codon", "stop_codon", "five_prime_utr", "three_prime_utr", "UTR"}:
                continue
            a = parse_gtf_attributes(attrs)
            tid = a.get("transcript_id") or a.get("transcript") or a.get("ID", "")
            if not tid:
                continue
            tidb = transcript_base_id(tid)
            if transcript_ids_base and tidb not in transcript_ids_base:
                continue
            gid = a.get("gene_id", a.get("gene", "NA"))
            gname = a.get("gene_name", a.get("gene", "NA"))
            ttype = a.get("transcript_type", a.get("transcript_biotype", a.get("transcript_bio_type", "NA")))
            if tidb not in models:
                models[tidb] = TranscriptModel(
                    transcript_id=tid,
                    transcript_id_base=tidb,
                    gene_id=gid,
                    gene_name=gname,
                    chrom=chrom,
                    strand=strand,
                    transcript_type=ttype,
                    exons=[],
                    cds=[],
                    start_codon=[],
                    stop_codon=[],
                )
            m = models[tidb]
            s, e = int(start), int(end)
            if feature == "exon":
                m.exons.append((s, e))
            elif feature == "CDS":
                m.cds.append((s, e))
            elif feature == "start_codon":
                m.start_codon.append((s, e))
            elif feature == "stop_codon":
                m.stop_codon.append((s, e))

    for m in models.values():
        m.exons = sort_intervals_by_strand(m.exons, m.strand)
        m.cds = sort_intervals_by_strand(m.cds, m.strand)
        m.start_codon = sort_intervals_by_strand(m.start_codon, m.strand)
        m.stop_codon = sort_intervals_by_strand(m.stop_codon, m.strand)
    return models


def sort_intervals_by_strand(intervals: List[Tuple[int, int]], strand: str) -> List[Tuple[int, int]]:
    if strand == "+":
        return sorted(intervals, key=lambda x: (x[0], x[1]))
    return sorted(intervals, key=lambda x: (x[0], x[1]), reverse=True)


def fetch_spliced_sequence(genome: Fasta, chrom: str, intervals: Sequence[Tuple[int, int]], strand: str) -> str:
    if not intervals:
        return ""
    seq = "".join(str(genome[chrom][s - 1:e]).upper() for s, e in intervals)
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
    return seq


def project_genomic_to_transcript(intervals: Sequence[Tuple[int, int]], strand: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    tpos = 0
    ordered = sort_intervals_by_strand(list(intervals), strand)
    if strand == "+":
        for s, e in ordered:
            for g in range(s, e + 1):
                mapping[g] = tpos
                tpos += 1
    else:
        for s, e in ordered:
            for g in range(e, s - 1, -1):
                mapping[g] = tpos
                tpos += 1
    return mapping


def split_utr_cds(model: TranscriptModel, genome: Fasta) -> Dict[str, str]:
    tx_seq = fetch_spliced_sequence(genome, model.chrom, model.exons, model.strand)
    cds_seq = fetch_spliced_sequence(genome, model.chrom, model.cds, model.strand)
    if not model.cds:
        return {"transcript": tx_seq, "five_utr": tx_seq, "cds": "", "three_utr": ""}

    map_g2t = project_genomic_to_transcript(model.exons, model.strand)
    cds_genomic = []
    for s, e in model.cds:
        if model.strand == "+":
            cds_genomic.extend(range(s, e + 1))
        else:
            cds_genomic.extend(range(e, s - 1, -1))
    t_indices = [map_g2t[g] for g in cds_genomic if g in map_g2t]
    if not t_indices:
        return {"transcript": tx_seq, "five_utr": tx_seq, "cds": "", "three_utr": ""}
    cds_start, cds_end = min(t_indices), max(t_indices)
    five = tx_seq[:cds_start]
    cds = tx_seq[cds_start:cds_end + 1]
    three = tx_seq[cds_end + 1:]
    return {"transcript": tx_seq, "five_utr": five, "cds": cds_seq if cds_seq else cds, "three_utr": three}


def translate_cds(cds_seq: str) -> str:
    if not cds_seq:
        return ""
    usable = len(cds_seq) - (len(cds_seq) % 3)
    if usable <= 0:
        return ""
    pep = str(Seq(cds_seq[:usable]).translate(to_stop=False))
    return pep.rstrip("*")


def write_fasta(records: Dict[str, str], out_fa: str | Path) -> None:
    with open(out_fa, "w", encoding="utf-8") as f:
        for rid, seq in records.items():
            f.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def kozak_context_score(seq: str, start_idx: int) -> Tuple[str, int, str]:
    if not seq or start_idx < 0 or start_idx + 3 > len(seq):
        return ("NA", np.nan, "NA")
    left = max(0, start_idx - 6)
    right = min(len(seq), start_idx + 4)
    context = seq[left:right]
    minus3_idx = start_idx - 3
    plus4_idx = start_idx + 3
    score = 0
    if 0 <= minus3_idx < len(seq) and seq[minus3_idx] in {"A", "G"}:
        score += 1
    if 0 <= plus4_idx < len(seq) and seq[plus4_idx] == "G":
        score += 1
    strength = "strong" if score == 2 else "moderate" if score == 1 else "weak"
    return (context, score, strength)


def scan_uorfs(five_utr_seq: str) -> dict:
    seq = (five_utr_seq or "").upper()
    starts = [i for i in range(0, len(seq) - 2) if seq[i:i + 3] == "ATG"]
    uorfs = []
    strong_cnt = 0
    for s in starts:
        stop = None
        for j in range(s + 3, len(seq) - 2, 3):
            codon = seq[j:j + 3]
            if codon in STOP_CODONS:
                stop = j + 3
                break
        if stop is None:
            continue
        ctx, sc, st = kozak_context_score(seq, s)
        if st == "strong":
            strong_cnt += 1
        uorfs.append({"start": s + 1, "end": stop, "len_nt": stop - s, "kozak": st, "ctx": ctx})
    longest = max([x["len_nt"] for x in uorfs], default=0)
    coords = ";".join([f"{u['start']}-{u['end']}({u['kozak']})" for u in uorfs]) if uorfs else ""
    return {
        "uORF_count": len(uorfs),
        "longest_uORF_length_nt": longest,
        "longest_uORF_length_aa": longest // 3 if longest else 0,
        "uORF_with_strong_kozak_count": strong_cnt,
        "uORF_coordinates_in_5utr": coords,
    }


def scan_polya(three_utr_seq: str) -> dict:
    seq = (three_utr_seq or "").upper()
    hits = []
    for i in range(0, len(seq) - 5):
        m = seq[i:i + 6]
        if m in POLYA_CANONICAL or m in POLYA_VARIANT:
            dist = len(seq) - (i + 6)
            hits.append((i + 1, m, dist))
    nearest = min(hits, key=lambda x: x[2]) if hits else None
    return {
        "polyA_signal_count": len(hits),
        "nearest_polyA_signal_to_3end": nearest[1] if nearest else "NA",
        "nearest_polyA_signal_distance_to_3end": nearest[2] if nearest else np.nan,
        "has_canonical_polyA_signal": any(h[1] in POLYA_CANONICAL for h in hits),
        "has_variant_polyA_signal": any(h[1] in POLYA_VARIANT for h in hits),
    }


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_rnafold(seq: str, rnafold_bin: str = "RNAfold") -> Optional[float]:
    if not seq:
        return np.nan
    if which(rnafold_bin) is None:
        return np.nan
    try:
        p = subprocess.run([rnafold_bin, "--noPS"], input=seq + "\n", text=True, capture_output=True, check=False)
        if p.returncode != 0:
            return np.nan
        for line in p.stdout.splitlines():
            m = re.search(r"\(([-0-9.]+)\)", line)
            if m:
                return float(m.group(1))
    except Exception:
        return np.nan
    return np.nan


def pick_reference_transcript(group: pd.DataFrame, canonical_map: Dict[str, str]) -> str:
    gene = str(group["gene_name"].iloc[0])
    if gene in canonical_map:
        ref = transcript_base_id(canonical_map[gene])
        if ref in set(group["transcript_id_base"]):
            return ref
    group2 = group.copy()
    if "cds_length" in group2.columns:
        group2 = group2.sort_values(["cds_length", "M_mean"], ascending=[False, False])
        return str(group2.iloc[0]["transcript_id_base"])
    group2 = group.sort_values(["M_mean"], ascending=False)
    return str(group2.iloc[0]["transcript_id_base"])


def read_optional_mapping(path: Optional[str | Path], key_col: str, val_col: str) -> Dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, sep=None, engine="python")
    if key_col not in df.columns or val_col not in df.columns:
        return {}
    return dict(zip(df[key_col].astype(str), df[val_col].astype(str)))


def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def add_common_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--log", default=None, help="Optional log file")
    return parser
