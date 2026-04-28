#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    add_common_cli_args,
    parse_gtf_models,
    pick_reference_transcript,
    read_config,
    read_optional_mapping,
    setup_logger,
)


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


def compare_orf(query_cds: str, ref_cds: str, query_prot: str, ref_prot: str):
    q = query_cds or ""
    r = ref_cds or ""
    qp = query_prot or ""
    rp = ref_prot or ""
    return {
        "orf_changed_vs_reference": q != r,
        "cds_length_changed_vs_reference": len(q) != len(r),
        "protein_changed_vs_reference": qp != rp,
        "n_terminal_change": (qp[:30] != rp[:30]) if qp and rp else np.nan,
        "c_terminal_change": (qp[-30:] != rp[-30:]) if qp and rp else np.nan,
        "frame_change_or_premature_stop": ((len(q) % 3 != 0) or ("*" in qp[:-1])) if q else np.nan,
    }


def compute_nmd_fields(exon_coords: str, cds_len: int):
    if not exon_coords or pd.isna(cds_len) or cds_len <= 0:
        return (np.nan, np.nan, np.nan, "NA", "no_cds_or_no_exons")
    exons = [tuple(map(int, x.split("-"))) for x in exon_coords.split(";") if "-" in x]
    exon_lengths = [abs(e - s) + 1 for s, e in exons]
    tx_len = sum(exon_lengths)
    stop_pos = cds_len
    if len(exons) <= 1:
        return (stop_pos, np.nan, np.nan, "low", "single_exon")
    last_ejc = tx_len - exon_lengths[-1]
    dist = last_ejc - stop_pos
    if dist > 55:
        return (stop_pos, last_ejc, dist, "high", "stop_upstream_of_last_EJC_gt55nt")
    return (stop_pos, last_ejc, dist, "low", "stop_in_last_exon_or_close_to_last_EJC")


def parse_domain_file(domain_fp: Path) -> pd.DataFrame:
    df = pd.read_csv(domain_fp, sep=None, engine="python")
    tid_col = None
    for c in ["transcript_id_base", "transcript_id", "protein_id", "query"]:
        if c in df.columns:
            tid_col = c
            break
    dom_col = None
    for c in ["domain", "signature_accession", "pfam", "accession", "ipr"]:
        if c in df.columns:
            dom_col = c
            break
    if not tid_col or not dom_col:
        return pd.DataFrame(columns=["transcript_id_base", "domain_set"])
    df["transcript_id_base"] = df[tid_col].astype(str).map(lambda x: x.split(".")[0])
    agg = df.groupby("transcript_id_base")[dom_col].apply(lambda x: sorted(set(map(str, x)))).reset_index()
    agg["domain_set"] = agg[dom_col].apply(lambda x: ";".join(x))
    return agg[["transcript_id_base", "domain_set"]]


def main():
    parser = argparse.ArgumentParser(description="Predict ORF/NMD and compare with reference isoform")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("03_predict_orf_uorf_nmd", Path(args.log) if args.log else None)

    features = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "transcript_features_motifs.tsv", sep="\t")
    cds = read_fasta_dict(Path(cfg["outputs"]["sequences_dir"]) / "cds.fa")
    prot = read_fasta_dict(Path(cfg["outputs"]["sequences_dir"]) / "protein.fa")

    canonical_map = read_optional_mapping(cfg["inputs"].get("canonical_transcripts"), "gene_name", "transcript_id")

    # reference selection
    ref_map = {}
    for gene, g in features.groupby("gene_name"):
        ref_map[gene] = pick_reference_transcript(g, canonical_map)

    # optional domain file
    domain_fp = cfg["inputs"].get("domain_annotation")
    domain_df = parse_domain_file(Path(domain_fp)) if domain_fp and Path(domain_fp).exists() else pd.DataFrame(columns=["transcript_id_base", "domain_set"])
    domain_map = dict(zip(domain_df["transcript_id_base"], domain_df["domain_set"]))

    out_rows = []
    for _, r in features.iterrows():
        tid = r["transcript_id_base"]
        gene = r["gene_name"]
        ref = ref_map.get(gene, tid)

        q_cds, r_cds = cds.get(tid, ""), cds.get(ref, "")
        q_prot, r_prot = prot.get(tid, ""), prot.get(ref, "")
        cmp = compare_orf(q_cds, r_cds, q_prot, r_prot)

        q_dom = set(str(domain_map.get(tid, "")).split(";")) if tid in domain_map else set()
        r_dom = set(str(domain_map.get(ref, "")).split(";")) if ref in domain_map else set()
        q_dom.discard("")
        r_dom.discard("")

        stop_pos, last_ejc, dist, nmd, reason = compute_nmd_fields(r.get("exon_coordinates", ""), r.get("cds_length", 0))

        out_rows.append(
            {
                **r.to_dict(),
                "reference_transcript": ref,
                **cmp,
                "domain_changed_vs_reference": (q_dom != r_dom) if domain_fp else np.nan,
                "domain_lost": ";".join(sorted(r_dom - q_dom)) if domain_fp else "NA",
                "domain_gained": ";".join(sorted(q_dom - r_dom)) if domain_fp else "NA",
                "stop_codon_position_in_transcript": stop_pos,
                "last_exon_junction_position": last_ejc,
                "distance_stop_to_last_EJC": dist,
                "nmd_likelihood": nmd,
                "nmd_reason": reason,
            }
        )

    out = pd.DataFrame(out_rows)
    out_path = Path(cfg["outputs"]["tables_dir"]) / "transcript_features_orf_nmd.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    logger.info("ORF/NMD feature table written: %s", out_path)


if __name__ == "__main__":
    main()
