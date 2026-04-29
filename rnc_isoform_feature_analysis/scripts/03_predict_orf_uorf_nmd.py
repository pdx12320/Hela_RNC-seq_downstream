#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from utils import load_config, setup_logger, transcript_base_id


def parse_domain_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    tx_col = next((c for c in ["transcript_id", "protein_id", "query", "sequence_id"] if c in df.columns), None)
    dom_col = next((c for c in ["domain", "signature_accession", "pfam_id", "accession"] if c in df.columns), None)
    if tx_col is None or dom_col is None:
        return pd.DataFrame(columns=["transcript_id", "domain_set"])
    out = df.groupby(tx_col)[dom_col].apply(lambda x: sorted(set([str(v) for v in x if pd.notna(v)]))).reset_index()
    out.columns = ["transcript_id", "domain_set"]
    return out


def choose_reference(group: pd.DataFrame, canonical_map: dict) -> str:
    gene = str(group["gene_name"].iloc[0])
    if gene in canonical_map:
        c = canonical_map[gene]
        if c in set(group["transcript_id"].astype(str)):
            return c
        cbase = transcript_base_id(c)
        candidates = [x for x in group["transcript_id"].astype(str) if transcript_base_id(x) == cbase]
        if candidates:
            return candidates[0]
    coding = group.copy()
    coding["cds_length"] = pd.to_numeric(coding["cds_length"], errors="coerce").fillna(0)
    if (coding["cds_length"] > 0).any():
        return coding.sort_values(["cds_length", "M_mean"], ascending=[False, False])["transcript_id"].iloc[0]
    return group.sort_values("M_mean", ascending=False)["transcript_id"].iloc[0]


def nmd_for_row(r, threshold=55):
    try:
        if str(r.get("coding_status", "")) != "coding":
            return (np.nan, np.nan, np.nan, "NA", "no_cds")
        exons = str(r.get("exon_coordinates", ""))
        if not exons or exons == "nan":
            return (r.get("stop_codon_position_in_transcript", np.nan), np.nan, np.nan, "NA", "no_exon_info")
        parts = [p for p in exons.split(";") if p]
        exon_count = len(parts)
        if exon_count <= 1:
            return (r.get("stop_codon_position_in_transcript", np.nan), np.nan, np.nan, "low", "single_exon")

        tx_len = int(r.get("transcript_length", 0) or 0)
        last_junc = tx_len - int(parts[-1].split("-")[1]) + int(parts[-1].split("-")[0]) - 1
        stop_pos = int(r.get("stop_codon_position_in_transcript", np.nan)) if pd.notna(r.get("stop_codon_position_in_transcript", np.nan)) else np.nan
        if pd.isna(stop_pos):
            return (np.nan, last_junc, np.nan, "NA", "stop_unknown")
        dist = last_junc - stop_pos
        if dist > threshold:
            return (stop_pos, last_junc, dist, "high", f"stop_{dist}nt_upstream_of_last_EJC")
        return (stop_pos, last_junc, dist, "low", "stop_in_last_exon_or_close_to_last_EJC")
    except Exception:
        return (np.nan, np.nan, np.nan, "NA", "parse_error")


def main():
    ap = argparse.ArgumentParser(description="ORF/domain/NMD comparative analysis")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("03_orf", os.path.join(cfg["output"]["tables_dir"], "03_predict_orf_uorf_nmd.log"))
    df = pd.read_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step2.tsv"), sep="\t")

    canonical_map = {}
    canon_path = cfg["input"].get("canonical_transcripts_tsv")
    if canon_path and os.path.exists(canon_path):
        cdf = pd.read_csv(canon_path, sep="\t")
        gcol = next((c for c in ["gene_name", "gene", "symbol"] if c in cdf.columns), None)
        tcol = next((c for c in ["transcript_id", "canonical_transcript", "mane_transcript"] if c in cdf.columns), None)
        if gcol and tcol:
            canonical_map = dict(zip(cdf[gcol].astype(str), cdf[tcol].astype(str)))

    domain_df = None
    domain_path = cfg["input"].get("domain_tsv")
    if domain_path and os.path.exists(domain_path):
        domain_df = parse_domain_table(domain_path)
        logger.info("Loaded domain annotation: %s", domain_path)

    if domain_df is not None:
        df = df.merge(domain_df, on="transcript_id", how="left")
    else:
        df["domain_set"] = np.nan

    references = {}
    for gene, g in df.groupby("gene_name", dropna=False):
        references[gene] = choose_reference(g, canonical_map)

    rows = []
    for gene, g in df.groupby("gene_name", dropna=False):
        ref_id = references[gene]
        ref = g[g["transcript_id"] == ref_id].iloc[0]
        ref_cds = str(ref.get("cds_sequence", "") or "")
        ref_prot = str(ref.get("protein_sequence", "") or "")
        ref_start = ref.get("start_codon_pos_in_transcript", np.nan)
        ref_stop = ref.get("stop_codon_position_in_transcript", np.nan)
        ref_domain = set(ref["domain_set"] if isinstance(ref["domain_set"], list) else [])

        for _, r in g.iterrows():
            cds = str(r.get("cds_sequence", "") or "")
            prot = str(r.get("protein_sequence", "") or "")
            orf_same = cds == ref_cds and cds != ""
            prot_same = prot == ref_prot and prot != ""
            nterm = bool(prot and ref_prot and (not prot.startswith(ref_prot[: min(20, len(ref_prot))])))
            cterm = bool(prot and ref_prot and (not prot.endswith(ref_prot[-min(20, len(ref_prot)):])) )
            frame_change = (len(cds) % 3 != 0) if cds else False
            premature_stop = "*" in prot[:-1] if prot else False

            qdom = set(r["domain_set"] if isinstance(r["domain_set"], list) else [])
            gained = sorted(qdom - ref_domain)
            lost = sorted(ref_domain - qdom)
            domain_changed = len(gained) > 0 or len(lost) > 0

            stop_pos, last_junc, dist, nmd, reason = nmd_for_row(r, cfg["params"].get("nmd_distance_threshold", 55))

            rows.append(
                {
                    "transcript_id": r["transcript_id"],
                    "reference_transcript": ref_id,
                    "orf_changed_vs_reference": not orf_same,
                    "protein_changed_vs_reference": not prot_same,
                    "cds_length_changed_vs_reference": int(r.get("cds_length", 0) or 0) - int(ref.get("cds_length", 0) or 0),
                    "start_codon_changed_vs_reference": bool(pd.notna(r.get("start_codon_pos_in_transcript")) and pd.notna(ref_start) and int(r.get("start_codon_pos_in_transcript")) != int(ref_start)),
                    "stop_codon_changed_vs_reference": bool(pd.notna(r.get("stop_codon_position_in_transcript")) and pd.notna(ref_stop) and int(r.get("stop_codon_position_in_transcript")) != int(ref_stop)),
                    "n_terminal_change_vs_reference": nterm,
                    "c_terminal_change_vs_reference": cterm,
                    "frame_change_vs_reference": frame_change,
                    "premature_stop_codon": premature_stop,
                    "domain_changed_vs_reference": domain_changed,
                    "domain_lost": ";".join(lost),
                    "domain_gained": ";".join(gained),
                    "stop_codon_position_in_transcript": stop_pos,
                    "last_exon_junction_position": last_junc,
                    "distance_stop_to_last_EJC": dist,
                    "nmd_likelihood": nmd,
                    "nmd_reason": reason,
                }
            )

    cmp = pd.DataFrame(rows)
    out = df.merge(cmp, on="transcript_id", how="left")
    out.to_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step3.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
