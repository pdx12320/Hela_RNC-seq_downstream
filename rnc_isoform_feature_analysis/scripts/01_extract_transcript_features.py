#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import gffutils
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pyfaidx import Fasta

from utils import ensure_dirs, load_config, setup_logger, transcript_base_id, translate_cds


def sort_exons(exons, strand):
    exons = sorted(exons, key=lambda x: x[0])
    if strand == "-":
        exons = exons[::-1]
    return exons


def fetch_spliced_seq(genome, chrom, blocks, strand):
    seq = "".join([str(genome[chrom][s - 1:e]) for s, e in blocks])
    seq = seq.upper()
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
    return seq


def genomic_to_transcript_pos(blocks, strand, gpos):
    pos = 0
    ordered = blocks if strand == "+" else blocks[::-1]
    for s, e in ordered:
        if s <= gpos <= e:
            if strand == "+":
                return pos + (gpos - s)
            return pos + (e - gpos)
        pos += e - s + 1
    return None


def get_tx_id(f):
    return f.attributes.get("transcript_id", [f.id])[0]


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8") as w:
        for rid, seq in records:
            w.write(f">{rid}\n")
            for i in range(0, len(seq), 60):
                w.write(seq[i:i + 60] + "\n")


def main():
    ap = argparse.ArgumentParser(description="Extract transcript/CDS/UTR/protein features and sequences")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs([cfg["output"]["tables_dir"], cfg["output"]["sequences_dir"]])
    logger = setup_logger("01_extract", os.path.join(cfg["output"]["tables_dir"], "01_extract_transcript_features.log"))

    matched = pd.read_csv(os.path.join(cfg["output"]["tables_dir"], "results_with_match.tsv"), sep="\t")
    matched = matched[matched["matched_transcript_id"].notna()].copy()
    matched["match_base"] = matched["matched_transcript_id"].astype(str).map(transcript_base_id)

    db = gffutils.FeatureDB(cfg["params"].get("gffutils_db", os.path.join(cfg["output"]["tables_dir"], "annotation.db")), keep_order=True)
    genome = Fasta(cfg["input"]["genome_fasta"], as_raw=True, sequence_always_upper=True)

    tx_map = {}
    for tx in db.features_of_type("transcript"):
        txid = get_tx_id(tx)
        tx_map[txid] = tx

    unique_tx = sorted(set(matched["matched_transcript_id"].astype(str)))

    rows = []
    tx_records, utr5_records, cds_records, utr3_records, prot_records = [], [], [], [], []

    for txid in unique_tx:
        if txid not in tx_map:
            base = transcript_base_id(txid)
            cand = [k for k in tx_map if transcript_base_id(k) == base]
            if not cand:
                logger.warning("Missing transcript in DB: %s", txid)
                continue
            txid = cand[0]
        tx = tx_map[txid]
        attrs = tx.attributes
        chrom = tx.chrom
        strand = tx.strand
        gene_id = attrs.get("gene_id", ["NA"])[0]
        gene_name = attrs.get("gene_name", ["NA"])[0]
        biotype = attrs.get("transcript_type", attrs.get("transcript_biotype", ["NA"]))[0]

        exons = [(e.start, e.end) for e in db.children(tx, featuretype="exon", order_by="start")]
        cds = [(c.start, c.end) for c in db.children(tx, featuretype="CDS", order_by="start")]
        start_codons = [(c.start, c.end) for c in db.children(tx, featuretype="start_codon", order_by="start")]
        stop_codons = [(c.start, c.end) for c in db.children(tx, featuretype="stop_codon", order_by="start")]

        exons_oriented = sort_exons(exons, strand)
        tx_seq = fetch_spliced_seq(genome, chrom, exons_oriented, strand) if exons else ""

        cds_seq = ""
        cds_len = 0
        cds_oriented = sort_exons(cds, strand)
        if cds_oriented:
            cds_seq = fetch_spliced_seq(genome, chrom, cds_oriented, strand)
            cds_len = len(cds_seq)

        coding = bool(cds_oriented)
        protein_seq = translate_cds(cds_seq) if coding else ""

        five_utr_seq = ""
        three_utr_seq = ""
        start_pos_tx = None
        stop_pos_tx = None

        if cds_oriented and exons_oriented:
            cds_g_start = min(s for s, _ in cds) if strand == "+" else max(e for _, e in cds)
            cds_g_end = max(e for _, e in cds) if strand == "+" else min(s for s, _ in cds)

            start_pos_tx = genomic_to_transcript_pos(exons, strand, cds_g_start)
            stop_last_base = cds_g_end
            stop_pos_tx = genomic_to_transcript_pos(exons, strand, stop_last_base)

            if start_pos_tx is not None:
                five_utr_seq = tx_seq[:start_pos_tx]
            if stop_pos_tx is not None:
                three_utr_seq = tx_seq[stop_pos_tx + 1:]

        row = {
            "transcript_id": txid,
            "transcript_id_base": transcript_base_id(txid),
            "gene_id": gene_id,
            "gene_name": gene_name,
            "chromosome": chrom,
            "strand": strand,
            "transcript_type": biotype,
            "exon_coordinates": ";".join([f"{a}-{b}" for a, b in exons_oriented]),
            "cds_coordinates": ";".join([f"{a}-{b}" for a, b in cds_oriented]),
            "start_codon_coordinates": ";".join([f"{a}-{b}" for a, b in start_codons]),
            "stop_codon_coordinates": ";".join([f"{a}-{b}" for a, b in stop_codons]),
            "transcript_length": len(tx_seq),
            "exon_count": len(exons_oriented),
            "five_utr_length": len(five_utr_seq),
            "cds_length": cds_len,
            "three_utr_length": len(three_utr_seq),
            "protein_length": len(protein_seq.replace("*", "")),
            "coding_status": "coding" if coding else "noncoding",
            "start_codon_pos_in_transcript": start_pos_tx if start_pos_tx is not None else np.nan,
            "stop_codon_position_in_transcript": stop_pos_tx if stop_pos_tx is not None else np.nan,
            "transcript_sequence": tx_seq,
            "five_utr_sequence": five_utr_seq,
            "cds_sequence": cds_seq,
            "three_utr_sequence": three_utr_seq,
            "protein_sequence": protein_seq,
        }
        rows.append(row)

        tx_records.append((txid, tx_seq))
        utr5_records.append((txid, five_utr_seq))
        cds_records.append((txid, cds_seq))
        utr3_records.append((txid, three_utr_seq))
        prot_records.append((txid, protein_seq))

    features = pd.DataFrame(rows)
    merged = matched.merge(features, on=["transcript_id_base"], how="left", suffixes=("", "_ann"))
    if "transcript_id_ann" in merged.columns:
        merged.rename(columns={"transcript_id_ann": "matched_annotation_transcript_id"}, inplace=True)

    features.to_csv(os.path.join(cfg["output"]["tables_dir"], "annotation_features.tsv"), sep="\t", index=False)
    merged.to_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_features_step1.tsv"), sep="\t", index=False)

    write_fasta(os.path.join(cfg["output"]["sequences_dir"], "transcript.fa"), tx_records)
    write_fasta(os.path.join(cfg["output"]["sequences_dir"], "five_utr.fa"), utr5_records)
    write_fasta(os.path.join(cfg["output"]["sequences_dir"], "cds.fa"), cds_records)
    write_fasta(os.path.join(cfg["output"]["sequences_dir"], "three_utr.fa"), utr3_records)
    write_fasta(os.path.join(cfg["output"]["sequences_dir"], "protein.fa"), prot_records)

    logger.info("Extracted features for %d transcripts", len(features))


if __name__ == "__main__":
    main()
