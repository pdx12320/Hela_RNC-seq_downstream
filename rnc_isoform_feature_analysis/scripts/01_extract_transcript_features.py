#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

from utils import (
    add_common_cli_args,
    load_results_table,
    parse_gtf_models,
    read_config,
    setup_logger,
    split_utr_cds,
    translate_cds,
    write_fasta,
)


def main():
    parser = argparse.ArgumentParser(description="Extract transcript/UTR/CDS/protein sequences and basic lengths")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("01_extract_transcript_features", Path(args.log) if args.log else None)

    results = load_results_table(cfg["inputs"]["results_csv"])
    models = parse_gtf_models(cfg["inputs"]["annotation_gtf"], set(results["transcript_id_base"]))
    genome = Fasta(cfg["inputs"]["genome_fasta"], as_raw=True, sequence_always_upper=True)

    tx_records, five_records, cds_records, three_records, prot_records = {}, {}, {}, {}, {}
    rows = []

    for _, row in results.iterrows():
        tidb = row["transcript_id_base"]
        if tidb not in models:
            continue
        m = models[tidb]
        seqs = split_utr_cds(m, genome)
        prot = translate_cds(seqs["cds"])

        tx_records[tidb] = seqs["transcript"]
        five_records[tidb] = seqs["five_utr"]
        cds_records[tidb] = seqs["cds"]
        three_records[tidb] = seqs["three_utr"]
        prot_records[tidb] = prot

        rows.append(
            {
                "gene_name": row["gene_name"],
                "transcript_id": row["transcript_id"],
                "transcript_id_base": tidb,
                "gene_id": m.gene_id,
                "gene_name_gtf": m.gene_name,
                "chrom": m.chrom,
                "strand": m.strand,
                "transcript_type": m.transcript_type,
                "exon_coordinates": ";".join([f"{a}-{b}" for a, b in m.exons]),
                "cds_coordinates": ";".join([f"{a}-{b}" for a, b in m.cds]),
                "start_codon_coordinates": ";".join([f"{a}-{b}" for a, b in m.start_codon]),
                "stop_codon_coordinates": ";".join([f"{a}-{b}" for a, b in m.stop_codon]),
                "M_mean": row["M_mean"],
                "R_mean": row["R_mean"],
                "log2FC": row["log2FC"],
                "IF_Total": row["IF_Total"],
                "IF_Ribo": row["IF_Ribo"],
                "Delta_IF": row["Delta_IF"],
                "transcript_length": len(seqs["transcript"]),
                "exon_count": len(m.exons),
                "five_utr_length": len(seqs["five_utr"]),
                "cds_length": len(seqs["cds"]),
                "three_utr_length": len(seqs["three_utr"]),
                "protein_length": len(prot),
                "coding_status": "coding" if len(seqs["cds"]) > 0 else "noncoding",
            }
        )

    out_seq = Path(cfg["outputs"]["sequences_dir"])
    out_seq.mkdir(parents=True, exist_ok=True)
    write_fasta(tx_records, out_seq / "transcript.fa")
    write_fasta(five_records, out_seq / "five_utr.fa")
    write_fasta(cds_records, out_seq / "cds.fa")
    write_fasta(three_records, out_seq / "three_utr.fa")
    write_fasta(prot_records, out_seq / "protein.fa")

    out_table = Path(cfg["outputs"]["tables_dir"]) / "transcript_basic_features.tsv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_table, sep="\t", index=False)
    logger.info("Extracted basic features for %d transcripts", len(rows))
    logger.info("Sequences written to %s", out_seq)


if __name__ == "__main__":
    main()
