#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import add_common_cli_args, load_results_table, parse_gtf_models, read_config, setup_logger


def main():
    parser = argparse.ArgumentParser(description="Prepare annotation subset and detect missing transcripts")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("00_prepare_annotation", Path(args.log) if args.log else None)

    results = load_results_table(cfg["inputs"]["results_csv"])
    requested = set(results["transcript_id_base"])
    models = parse_gtf_models(cfg["inputs"]["annotation_gtf"], requested)
    found = set(models.keys())

    missing = sorted(requested - found)
    out_missing = Path(cfg["outputs"]["tables_dir"]) / "missing_transcripts.tsv"
    out_missing.parent.mkdir(parents=True, exist_ok=True)
    miss_df = results[results["transcript_id_base"].isin(missing)].copy()
    miss_df.to_csv(out_missing, sep="\t", index=False)

    meta = []
    for tid, m in models.items():
        meta.append(
            {
                "transcript_id_base": tid,
                "transcript_id": m.transcript_id,
                "gene_id": m.gene_id,
                "gene_name_gtf": m.gene_name,
                "chrom": m.chrom,
                "strand": m.strand,
                "transcript_type": m.transcript_type,
                "exon_count": len(m.exons),
                "cds_exon_count": len(m.cds),
            }
        )
    pd.DataFrame(meta).to_csv(Path(cfg["outputs"]["tables_dir"]) / "transcript_annotation_summary.tsv", sep="\t", index=False)
    logger.info("Requested transcripts: %d", len(requested))
    logger.info("Matched transcripts in GTF: %d", len(found))
    logger.info("Missing transcripts: %d -> %s", len(missing), out_missing)


if __name__ == "__main__":
    main()
