#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gffutils
import pandas as pd

from utils import ensure_dirs, load_config, normalize_transcript_id_column, setup_logger, transcript_base_id


def main():
    ap = argparse.ArgumentParser(description="Prepare annotation DB and match transcripts")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs([cfg["output"]["tables_dir"], cfg["output"]["sequences_dir"], cfg["output"]["figures_dir"]])
    logger = setup_logger("00_prepare", os.path.join(cfg["output"]["tables_dir"], "00_prepare_annotation.log"))

    results = pd.read_csv(cfg["input"]["results_csv"])
    results = normalize_transcript_id_column(results)

    gtf = cfg["input"]["annotation_gtf"]
    db_path = cfg["params"].get("gffutils_db", os.path.join(cfg["output"]["tables_dir"], "annotation.db"))

    if not os.path.exists(db_path):
        logger.info("Building gffutils DB: %s", db_path)
        gffutils.create_db(
            gtf,
            dbfn=db_path,
            force=True,
            keep_order=True,
            disable_infer_genes=True,
            disable_infer_transcripts=True,
            merge_strategy="merge",
            sort_attribute_values=True,
        )
    else:
        logger.info("Using existing gffutils DB: %s", db_path)

    db = gffutils.FeatureDB(db_path, keep_order=True)

    tx_ids = set()
    tx_base = set()
    for tx in db.features_of_type("transcript"):
        txid = tx.attributes.get("transcript_id", [tx.id])[0]
        tx_ids.add(txid)
        tx_base.add(transcript_base_id(txid))

    rows = []
    missing = []
    for _, r in results.iterrows():
        raw = str(r["transcript_id"])
        base = str(r["transcript_id_base"])
        matched = raw if raw in tx_ids else (base if base in tx_base else None)
        rows.append({"transcript_id": raw, "transcript_id_base": base, "matched_transcript_id": matched})
        if matched is None:
            missing.append({"transcript_id": raw, "transcript_id_base": base, "gene_name": r.get("gene_name", "NA")})

    match_df = pd.DataFrame(rows)
    match_df.to_csv(os.path.join(cfg["output"]["tables_dir"], "transcript_match_table.tsv"), sep="\t", index=False)
    pd.DataFrame(missing).to_csv(os.path.join(cfg["output"]["tables_dir"], "missing_transcripts.tsv"), sep="\t", index=False)

    merged = results.merge(match_df, on=["transcript_id", "transcript_id_base"], how="left")
    merged.to_csv(os.path.join(cfg["output"]["tables_dir"], "results_with_match.tsv"), sep="\t", index=False)
    logger.info("Input transcripts: %d; matched: %d; missing: %d", len(results), merged["matched_transcript_id"].notna().sum(), len(missing))


if __name__ == "__main__":
    main()
