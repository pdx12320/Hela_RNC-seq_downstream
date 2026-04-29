#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_step(script: str, config_path: str, extra_args: list[str] | None = None):
    cmd = [sys.executable, script, "--config", config_path]
    if extra_args:
        cmd.extend(extra_args)
    print("[RUN]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(f"Step failed: {script}")


def main():
    ap = argparse.ArgumentParser(description="Run full RNC isoform feature pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--ddif-threshold",
        type=float,
        default=None,
        help="Candidate pair filter threshold for abs(delta_Delta_IF). "
        "Default: params.ddif_threshold in config, fallback 0.0.",
    )
    args = ap.parse_args()

    cfg_path = args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ddif_threshold = (
        args.ddif_threshold
        if args.ddif_threshold is not None
        else float(cfg.get("params", {}).get("ddif_threshold", 0.0))
    )

    for d in [cfg["output"]["base_dir"], cfg["output"]["sequences_dir"], cfg["output"]["tables_dir"], cfg["output"]["figures_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    steps = [
        "scripts/00_prepare_annotation.py",
        "scripts/01_extract_transcript_features.py",
        "scripts/02_scan_sequence_motifs.py",
        "scripts/03_predict_orf_uorf_nmd.py",
        "scripts/04_run_rnafold.py",
        "scripts/05_compare_isoforms_within_gene.py",
        "scripts/06_plot_results.py",
    ]
    for s in steps:
        if s.endswith("05_compare_isoforms_within_gene.py") or s.endswith("06_plot_results.py"):
            run_step(s, cfg_path, ["--ddif-threshold", str(ddif_threshold)])
        else:
            run_step(s, cfg_path)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
