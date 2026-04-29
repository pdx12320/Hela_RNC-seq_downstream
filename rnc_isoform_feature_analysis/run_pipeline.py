#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full RNC isoform feature analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    scripts = [
        "00_prepare_annotation.py",
        "01_extract_transcript_features.py",
        "02_scan_sequence_motifs.py",
        "03_predict_orf_uorf_nmd.py",
        "04_run_rnafold.py",
        "05_compare_isoforms_within_gene.py",
        "06_plot_results.py",
    ]
    log_dir = base / "output" / "tables"
    log_dir.mkdir(parents=True, exist_ok=True)

    for s in scripts:
        run([args.python, str(base / "scripts" / s), "--config", args.config, "--log", str(log_dir / f"{s}.log")])

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
