#!/usr/bin/env python3
"""
Download the measurement-db dataset from HuggingFace.

Source: https://huggingface.co/datasets/aims-foundation/torch-measure-data
License: See dataset page for license details.

This script downloads the 16 public/audited benchmarks plus the registry
files (subjects, items, benchmarks) needed to run the pipeline.

Usage:
    python download_data.py
"""

import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError(
        "huggingface-hub is required. Install with: pip install huggingface-hub"
    )

REPO_ID = "aims-foundation/torch-measure-data"
REPO_TYPE = "dataset"
DATA_DIR = Path(__file__).parent / "data"

# Registry files
REGISTRY_FILES = [
    "subjects.parquet",
    "items.parquet",
    "benchmarks.parquet",
]

# The 16 audited benchmarks used in this project
BENCHMARKS = [
    "afrimedqa",
    "agentdojo",
    "ai2d_test",
    "androidworld",
    "bfcl",
    "cybench",
    "hle",
    "livecodebench",
    "matharena",
    "mathvista_mini",
    "mmbench_v11",
    "mmlupro",
    "mtbench",
    "rewardbench",
    "swebench",
    "ultrafeedback",
]


def download_file(filename: str) -> None:
    """Download a single file from the HuggingFace dataset repo."""
    dest = DATA_DIR / filename
    if dest.exists():
        print(f"  [skip] {filename} (already exists)")
        return
    print(f"  [download] {filename}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=DATA_DIR,
    )


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Downloading registry files...")
    for f in REGISTRY_FILES:
        download_file(f)

    print(f"\nDownloading {len(BENCHMARKS)} benchmark response files...")
    for bm in BENCHMARKS:
        download_file(f"{bm}.parquet")

    print(f"\nDone. Data saved to {DATA_DIR}/")
    print(f"Total files: {len(list(DATA_DIR.glob('*.parquet')))}")


if __name__ == "__main__":
    main()
