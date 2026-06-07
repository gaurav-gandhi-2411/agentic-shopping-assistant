"""
Download the Myntra Fashion Product Dataset from Kaggle.

Pre-requisites:
    pip install kaggle
    # Place your API key at ~/.kaggle/kaggle.json (chmod 600)
    # Or set KAGGLE_USERNAME and KAGGLE_KEY env vars.

Usage:
    python scripts/download_myntra.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "hiteshsuthar101/myntra-fashion-product-dataset"
RAW_DIR = Path("data/raw/myntra")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_SLUG} -> {RAW_DIR} ...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets", "download",
            "-d", DATASET_SLUG,
            "--unzip",
            "-p", str(RAW_DIR),
        ],
        check=False,
    )
    if result.returncode != 0:
        print(
            "\n[ERROR] kaggle download failed.\n"
            "  1. Install the CLI:  pip install kaggle\n"
            "  2. Create token at https://www.kaggle.com/settings > API > Create New Token\n"
            "  3. Save it to ~/.kaggle/kaggle.json  (chmod 600 on Linux/Mac)\n"
            "  4. Re-run this script.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nDownloaded files:")
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(RAW_DIR)}  ({p.stat().st_size / 1024:.0f} KB)")

    print(
        f"\nSet catalogue_path in brands/myntra.yaml to the CSV path shown above.\n"
        f"Default assumed: {RAW_DIR / 'Myntra Fashion Products.csv'}"
    )


if __name__ == "__main__":
    main()
