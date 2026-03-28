#!/usr/bin/env python3
"""
NanoSplat :: Model Downloader
==============================
Downloads MiDaS-Small ONNX model for depth estimation.
Run this once before starting NanoSplat.

Usage:
  python nanosplat/scripts/download_models.py
  python nanosplat/scripts/download_models.py --model-dir /path/to/models
"""

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# MiDaS Small ONNX — from isl-org's official release
MODELS = {
    "midas_small.onnx": {
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        "size_mb": 21,
        "md5": None,  # skip checksum for simplicity
    },
}

def download(url: str, dest: Path, size_mb: int):
    log.info(f"Downloading {dest.name} (~{size_mb} MB)...")

    def progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(100, 100 * downloaded / total_size)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()
    log.info(f"Saved: {dest}  ({dest.stat().st_size / 1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--force",     action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, info in MODELS.items():
        dest = model_dir / name
        if dest.exists() and not args.force:
            log.info(f"Already exists: {dest}  (use --force to re-download)")
            continue
        try:
            download(info["url"], dest, info["size_mb"])
        except Exception as e:
            log.error(f"Download failed: {e}")
            log.error(f"Manual download: {info['url']}")
            sys.exit(1)

    log.info("\n✅  All models ready.")
    log.info("Next step:  python main.py --target <object>")


if __name__ == "__main__":
    main()
