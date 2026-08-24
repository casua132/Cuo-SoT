"""Download the PersonaMem benchmark data files into ``benchmark/`` if they are missing.

Usage
-----
    python data_download.py                 # all sizes
    python data_download.py --size 32k      # one size
    python data_download.py --force         # re-download even if present
"""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download

from utils import BENCHMARK_DIR, BENCHMARK_SIZES, context_jsonl_path, question_csv_path

REPO_ID = "bowen-upenn/PersonaMem-v1"
REPO_TYPE = "dataset"

FILES = {size: (f"questions_{size}.csv", f"shared_contexts_{size}.jsonl") for size in BENCHMARK_SIZES}


def _already_present(size: str, filename: str) -> bool:
    path = question_csv_path(size) if filename.startswith("questions") else context_jsonl_path(size)
    return path.exists()


def download(size: str | None = None, force: bool = False) -> list[str]:
    """Download the benchmark files for ``size`` (or all sizes) into ``benchmark/``.

    Returns the list of paths downloaded (or skipped).
    """
    sizes = [size] if size else list(BENCHMARK_SIZES)
    paths = []
    for s in sizes:
        for filename in FILES[s]:
            if not force and _already_present(s, filename):
                print(f"[data_download] {filename} already present, skipping")
                continue
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=str(BENCHMARK_DIR),
            )
            paths.append(path)
            print(f"[data_download] downloaded {path}")
    if not paths:
        print("[data_download] nothing to do; all files present (use --force to re-download).")
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download PersonaMem benchmark data into benchmark/.")
    parser.add_argument("--size", choices=BENCHMARK_SIZES, default=None, help="only this size (default: all)")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args(argv)
    download(size=args.size, force=args.force)


if __name__ == "__main__":
    main()
