#!/usr/bin/env python3
"""
Download the glaucoma screening checkpoint for the openDR pipeline.

Fetches ``convnext_tiny.pt`` (~111 MB) and ``convnext_tiny.json`` from the
``tiagopessoalim/glaucoma`` Hugging Face Space and writes them to
``<OPEN_DR_BASE>/models/`` (or ``--dest``), the location
:mod:`modules.glaucoma` looks in by default.

The checkpoint is not committed to this repository — this script is the
supported way to fetch it. Uses ``requests`` (already an openDR dependency
via ``modules.theia``) with a streaming GET; no extra dependency needed.

Usage::

    python tools/download_glaucoma_model.py
    python tools/download_glaucoma_model.py --dest /path/to/models
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

_SPACE_RESOLVE_BASE = (
    "https://huggingface.co/spaces/tiagopessoalim/glaucoma/resolve/main/models"
)
_FILES = ("convnext_tiny.pt", "convnext_tiny.json")
_CHUNK_SIZE = 1024 * 1024
_REQUEST_TIMEOUT_S = 30


def _default_dest() -> Path:
    base_folder = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
    return base_folder / "models"


def _download_one(filename: str, dest_dir: Path) -> None:
    url = f"{_SPACE_RESOLVE_BASE}/{filename}"
    dest_path = dest_dir / filename
    tmp_path = dest_dir / f"{filename}.part"

    print(f"Downloading {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT_S) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", 0))
        written_bytes = 0

        with tmp_path.open("wb") as tmp_file:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if not chunk:
                    continue
                tmp_file.write(chunk)
                written_bytes += len(chunk)
                if total_bytes:
                    percent = 100 * written_bytes / total_bytes
                    print(f"\r  {written_bytes / 1e6:.1f} MB / {total_bytes / 1e6:.1f} MB ({percent:.0f}%)", end="")

    print()
    if total_bytes and written_bytes != total_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download incomplete for {filename}: got {written_bytes} bytes, "
            f"expected {total_bytes}."
        )
    tmp_path.replace(dest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: <OPEN_DR_BASE>/models).",
    )
    args = parser.parse_args()

    dest_dir = args.dest or _default_dest()
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in _FILES:
        try:
            _download_one(filename, dest_dir)
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            print(f"ERROR: failed to download {filename}: {exc}", file=sys.stderr)
            return 1

    print(f"Done. Glaucoma model files are in {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
