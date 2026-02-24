#!/usr/bin/env python3
"""Create a SHA-256 manifest (CSV) for a directory tree.


Output CSV columns:
- file: POSIX-style relative path from --root (uses forward slashes, even on Windows)
- sha256: hex digest
- size_bytes: file size in bytes

Usage:
    python create_master_manifest.py --root . --output MASTER_MANIFEST.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Tuple

DEFAULT_EXCLUDES = [
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
]

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 in chunks to avoid loading large files into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def should_skip(rel_parts: Tuple[str, ...], exclude_names: Iterable[str]) -> bool:
    """Skip if any path component matches an excluded directory/name."""
    return any(part in exclude_names for part in rel_parts)

def build_manifest_rows(root: Path, exclude_names: List[str], skip_files: List[str]) -> List[dict]:
    rows: List[dict] = []
    root = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        rel_dir = dirpath_p.relative_to(root)

        # prune excluded directories in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in exclude_names]

        if should_skip(tuple(rel_dir.parts), exclude_names):
            continue

        for fname in filenames:
            if fname in skip_files:
                continue

            fpath = dirpath_p / fname
            relpath = fpath.relative_to(root)

            # POSIX-style path for cross-platform manifests
            rel_posix = PurePosixPath(*relpath.parts).as_posix()

            rows.append(
                {
                    "file": rel_posix,
                    "sha256": sha256_file(fpath),
                    "size_bytes": fpath.stat().st_size,
                }
            )

    rows.sort(key=lambda r: r["file"])
    return rows

def write_manifest(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"])
        w.writeheader()
        w.writerows(rows)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="Directory to manifest (default: current directory).")
    ap.add_argument("--output", type=Path, default=Path("MASTER_MANIFEST.csv"), help="Output CSV path.")
    ap.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Directory or file name to exclude (repeatable). Default excludes common cache dirs.",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    exclude_names = DEFAULT_EXCLUDES + list(args.exclude_name)

    # Don't include the output file itself if it's inside root
    skip_files = [args.output.name]

    rows = build_manifest_rows(root, exclude_names=exclude_names, skip_files=skip_files)
    out_csv = args.output
    if not out_csv.is_absolute():
        out_csv = root / out_csv

    write_manifest(rows, out_csv)

    total_bytes = sum(int(r["size_bytes"]) for r in rows)
    print(f"Master manifest written: {out_csv}")
    print(f"Files: {len(rows)}")
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
