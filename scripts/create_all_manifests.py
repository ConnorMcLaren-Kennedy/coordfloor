#!/usr/bin/env python3
"""Create manifest_sha256.csv for each release folder under <root>/<releases-dir>/.

Each manifest is written inside the release folder:
  <release>/manifest_sha256.csv

Manifest paths are written POSIX-style (forward slashes) so manifests can be
generated on Windows and verified on macOS/Linux, and vice versa.

Usage:
    python create_all_manifests.py --root .                 # default releases dir is ./releases
    python create_all_manifests.py --root . --overwrite     # overwrite any existing manifests
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import List

DEFAULT_EXCLUDES = [
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
]

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def create_manifest(release_path: Path, manifest_name: str, exclude_names: List[str]) -> int:
    rows = []
    release_path = release_path.resolve()

    for dirpath, dirnames, filenames in os.walk(release_path):
        dirpath_p = Path(dirpath)

        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in exclude_names]

        for fname in filenames:
            if fname == manifest_name:
                continue
            fpath = dirpath_p / fname
            relpath = fpath.relative_to(release_path)
            rel_posix = PurePosixPath(*relpath.parts).as_posix()
            rows.append(
                {
                    "file": rel_posix,
                    "sha256": sha256_file(fpath),
                    "size_bytes": fpath.stat().st_size,
                }
            )

    rows.sort(key=lambda r: r["file"])
    manifest_path = release_path / manifest_name
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"])
        w.writeheader()
        w.writerows(rows)

    return len(rows)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="Project/handoff root (default: current directory).")
    ap.add_argument("--releases-dir", type=str, default="releases", help="Folder name under root containing releases.")
    ap.add_argument("--manifest-name", type=str, default="manifest_sha256.csv", help="Manifest filename to write.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite manifests if they already exist.")
    ap.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Directory name to exclude (repeatable). Defaults exclude common cache dirs.",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    releases_dir = root / args.releases_dir
    if not releases_dir.exists():
        print(f"ERROR: releases directory not found: {releases_dir}")
        return 2

    exclude_names = DEFAULT_EXCLUDES + list(args.exclude_name)

    releases = sorted([d for d in releases_dir.iterdir() if d.is_dir()])
    if not releases:
        print(f"WARNING: no release folders found under {releases_dir}")
        return 0

    print(f"Creating manifests under: {releases_dir}")
    total = 0
    for release in releases:
        manifest_path = release / args.manifest_name
        if manifest_path.exists() and not args.overwrite:
            print(f"{release.name}: SKIP (manifest exists)")
            continue
        n_files = create_manifest(release, manifest_name=args.manifest_name, exclude_names=exclude_names)
        total += 1
        print(f"{release.name}: wrote {args.manifest_name} for {n_files} files")

    print(f"Done. Processed {total} release folders.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
