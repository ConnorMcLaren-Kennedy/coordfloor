#!/usr/bin/env python3
"""Verify file integrity using SHA-256 manifests.

Two common modes:

1) Verify all release manifests:
    python verify_checksums.py --root . --all-releases

   This expects:
     <root>/releases/<release_name>/manifest_sha256.csv

2) Verify a single manifest:
    python verify_checksums.py --manifest path/to/manifest_sha256.csv --base-dir path/to/release

The manifest format is CSV with columns:
  file, sha256, size_bytes

Paths in the manifest are interpreted as POSIX-style relative paths (forward slashes),
so manifests created on Windows can be verified on macOS/Linux and vice versa.
Backslashes are normalized automatically.

Exit code:
  0 = all checks passed
  1 = one or more mismatches/missing files
  2 = configuration error (missing dirs/manifests)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
from typing import List, Tuple

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def read_manifest(manifest_path: Path) -> List[dict]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    required = {"file", "sha256", "size_bytes"}
    if not required.issubset(set(r.fieldnames or [])):
        raise ValueError(f"Manifest missing required columns {required}: {manifest_path}")
    return rows

def verify_manifest(manifest_path: Path, base_dir: Path, manifest_name: str = "manifest_sha256.csv") -> Tuple[int, List[str]]:
    rows = read_manifest(manifest_path)
    errors: List[str] = []
    checked = 0

    for row in rows:
        rel = row["file"]
        rel_norm = str(rel).replace("\\", "/")

        if rel_norm == manifest_name:
            continue

        rel_posix = PurePosixPath(rel_norm)
        filepath = base_dir / Path(*rel_posix.parts)

        if not filepath.exists():
            errors.append(f"MISSING: {rel_norm}")
            continue

        actual_hash = sha256_file(filepath)
        if actual_hash.lower() != str(row["sha256"]).strip().lower():
            errors.append(f"MISMATCH: {rel_norm}")
            continue

        checked += 1

    return checked, errors

def verify_all_releases(root: Path, releases_dirname: str, manifest_name: str) -> int:
    releases_dir = root / releases_dirname
    if not releases_dir.exists():
        print(f"ERROR: releases directory not found: {releases_dir}")
        return 2

    releases = sorted([d for d in releases_dir.iterdir() if d.is_dir()])
    if not releases:
        print(f"WARNING: no release folders found under {releases_dir}")
        return 0

    print("=" * 72)
    print("INTEGRITY VERIFICATION (SHA-256)")
    print("=" * 72)

    total_checked = 0
    total_errors = 0
    missing_manifests = 0

    for release in releases:
        manifest_path = release / manifest_name
        if not manifest_path.exists():
            print(f"\n{release.name}: NO MANIFEST ({manifest_name})")
            missing_manifests += 1
            continue

        checked, errors = verify_manifest(manifest_path, base_dir=release, manifest_name=manifest_name)
        if errors:
            print(f"\n{release.name}: {len(errors)} ERRORS")
            for e in errors:
                print(f"  {e}")
            total_errors += len(errors)
        else:
            print(f"\n{release.name}: {checked} files OK")
            total_checked += checked

    print("\n" + "=" * 72)
    print(f"SUMMARY: {total_checked} files verified, {total_errors} errors, {missing_manifests} releases missing manifests")
    print("=" * 72)

    if missing_manifests > 0:
        print("\nNOTE: Some releases are missing manifests. Run create_all_manifests.py first.")
        return 2

    return 0 if total_errors == 0 else 1

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="Project/handoff root (default: current directory).")
    ap.add_argument("--releases-dir", type=str, default="releases", help="Folder name under root containing releases.")
    ap.add_argument("--manifest-name", type=str, default="manifest_sha256.csv", help="Manifest filename.")
    ap.add_argument("--all-releases", action="store_true", help="Verify all release manifests under <root>/<releases-dir>.")
    ap.add_argument("--manifest", type=Path, default=None, help="Path to a single manifest to verify.")
    ap.add_argument("--base-dir", type=Path, default=None, help="Base directory for --manifest (defaults to manifest parent).")
    args = ap.parse_args()

    root = args.root.resolve()

    if args.all_releases:
        return verify_all_releases(root, releases_dirname=args.releases_dir, manifest_name=args.manifest_name)

    if args.manifest is None:
        print("ERROR: Choose either --all-releases or --manifest <path>")
        return 2

    manifest_path = args.manifest.resolve()
    base_dir = (args.base_dir.resolve() if args.base_dir else manifest_path.parent.resolve())

    checked, errors = verify_manifest(manifest_path, base_dir=base_dir, manifest_name=args.manifest_name)
    if errors:
        print(f"{manifest_path}: {len(errors)} ERRORS")
        for e in errors:
            print(f"  {e}")
        return 1

    print(f"{manifest_path}: OK ({checked} files verified)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
