#!/usr/bin/env python3
"""scripts/03b_full_pairwise_fidelity.py

Compute directional cross-reconstruction fidelity for all ordered speed
pairs (a -> b) within each (dataset, subject, trial) group.

This complements `03_compute_fidelity.py`, which only compares the fastest
reference speed to slower speeds.

Inputs
------
* outputs/stability_summary.csv (from 02c_nmf_stability.py)

Outputs
-------
* outputs/full_pairwise_fidelity.csv
* outputs/full_pairwise_fidelity_summary.txt
* outputs/full_pairwise_fidelity_params.txt

Definition
----------
For an ordered pair (a -> b), with synergy weights W_a and activation data X_b:

    H = argmin_{H>=0} ||X_b - W_a H||_F^2  (solved column-wise with NNLS)
    fidelity(a->b) = 1 - ||X_b - W_a H||_F^2 / ||X_b||_F^2

The NNLS solver is scipy.optimize.nnls, applied independently to each time
sample (column).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import nnls

N_POINTS_DEFAULT = 200

OUTPUT_COLUMNS = [
    "dataset",
    "subject",
    "trial",
    "speed_a_mps",
    "speed_b_mps",
    "k_a",
    "k_b",
    "n_strides_a",
    "n_strides_b",
    "fidelity_a_to_b",
    "notes",
]


def truthy(val) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


def load_npz_key(npz_path: Path, key: str) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")
    data = np.load(str(npz_path))
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path}. Keys={list(data.keys())}")
    return data[key]


def load_X(file_x_path: Path) -> np.ndarray:
    return load_npz_key(file_x_path, "X")


def load_W(file_npz_path: Path) -> np.ndarray:
    return load_npz_key(file_npz_path, "W")


def solve_H_nnls(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Solve H>=0 minimizing ||X - W @ H||_F^2 with W fixed.

    W: (m, k)
    X: (m, n_obs)
    H: (k, n_obs)
    """
    k = W.shape[1]
    n_obs = X.shape[1]
    H = np.zeros((k, n_obs), dtype=float)
    for j in range(n_obs):
        H[:, j], _ = nnls(W, X[:, j])
    return H


def compute_fidelity(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """fidelity = 1 - ||X - W @ H||_F^2 / ||X||_F^2"""
    residual = X - (W @ H)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum(X**2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def count_strides(X: np.ndarray, n_points: int) -> int:
    return int(X.shape[1] // n_points)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute directional cross-speed fidelity for all ordered speed pairs via NNLS cross-reconstruction."
    )
    parser.add_argument(
        "--rel7",
        required=True,
        help="Path to upstream folder containing outputs/stability_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <this_release>/outputs)",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=N_POINTS_DEFAULT,
        help="Samples per stride (default: 200)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files in out_dir.",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    release_dir = script_path.parents[1]  # .../<release_name>/
    default_out_dir = release_dir / "outputs"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir

    rel7_dir = Path(args.rel7).resolve()
    stability_path = rel7_dir / "outputs" / "stability_summary.csv"
    if not stability_path.exists():
        print(f"ERROR: stability_summary.csv not found: {stability_path}")
        return 1

    out_csv = out_dir / "full_pairwise_fidelity.csv"
    out_summary = out_dir / "full_pairwise_fidelity_summary.txt"
    out_params = out_dir / "full_pairwise_fidelity_params.txt"

    if not args.overwrite:
        for p in (out_csv, out_summary, out_params):
            if p.exists():
                print(f"ERROR: output already exists: {p}")
                print("       Use --overwrite or choose a fresh --out_dir.")
                return 1

    start_time = time.time()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("FULL PAIRWISE COORDINATION FIDELITY")
    print("=" * 50)
    print(f"Run timestamp: {run_ts}")
    print(f"Upstream: {rel7_dir}")
    print(f"OUT_DIR: {out_dir}")
    print(f"N_POINTS: {args.n_points}")
    print(f"python: {sys.version.split()[0]}")
    print(f"numpy: {np.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"pandas: {pd.__version__}")
    print("")

    df = pd.read_csv(stability_path)
    n_total = len(df)
    required_cols = [
        "dataset",
        "subject",
        "trial",
        "speed_mps",
        "file_x",
        "file_npz",
        "pass_stability",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: stability_summary.csv missing required columns: {missing_cols}")
        return 1

    df = df[df["pass_stability"].apply(truthy)].copy()
    n_pass = len(df)
    if n_pass == 0:
        print("ERROR: No rows with pass_stability == True after truthy parsing.")
        return 1

    df["speed_mps"] = df["speed_mps"].astype(float)
    # In upstream scripts, trial is a string (e.g., "00"); casting to float makes grouping stable.
    df["trial"] = df["trial"].astype(float)

    group_cols = ["dataset", "subject", "trial"]
    groups = df.groupby(group_cols, sort=True)

    speeds_per_group = df.groupby(group_cols)["speed_mps"].nunique()
    n_groups = int(len(speeds_per_group))
    n_groups_multi_speed = int((speeds_per_group > 1).sum())

    print(f"Loaded stability_summary.csv: total_rows={n_total} pass_rows={n_pass}")
    print(f"Unique groups (dataset, subject, trial): {n_groups}")
    print(f"Groups with >1 speed (pairs possible): {n_groups_multi_speed}")
    print("")

    results = []
    n_groups_processed = 0
    n_skipped_single_speed = 0
    n_pairs = 0

    for (dataset, subject, trial), grp in groups:
        n_groups_processed += 1
        speeds = sorted(grp["speed_mps"].unique())
        if len(speeds) < 2:
            n_skipped_single_speed += 1
            continue

        # Load all X and W once per speed
        X_by_speed = {}
        W_by_speed = {}
        k_by_speed = {}
        n_strides_by_speed = {}

        for sp in speeds:
            row = grp[grp["speed_mps"] == sp]
            if len(row) != 1:
                raise RuntimeError(f"Expected exactly 1 row for speed={sp}, got {len(row)}")
            row = row.iloc[0]
            X = load_X(Path(row["file_x"]))
            W = load_W(Path(row["file_npz"]))
            X_by_speed[sp] = X
            W_by_speed[sp] = W
            k_by_speed[sp] = int(W.shape[1])
            n_strides_by_speed[sp] = count_strides(X, args.n_points)

        # Sanity check muscle dimension harmonisation within group
        m0 = next(iter(X_by_speed.values())).shape[0]
        for sp in speeds:
            if X_by_speed[sp].shape[0] != m0 or W_by_speed[sp].shape[0] != m0:
                raise AssertionError(
                    f"Muscle dimension mismatch within group {dataset}/{subject}/trial={trial}: "
                    f"speed={sp} X_rows={X_by_speed[sp].shape[0]} W_rows={W_by_speed[sp].shape[0]} expected={m0}"
                )

        print(
            f"Processing {dataset}/{subject}/trial={trial:.1f} speeds={speeds} "
            f"n_pairs={len(speeds) * (len(speeds) - 1)}"
        )

        # All ordered pairs
        for sp_a in speeds:
            W_a = W_by_speed[sp_a]
            k_a = k_by_speed[sp_a]
            n_strides_a = n_strides_by_speed[sp_a]
            for sp_b in speeds:
                if sp_b == sp_a:
                    continue
                X_b = X_by_speed[sp_b]
                W_b = W_by_speed[sp_b]
                k_b = k_by_speed[sp_b]
                n_strides_b = n_strides_by_speed[sp_b]

                # Required muscle-dimension checks
                assert W_a.shape[0] == X_b.shape[0], (
                    f"Muscle dimension mismatch: W_a rows={W_a.shape[0]} != X_b rows={X_b.shape[0]} "
                    f"for {dataset}/{subject}/trial={trial} {sp_a}->{sp_b}"
                )
                assert W_b.shape[0] == X_b.shape[0], (
                    f"Muscle dimension mismatch: W_b rows={W_b.shape[0]} != X_b rows={X_b.shape[0]} "
                    f"for {dataset}/{subject}/trial={trial} speed_b={sp_b}"
                )

                H_b_from_a = solve_H_nnls(W_a, X_b)
                fidelity_a_to_b = compute_fidelity(X_b, W_a, H_b_from_a)

                results.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "trial": float(trial),
                        "speed_a_mps": float(sp_a),
                        "speed_b_mps": float(sp_b),
                        "k_a": int(k_a),
                        "k_b": int(k_b),
                        "n_strides_a": int(n_strides_a),
                        "n_strides_b": int(n_strides_b),
                        "fidelity_a_to_b": float(fidelity_a_to_b),
                        "notes": "",
                    }
                )
                n_pairs += 1

    df_out = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    if len(df_out) > 0:
        df_out = df_out.sort_values(
            by=["dataset", "subject", "trial", "speed_a_mps", "speed_b_mps"],
            kind="mergesort",
        ).reset_index(drop=True)

    elapsed = time.time() - start_time

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    # Summary stats
    if len(df_out) > 0:
        fmin = float(df_out["fidelity_a_to_b"].min())
        fmax = float(df_out["fidelity_a_to_b"].max())
        fmean = float(df_out["fidelity_a_to_b"].mean())
    else:
        fmin = fmax = fmean = float("nan")

    ds_counts = df_out["dataset"].value_counts().to_dict() if len(df_out) > 0 else {}

    summary_lines = []
    summary_lines.append("Full Pairwise Coordination Fidelity Summary")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    summary_lines.append(f"Run timestamp: {run_ts}")
    summary_lines.append(f"Upstream (input): {rel7_dir}")
    summary_lines.append(f"OUT_DIR: {out_dir}")
    summary_lines.append(f"N_POINTS: {args.n_points}")
    summary_lines.append("")
    summary_lines.append(f"Total groups processed: {n_groups_processed}")
    summary_lines.append(f"Groups skipped (single speed): {n_skipped_single_speed}")
    summary_lines.append(f"Total ordered pairs computed: {n_pairs}")
    summary_lines.append(f"Output rows: {len(df_out)}")
    summary_lines.append(f"Runtime: {elapsed:.2f} seconds")
    summary_lines.append("")
    summary_lines.append("Fidelity statistics (directional):")
    summary_lines.append(f"  fidelity_a_to_b: min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}")
    summary_lines.append("")
    summary_lines.append("By dataset:")
    if ds_counts:
        for k, v in sorted(ds_counts.items()):
            summary_lines.append(f"  {k}: {v} ordered pairs")
    else:
        summary_lines.append("  (none)")
    summary_text = "\n".join(summary_lines) + "\n"
    write_text(out_summary, summary_text)

    params_lines = []
    params_lines.append("Full Pairwise Coordination Fidelity Parameters")
    params_lines.append("=" * 50)
    params_lines.append("")
    params_lines.append(f"Run timestamp: {run_ts}")
    params_lines.append(f"Script: {script_path}")
    params_lines.append(f"Output folder: {release_dir}")
    params_lines.append(f"Upstream: {rel7_dir}")
    params_lines.append(f"OUT_DIR: {out_dir}")
    params_lines.append(f"N_POINTS (samples per stride): {args.n_points}")
    params_lines.append("Algorithm: scipy.optimize.nnls (column-wise)")
    params_lines.append("Fidelity formula: 1 - ||X - W @ H||_F^2 / ||X||_F^2")
    params_lines.append("Pairs: all ordered (speed_a != speed_b) within each group")
    params_lines.append("Filter: pass_stability parsed as truthy (true/1/yes/y/t)")
    params_lines.append(f"python: {sys.version.split()[0]}")
    params_lines.append(f"numpy: {np.__version__}")
    params_lines.append(f"scipy: {scipy.__version__}")
    params_lines.append(f"pandas: {pd.__version__}")
    params_text = "\n".join(params_lines) + "\n"
    write_text(out_params, params_text)

    print("")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_params}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
