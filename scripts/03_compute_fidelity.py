#!/usr/bin/env python3
"""
Script: 03_compute_fidelity.py
Purpose: Compute bidirectional cross-reconstruction fidelity between speeds.

Inputs:
    - outputs/stability_summary.csv (from 02c_nmf_stability.py)

Outputs:
    - outputs/coordination_fidelity.csv
    - outputs/fidelity_summary.txt
    - outputs/fidelity_params.txt

Algorithm:
    For each (subject, trial) group, compute bidirectional fidelity between
    reference speed (fastest) and all slower speeds:

    H_q(ref) = argmin_{H>=0} ||X_q - W_ref H||_F^2
    Fidelity = 1 - ||X - W @ H||_F^2 / ||X||_F^2

    Solver: scipy.optimize.nnls (column-wise)
"""

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
    "ref_speed_mps",
    "q_speed_mps",
    "k_ref",
    "k_q",
    "n_strides_ref",
    "n_strides_q",
    "fidelity_ref_to_q",
    "fidelity_q_to_ref",
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
    """
    Solve for H >= 0 minimizing ||X - W @ H||_F^2 with W fixed.
    Uses scipy.optimize.nnls for each column of X independently.

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
    """
    fidelity = 1 - ||X - W @ H||_F^2 / ||X||_F^2
    """
    residual = X - (W @ H)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum(X ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def count_strides(X: np.ndarray, n_points: int) -> int:
    return int(X.shape[1] // n_points)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def pick_one_row(grp: pd.DataFrame, speed: float) -> pd.Series:
    # exact equality is OK here because speed_mps values are canonical (e.g., 5.0, 2.78)
    rows = grp[grp["speed_mps"] == speed]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly 1 row for speed={speed}, got {len(rows)}")
    return rows.iloc[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute bidirectional coordination fidelity via NNLS cross-reconstruction."
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

    out_csv = out_dir / "coordination_fidelity.csv"
    out_summary = out_dir / "fidelity_summary.txt"
    out_params = out_dir / "fidelity_params.txt"

    if not args.overwrite:
        for p in (out_csv, out_summary, out_params):
            if p.exists():
                print(f"ERROR: output already exists: {p}")
                print("       Use --overwrite or choose a fresh --out_dir.")
                return 1

    start_time = time.time()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("COORDINATION FIDELITY")
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

    required_cols = ["dataset", "subject", "trial", "speed_mps", "file_x", "file_npz", "pass_stability"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: stability_summary.csv missing required columns: {missing_cols}")
        return 1

    # Robust pass-stability filter
    df = df[df["pass_stability"].apply(truthy)].copy()
    n_pass = len(df)
    if n_pass == 0:
        print("ERROR: No rows with pass_stability == True after truthy parsing.")
        return 1

    df["speed_mps"] = df["speed_mps"].astype(float)
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
        if len(speeds) == 1:
            n_skipped_single_speed += 1
            continue

        ref_speed = max(speeds)
        slower_speeds = [s for s in speeds if s < ref_speed]

        ref_row = pick_one_row(grp, ref_speed)
        X_ref = load_X(Path(ref_row["file_x"]))
        W_ref = load_W(Path(ref_row["file_npz"]))
        k_ref = int(W_ref.shape[1])
        n_strides_ref = count_strides(X_ref, args.n_points)

        print(f"Processing {dataset}/{subject}/trial={trial:.1f} ref_speed={ref_speed} k_ref={k_ref} n_strides_ref={n_strides_ref}")

        for q_speed in slower_speeds:
            q_row = pick_one_row(grp, q_speed)
            X_q = load_X(Path(q_row["file_x"]))
            W_q = load_W(Path(q_row["file_npz"]))
            k_q = int(W_q.shape[1])
            n_strides_q = count_strides(X_q, args.n_points)

            # Required muscle-dimension harmonisation checks
            assert W_ref.shape[0] == X_q.shape[0], (
                f"Muscle dimension mismatch: W_ref rows={W_ref.shape[0]} != X_q rows={X_q.shape[0]} "
                f"for {dataset}/{subject}/trial={trial} q_speed={q_speed}"
            )
            assert W_q.shape[0] == X_ref.shape[0], (
                f"Muscle dimension mismatch: W_q rows={W_q.shape[0]} != X_ref rows={X_ref.shape[0]} "
                f"for {dataset}/{subject}/trial={trial} q_speed={q_speed}"
            )

            # ref -> q
            H_q_from_ref = solve_H_nnls(W_ref, X_q)
            fidelity_ref_to_q = compute_fidelity(X_q, W_ref, H_q_from_ref)

            # q -> ref
            H_ref_from_q = solve_H_nnls(W_q, X_ref)
            fidelity_q_to_ref = compute_fidelity(X_ref, W_q, H_ref_from_q)

            results.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "trial": float(trial),
                    "ref_speed_mps": float(ref_speed),
                    "q_speed_mps": float(q_speed),
                    "k_ref": int(k_ref),
                    "k_q": int(k_q),
                    "n_strides_ref": int(n_strides_ref),
                    "n_strides_q": int(n_strides_q),
                    "fidelity_ref_to_q": float(fidelity_ref_to_q),
                    "fidelity_q_to_ref": float(fidelity_q_to_ref),
                    "notes": "",
                }
            )

            n_pairs += 1
            print(
                f"  q_speed={q_speed} k_q={k_q} "
                f"fidelity_ref_to_q={fidelity_ref_to_q:.4f} fidelity_q_to_ref={fidelity_q_to_ref:.4f}"
            )

    df_out = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    # Deterministic ordering
    if len(df_out) > 0:
        df_out = df_out.sort_values(
            by=["dataset", "subject", "trial", "q_speed_mps"],
            kind="mergesort",
        ).reset_index(drop=True)

    elapsed = time.time() - start_time

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    # Stats
    if len(df_out) > 0:
        rmin = float(df_out["fidelity_ref_to_q"].min())
        rmax = float(df_out["fidelity_ref_to_q"].max())
        rmean = float(df_out["fidelity_ref_to_q"].mean())
        qmin = float(df_out["fidelity_q_to_ref"].min())
        qmax = float(df_out["fidelity_q_to_ref"].max())
        qmean = float(df_out["fidelity_q_to_ref"].mean())
    else:
        rmin = rmax = rmean = float("nan")
        qmin = qmax = qmean = float("nan")

    # Dataset counts
    ds_counts = df_out["dataset"].value_counts().to_dict() if len(df_out) > 0 else {}

    summary_lines = []
    summary_lines.append("Coordination Fidelity Summary")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    summary_lines.append(f"Run timestamp: {run_ts}")
    summary_lines.append(f"Upstream (input): {rel7_dir}")
    summary_lines.append(f"OUT_DIR: {out_dir}")
    summary_lines.append(f"N_POINTS: {args.n_points}")
    summary_lines.append("")
    summary_lines.append(f"Total groups processed: {n_groups_processed}")
    summary_lines.append(f"Groups skipped (single speed): {n_skipped_single_speed}")
    summary_lines.append(f"Total fidelity pairs computed: {n_pairs}")
    summary_lines.append(f"Output rows: {len(df_out)}")
    summary_lines.append(f"Runtime: {elapsed:.2f} seconds")
    summary_lines.append("")
    summary_lines.append("Fidelity statistics:")
    summary_lines.append(f"  fidelity_ref_to_q: min={rmin:.4f}, max={rmax:.4f}, mean={rmean:.4f}")
    summary_lines.append(f"  fidelity_q_to_ref: min={qmin:.4f}, max={qmax:.4f}, mean={qmean:.4f}")
    summary_lines.append("")
    summary_lines.append("By dataset:")
    if ds_counts:
        for k, v in sorted(ds_counts.items()):
            summary_lines.append(f"  {k}: {v} pairs")
    else:
        summary_lines.append("  (none)")
    summary_text = "\n".join(summary_lines) + "\n"
    write_text(out_summary, summary_text)

    params_lines = []
    params_lines.append("Coordination Fidelity Parameters")
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
    params_lines.append("Reference speed: max(speed_mps) per (dataset, subject, trial) group")
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
