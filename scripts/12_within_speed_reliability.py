#!/usr/bin/env python3
"""scripts/12_within_speed_reliability.py

Compute within-speed (half-split) coordination fidelity for each speed.

Outputs
-------
- outputs/within_speed_reliability_all.csv
  One row per (dataset, subject, trial, speed) with:
    within_fid_mean, within_fid_q10, n_boot_success

- outputs/within_speed_reliability_all_params.txt
- outputs/within_speed_reliability_all_summary.txt


References
----------
Uses scikit-learn's NMF coordinate descent solver and SciPy's NNLS.
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
from sklearn.decomposition import NMF


OUTPUT_COLUMNS = [
    "dataset",
    "subject",
    "trial",
    "speed_mps",
    "k",
    "n_strides",
    "n_points",
    "n_boot",
    "n_boot_success",
    "within_fid_mean",
    "within_fid_q10",
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
    data = np.load(str(npz_path), allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path}. Keys={list(data.keys())}")
    return data[key]


def load_X(file_x_path: Path) -> tuple[np.ndarray, int]:
    """Load X plus n_points if present."""
    if not file_x_path.exists():
        raise FileNotFoundError(f"Missing file: {file_x_path}")
    with np.load(str(file_x_path), allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=float)
        # enforce non-negativity (should already be non-negative)
        X[X < 0] = 0.0
        n_points = int(z.get("n_points", 200))
    return X, n_points


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


def compute_fidelity(X: np.ndarray, W: np.ndarray) -> float:
    """Compute fidelity for X reconstructed through fixed W using NNLS."""
    denom = float(np.sum(X**2))
    if denom == 0.0:
        return float("nan")
    H = solve_H_nnls(W, X)
    residual = X - (W @ H)
    ss_res = float(np.sum(residual**2))
    return 1.0 - (ss_res / denom)


def nmf_best_cd(
    X: np.ndarray,
    k: int,
    *,
    seed_base: int,
    replicate_index: int,
    n_init: int,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Fit NMF (solver='cd', init='random') with multiple starts.

    Returns the synergy matrix W of shape (m, k).
    """
    # sklearn expects (n_samples, n_features)
    X_t = X.T  # (time, muscles)

    best_err = np.inf
    best_W = None

    for s in range(n_init):
        rs = int(seed_base + 100 * int(replicate_index) + int(s))
        model = NMF(
            n_components=int(k),
            init="random",
            solver="cd",
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=rs,
            shuffle=False,
        )
        W_time = model.fit_transform(X_t)  # (time, k)
        H_muscle = model.components_  # (k, muscles)

        W = H_muscle.T  # (muscles, k)
        H = W_time.T  # (k, time)

        err = float(np.linalg.norm(X - (W @ H), ord="fro"))
        if err < best_err:
            best_err = err
            best_W = W

    if best_W is None:
        raise RuntimeError("All NMF initializations failed")

    return np.asarray(best_W, dtype=float)


def reshape_to_strides(X: np.ndarray, n_points: int) -> np.ndarray:
    """Reshape X (m, n_strides*n_points) -> (m, n_strides, n_points)."""
    m, n_obs = X.shape
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if n_obs % n_points != 0:
        raise ValueError(
            f"X has n_obs={n_obs} columns which is not divisible by n_points={n_points}."
        )
    n_strides = int(n_obs // n_points)
    return X.reshape(m, n_strides, n_points)


def half_split_indices(n_strides: int, replicate_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stride half-split using RandomState(seed=replicate_index)."""
    rng = np.random.RandomState(int(replicate_index))
    perm = rng.permutation(int(n_strides))
    n_use = (int(n_strides) // 2) * 2
    perm = perm[:n_use]
    half = n_use // 2
    return perm[:half], perm[half:]


def summarize_distribution(vals: np.ndarray) -> tuple[float, float, int]:
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(vals)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return float("nan"), float("nan"), 0
    v = vals[ok]
    mean = float(np.mean(v))
    q10 = float(np.quantile(v, 0.10))
    return mean, q10, n_ok


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute within-speed half-split fidelity (mean and 10th percentile) for each speed."
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
        "--dataset",
        default="vanhooren",
        help="Dataset to compute (default: vanhooren; use 'all' for all datasets)",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=100,
        help="Number of half-split replicates per condition (default: 100)",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=20,
        help="Random starts per NMF fit (default: 20)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Max iterations for NMF (default: 500)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for NMF (default: 1e-4)",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=20260103,
        help="Base seed for NMF random_state schedule (default: 20260103)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files in out_dir.",
    )
    parser.add_argument(
        "--allow_empty_output",
        action="store_true",
        help="Write output CSV even if no successful within-speed estimates are produced (all rows fail).",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    release_dir = script_path.parents[1]
    default_out_dir = release_dir / "outputs"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir

    rel7_dir = Path(args.rel7).resolve()
    stability_path = rel7_dir / "outputs" / "stability_summary.csv"
    if not stability_path.exists():
        print(f"ERROR: stability_summary.csv not found: {stability_path}")
        return 1

    out_csv = out_dir / "within_speed_reliability_all.csv"
    out_summary = out_dir / "within_speed_reliability_all_summary.txt"
    out_params = out_dir / "within_speed_reliability_all_params.txt"

    if not args.overwrite:
        for p in (out_csv, out_summary, out_params):
            if p.exists():
                print(f"ERROR: output already exists: {p}")
                print("       Use --overwrite or choose a fresh --out_dir.")
                return 1

    start_time = time.time()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("WITHIN-SPEED RELIABILITY ACROSS SPEEDS")
    print("=" * 60)
    print(f"Run timestamp: {run_ts}")
    print(f"Upstream: {rel7_dir}")
    print(f"OUT_DIR: {out_dir}")
    print(f"Dataset filter: {args.dataset}")
    print(f"n_boot={args.n_boot} n_init={args.n_init} max_iter={args.max_iter} tol={args.tol}")
    print(f"seed_base={args.seed_base}")
    print(f"python: {sys.version.split()[0]}")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"scipy: {scipy.__version__}")
    print("")

    df = pd.read_csv(stability_path)
    n_total = len(df)

    required_cols = ["dataset", "subject", "trial", "speed_mps", "k", "file_x", "pass_stability"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: stability_summary.csv missing required columns: {missing_cols}")
        return 1

    df = df[df["pass_stability"].apply(truthy)].copy()
    if args.dataset.strip().lower() != "all":
        df = df[df["dataset"].astype(str) == args.dataset].copy()

    n_pass = len(df)
    if n_pass == 0:
        print("ERROR: No rows left after pass_stability + dataset filtering.")
        return 1

    df["speed_mps"] = df["speed_mps"].astype(float)
    df["trial"] = df["trial"].astype(float)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype(int)

    df = df.sort_values(["dataset", "subject", "trial", "speed_mps"], kind="mergesort").reset_index(drop=True)

    print(f"Loaded stability_summary.csv: total_rows={n_total} filtered_rows={n_pass}")
    print("")

    rows = []

    for i in range(len(df)):
        r = df.iloc[i]
        dataset = str(r["dataset"])
        subject = str(r["subject"])
        trial = float(r["trial"])
        speed = float(r["speed_mps"])
        k = int(r["k"])
        file_x = Path(str(r["file_x"]))

        print(f"[{i+1}/{len(df)}] {dataset}/{subject}/trial={trial:.1f} speed={speed} k={k}")

        try:
            X, n_points_in_file = load_X(file_x)
        except Exception as e:
            print(f"  ERROR loading X: {type(e).__name__}: {e}")
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "trial": trial,
                    "speed_mps": speed,
                    "k": k,
                    "n_strides": np.nan,
                    "n_points": np.nan,
                    "n_boot": int(args.n_boot),
                    "n_boot_success": 0,
                    "within_fid_mean": np.nan,
                    "within_fid_q10": np.nan,
                    "notes": f"LOAD_FAIL_X: {type(e).__name__}",
                }
            )
            continue

        n_points = int(n_points_in_file)
        try:
            X3 = reshape_to_strides(X, n_points)
        except Exception as e:
            print(f"  ERROR reshaping X: {type(e).__name__}: {e}")
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "trial": trial,
                    "speed_mps": speed,
                    "k": k,
                    "n_strides": np.nan,
                    "n_points": int(n_points),
                    "n_boot": int(args.n_boot),
                    "n_boot_success": 0,
                    "within_fid_mean": np.nan,
                    "within_fid_q10": np.nan,
                    "notes": f"BAD_X_SHAPE: {type(e).__name__}",
                }
            )
            continue

        m, n_strides, _ = X3.shape

        if n_strides < 2:
            print(f"  WARNING: n_strides={n_strides} < 2; skipping")
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "trial": trial,
                    "speed_mps": speed,
                    "k": k,
                    "n_strides": int(n_strides),
                    "n_points": int(n_points),
                    "n_boot": int(args.n_boot),
                    "n_boot_success": 0,
                    "within_fid_mean": np.nan,
                    "within_fid_q10": np.nan,
                    "notes": "TOO_FEW_STRIDES",
                }
            )
            continue

        # Ensure k is feasible
        if k <= 0 or k > m:
            print(f"  WARNING: k={k} not in [1,{m}] for this X; skipping")
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "trial": trial,
                    "speed_mps": speed,
                    "k": k,
                    "n_strides": int(n_strides),
                    "n_points": int(n_points),
                    "n_boot": int(args.n_boot),
                    "n_boot_success": 0,
                    "within_fid_mean": np.nan,
                    "within_fid_q10": np.nan,
                    "notes": "K_OUT_OF_RANGE",
                }
            )
            continue

        fidelities = np.full((int(args.n_boot),), np.nan, dtype=float)

        for boot_i in range(int(args.n_boot)):
            idx_a, idx_b = half_split_indices(n_strides, boot_i)
            # If n_strides is odd, one stride is discarded.
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue

            X_a = X3[:, idx_a, :].reshape(m, len(idx_a) * n_points)
            X_b = X3[:, idx_b, :].reshape(m, len(idx_b) * n_points)

            try:
                W_a = nmf_best_cd(
                    X_a,
                    k,
                    seed_base=int(args.seed_base),
                    replicate_index=int(boot_i),
                    n_init=int(args.n_init),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                )
                W_b = nmf_best_cd(
                    X_b,
                    k,
                    seed_base=int(args.seed_base),
                    replicate_index=int(boot_i),
                    n_init=int(args.n_init),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                )

                fid_b_from_a = compute_fidelity(X_b, W_a)
                fid_a_from_b = compute_fidelity(X_a, W_b)
                fidelities[boot_i] = 0.5 * (fid_b_from_a + fid_a_from_b)
            except Exception:
                # Leave as NaN
                continue

            if (boot_i + 1) % 250 == 0:
                # Lightweight progress
                ok = int(np.isfinite(fidelities[: boot_i + 1]).sum())
                print(f"    boot {boot_i+1}/{args.n_boot} (finite={ok})")

        within_mean, within_q10, n_ok = summarize_distribution(fidelities)

        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "trial": trial,
                "speed_mps": speed,
                "k": int(k),
                "n_strides": int(n_strides),
                "n_points": int(n_points),
                "n_boot": int(args.n_boot),
                "n_boot_success": int(n_ok),
                "within_fid_mean": float(within_mean),
                "within_fid_q10": float(within_q10),
                "notes": "",
            }
        )

        print(f"  within_fid_mean={within_mean:.4f} within_fid_q10={within_q10:.4f} n_ok={n_ok}")

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Safety guard: avoid overwriting a good CSV with an all-fail run.
    n_success = int((out_df["n_boot_success"] > 0).sum())
    if n_success == 0 and not args.allow_empty_output:
        print("ERROR: No successful within-speed estimates were produced (all rows failed).")
        print(f"       Refusing to write: {out_csv}")
        print("       This usually means upstream processed NPZ files are missing or the --rel7 path is wrong.")
        print("       Fix the upstream data/path and rerun, or pass --allow_empty_output to write the failure table anyway.")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    elapsed = time.time() - start_time

    # Summary
    finite_rows = out_df[np.isfinite(out_df["within_fid_mean"])].copy()
    summary_lines = []
    summary_lines.append("Within-speed Reliability Across Speeds Summary")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    summary_lines.append(f"Run timestamp: {run_ts}")
    summary_lines.append(f"Upstream (input): {rel7_dir}")
    summary_lines.append(f"OUT_DIR: {out_dir}")
    summary_lines.append(f"Dataset filter: {args.dataset}")
    summary_lines.append("")
    summary_lines.append(f"Conditions processed: {len(out_df)}")
    summary_lines.append(f"Conditions with finite results: {len(finite_rows)}")
    summary_lines.append(f"Runtime: {elapsed:.2f} seconds")
    summary_lines.append("")
    if len(finite_rows) > 0:
        summary_lines.append(
            f"within_fid_mean: min={finite_rows['within_fid_mean'].min():.4f} "
            f"max={finite_rows['within_fid_mean'].max():.4f} "
            f"mean={finite_rows['within_fid_mean'].mean():.4f}"
        )
        summary_lines.append(
            f"within_fid_q10:  min={finite_rows['within_fid_q10'].min():.4f} "
            f"max={finite_rows['within_fid_q10'].max():.4f} "
            f"mean={finite_rows['within_fid_q10'].mean():.4f}"
        )
    else:
        summary_lines.append("No finite results.")
    summary_text = "\n".join(summary_lines) + "\n"
    write_text(out_summary, summary_text)

    # Params
    params_lines = []
    params_lines.append("Within-speed Reliability Across Speeds Parameters")
    params_lines.append("=" * 60)
    params_lines.append("")
    params_lines.append(f"Run timestamp: {run_ts}")
    params_lines.append(f"Script: {script_path}")
    params_lines.append(f"Release folder: {release_dir}")
    params_lines.append(f"Upstream: {rel7_dir}")
    params_lines.append(f"OUT_DIR: {out_dir}")
    params_lines.append(f"dataset filter: {args.dataset}")
    params_lines.append("")
    params_lines.append(f"n_boot: {args.n_boot}")
    params_lines.append(f"n_init (per fit): {args.n_init}")
    params_lines.append(f"NMF solver: cd")
    params_lines.append(f"NMF init: random")
    params_lines.append(f"NMF max_iter: {args.max_iter}")
    params_lines.append(f"NMF tol: {args.tol}")
    params_lines.append(f"NMF random_state schedule: seed_base + 100*r + s (seed_base={args.seed_base})")
    params_lines.append("half-split stride permutation: np.random.RandomState(seed=r).permutation(n_strides)")
    params_lines.append("Fidelity: 1 - ||X - W H||_F^2 / ||X||_F^2, with H via NNLS per column")
    params_lines.append("")
    params_lines.append(f"python: {sys.version.split()[0]}")
    params_lines.append(f"numpy: {np.__version__}")
    params_lines.append(f"pandas: {pd.__version__}")
    params_lines.append(f"scipy: {scipy.__version__}")
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
