# -*- coding: utf-8 -*-
""" 
Script: 02a_nmf_baseline.py
Purpose: Extract muscle synergies using NMF across a candidate rank range.

Inputs:
    - outputs/unified_registry.csv (from 01_preprocess.py)

Outputs:
    - outputs/nmf_baseline_summary.csv
    - outputs/nmf_baseline/*.npz (W, H matrices per subject/speed/k)

Algorithm:
    Non-negative Matrix Factorization with multiplicative update rules
    (Frobenius norm objective). For each input, extracts synergies for a
    candidate rank range, selecting the best of 10 random
    initializations per k.

Parameters:
    - k_min = 2
    - k_max(vanhooren) = min(7, m-1)
    - k_max(santuz) = min(9, m-1)
    - n_init = 10 random starts per k
    - max_iter = 2000 (primary extraction)
    - tol = 1e-5 (convergence tolerance)
"""

from __future__ import annotations

import os
import time
import warnings
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =========================
# PATH CONFIGURATION
# =========================
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))
DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[1] / "data"))

RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"

# =========================
# NMF PARAMETERS
# =========================
# Candidate rank range
K_MIN = 2
K_MAX_VANHOOREN = 7
K_MAX_SANTUZ = 9
K_MAX_DEFAULT = 10
N_INIT = 10
MAX_ITER = 2000  # Primary extraction uses 2000 iterations
TOL = 1e-5
EPS = 1e-8
SAVE_H = True
DTYPE = "float32"


def stable_seed(parts) -> int:
    """Generate a deterministic seed from input parts."""
    s = "|".join([str(x) for x in parts])
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def normalize_wh(W: np.ndarray, H: np.ndarray, eps: float) -> tuple:
    """Normalize W columns to sum to 1, scale H accordingly."""
    scale = W.sum(axis=0) + eps
    return W / scale, H * scale.reshape(-1, 1)


def vaf_stats(X: np.ndarray, Xhat: np.ndarray, eps: float) -> tuple:
    """Compute VAF statistics (total, min, median, mean per muscle)."""
    sse = float(np.square(X - Xhat).sum())
    sst = float(np.square(X).sum()) + eps
    vaf_total = 1.0 - (sse / sst)

    vaf_m = []
    for i in range(X.shape[0]):
        sse_i = float(np.square(X[i] - Xhat[i]).sum())
        sst_i = float(np.square(X[i]).sum()) + eps
        vaf_m.append(1.0 - (sse_i / sst_i))
    vaf_m = np.array(vaf_m, dtype=float)

    return vaf_total, float(np.min(vaf_m)), float(np.median(vaf_m)), float(np.mean(vaf_m))


def nmf_mu(
    X: np.ndarray, k: int, seed: int, max_iter: int, tol: float, eps: float
) -> tuple:
    """
    NMF via multiplicative updates (Frobenius norm).

    Returns: (W, H, sse, iterations, converged)
    """
    X = np.asarray(X, dtype=float)
    if np.any(X < -1e-12):
        raise ValueError("X has negative entries; NMF requires X >= 0")

    rng = np.random.default_rng(seed)
    m, n = X.shape
    W = rng.random((m, k), dtype=float) + eps
    H = rng.random((k, n), dtype=float) + eps

    prev = None
    for it in range(1, max_iter + 1):
        WH = W @ H
        H *= (W.T @ X) / ((W.T @ WH) + eps)

        WH = W @ H
        W *= (X @ H.T) / ((WH @ H.T) + eps)

        if (it % 10) == 0 or it == max_iter:
            Xhat = W @ H
            err = float(np.square(X - Xhat).sum())
            if prev is not None:
                rel = abs(prev - err) / (prev + eps)
                if rel < tol:
                    return W, H, err, it, True
            prev = err

    Xhat = W @ H
    err = float(np.square(X - Xhat).sum())
    return W, H, err, max_iter, False


def speed_tag(speed_mps: float) -> str:
    """Convert speed to 4-digit tag (e.g., 3.10 -> 0310)."""
    if not np.isfinite(speed_mps):
        return "nan"
    return str(int(round(float(speed_mps) * 100.0))).zfill(4)


def trial_tag(trial: str) -> str:
    """Normalize trial identifier."""
    t = str(trial).strip()
    if (not t) or (t.lower() == "nan"):
        return "00"
    return t


def _parse_muscles_used(x: Any) -> List[str]:
    """Parse semicolon-separated muscle list."""
    s = str(x) if x is not None else ""
    s = s.strip()
    if not s or s.lower() == "nan":
        return []
    return [m.strip() for m in s.split(";") if m.strip()]


def load_X_from_npz(file_path: Path, muscles_keep: List[str]) -> Dict[str, Any]:
    """Load preprocessed EMG data from npz file."""
    with np.load(file_path, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=float)
        muscles = [str(m) for m in z["muscles"].tolist()] if "muscles" in z.files else []
        
        n_points = int(z.get("n_points", 200))
        if n_points <= 0:
            n_points = 200

        n_strides = int(z.get("n_strides", 0))
        if n_strides <= 0:
            if X.ndim == 2 and (X.shape[1] % n_points == 0):
                n_strides = int(X.shape[1] // n_points)
            else:
                n_strides = 0

    
    if muscles_keep and muscles:
        keep_idx = [i for i, m in enumerate(muscles) if m in muscles_keep]
        if keep_idx:
            X = X[keep_idx, :]
            muscles = [muscles[i] for i in keep_idx]

    return {
        "X": X,
        "muscles": muscles,
        "n_strides": n_strides,
        "n_points": n_points,
    }


def main() -> int:
    print("=" * 70)
    print("NMF BASELINE EXTRACTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load registry
    registry_path = UPSTREAM_PATH / "unified_registry.csv"
    if not registry_path.exists():
        registry_path = OUTPUT_DIR / "unified_registry.csv"

    if not registry_path.exists():
        print(f"ERROR: Cannot find unified_registry.csv at {registry_path}")
        return 1

    print(f"Loading registry from {registry_path}")
    df = pd.read_csv(registry_path)

    # Filter to valid rows
    if "ok" in df.columns:
        df = df[df["ok"] == True].copy()

    # Output directory for npz files
    nmf_out_root = OUTPUT_DIR / "nmf_baseline"
    nmf_out_root.mkdir(parents=True, exist_ok=True)

    # Runtime controls
    limit = int(os.environ.get("NMF_LIMIT", "0") or 0)
    ds_filter = os.environ.get("NMF_DATASET", "all").strip().lower()
    overwrite = os.environ.get("NMF_OVERWRITE", "0").strip() == "1"

    if ds_filter in ["vanhooren", "santuz"]:
        df = df[df["dataset"].astype(str) == ds_filter].reset_index(drop=True)

    if limit > 0:
        df = df.head(limit).copy()

    total_inputs = len(df)
    if total_inputs == 0:
        print("ERROR: No inputs to process after filtering")
        return 1

    print(f"Processing {total_inputs} inputs")
    print(
        f"Parameters: k_min={K_MIN}, k_max(vanhooren)={K_MAX_VANHOOREN}, "
        f"k_max(santuz)={K_MAX_SANTUZ}, n_init={N_INIT}, max_iter={MAX_ITER}, tol={TOL}"
    )

    summary_rows: List[Dict[str, Any]] = []
    n_inputs_fail = 0
    t0 = time.time()

    for i in range(total_inputs):
        row = df.iloc[i]
        dataset = str(row.get("dataset", "")).strip()
        subject = str(row.get("subject", "")).strip()
        trial = trial_tag(row.get("trial", ""))
        sp = float(row.get("speed_mps", np.nan))

        ref_sp = float(row.get("reference_speed_mps", np.nan))
        is_ref = bool(row.get("is_reference", False))

        muscles_keep = _parse_muscles_used(row.get("muscles_used", ""))
        if not muscles_keep:
            muscles_keep = _parse_muscles_used(row.get("common_muscles_global", ""))

        if not muscles_keep:
            n_inputs_fail += 1
            summary_rows.append({
                "dataset": dataset, "subject": subject, "trial": trial,
                "speed_mps": sp, "k": np.nan, "ok": False,
                "fail_reason": "NO_MUSCLES_USED_IN_REGISTRY",
            })
            continue

        # Find input file
        fp = str(row.get("file_matched_resolved", "")).strip()
        if not fp:
            fp = str(row.get("file_matched", "")).strip()

        sp_tag = speed_tag(sp)
        out_dir = nmf_out_root / dataset / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        x_path = out_dir / f"X_{dataset}_{subject}_t{trial}_s{sp_tag}.npz"
        file_x = str(x_path)

        # Load X for this input
        try:
            input_path = Path(fp)
            if not input_path.exists():
                # Try relative to DATA_PATH
                input_path = DATA_PATH / "processed" / dataset / subject / Path(fp).name

            if not input_path.exists():
                raise FileNotFoundError(f"Cannot find input file: {fp}")

            d = load_X_from_npz(input_path, muscles_keep)
            X = np.asarray(d["X"], dtype=float)
            muscles = d["muscles"]
            n_strides = d["n_strides"]
            n_points = d["n_points"]

            if float(np.nanmin(X)) < 0.0:
                X = np.maximum(X, 0.0)

            if not np.isfinite(X).all():
                raise ValueError("BAD_X nonfinite")

            m, n_obs = X.shape
            if n_strides <= 0 or n_points <= 0:
                raise ValueError(f"BAD_STRIDE_META n_strides={n_strides} n_points={n_points}")

            # Save X once per input
            if not x_path.exists() or overwrite:
                np.savez_compressed(
                    x_path,
                    X=np.asarray(X, dtype=np.float32),
                    muscles=np.asarray(muscles, dtype=object),
                    n_strides=int(n_strides),
                    n_points=int(n_points),
                    dataset=str(dataset),
                    subject=str(subject),
                    trial=str(trial),
                    speed_mps=float(sp) if np.isfinite(sp) else np.nan,
                    reference_speed_mps=float(ref_sp) if np.isfinite(ref_sp) else np.nan,
                    is_reference=bool(is_ref),
                    muscles_used=";".join(muscles_keep),
                )

        except Exception as e:
            n_inputs_fail += 1
            summary_rows.append({
                "dataset": dataset, "subject": subject, "trial": trial,
                "speed_mps": sp, "k": np.nan, "ok": False,
                "fail_reason": f"LOAD_FAIL: {type(e).__name__}: {str(e)[:200]}",
            })
            continue

        # NMF for this input
        m, n = X.shape
        # Upper bound: k <= min(K_MAX_dataset, m-1)
        if str(dataset).lower() == "vanhooren":
            k_hi = min(K_MAX_VANHOOREN, m - 1)
        elif str(dataset).lower() == "santuz":
            k_hi = min(K_MAX_SANTUZ, m - 1)
        else:
            k_hi = min(K_MAX_DEFAULT, m - 1)
        if k_hi < K_MIN:
            n_inputs_fail += 1
            summary_rows.append({
                "dataset": dataset, "subject": subject, "trial": trial,
                "speed_mps": sp, "k": np.nan, "ok": False,
                "fail_reason": f"K_RANGE_EMPTY: m={m}",
                "file_x": file_x, "muscles": ";".join(muscles),
                "n_muscles": int(m), "n_obs": int(n),
                "n_strides": int(n_strides), "n_points": int(n_points),
                "reference_speed_mps": ref_sp, "is_reference": bool(is_ref),
            })
            continue

        for k in range(K_MIN, k_hi + 1):
            fname = f"nmf_{dataset}_{subject}_t{trial}_s{sp_tag}_k{str(k).zfill(2)}.npz"
            out_path = out_dir / fname

            # Skip if exists and not overwriting
            if out_path.exists() and not overwrite:
                try:
                    with np.load(out_path, allow_pickle=True) as z:
                        z_mus = [str(x) for x in z["muscles"].tolist()] if "muscles" in z.files else []
                        if z_mus == muscles:
                            summary_rows.append({
                                "dataset": dataset, "subject": subject, "trial": trial,
                                "speed_mps": float(z.get("speed_mps", sp)),
                                "k": int(z.get("k", k)),
                                "n_muscles": int(z.get("n_muscles", m)),
                                "n_obs": int(z.get("n_obs", n)),
                                "n_strides": int(z.get("n_strides", n_strides)),
                                "n_points": int(z.get("n_points", n_points)),
                                "n_init": int(z.get("n_init", N_INIT)),
                                "max_iter": int(z.get("max_iter", MAX_ITER)),
                                "tol": float(z.get("tol", TOL)),
                                "eps": float(z.get("eps", EPS)),
                                "seed": int(z.get("seed", 0)),
                                "iters": int(z.get("iters", 0)),
                                "converged": bool(z.get("converged", False)),
                                "vaf_total": float(z.get("vaf_total", np.nan)),
                                "vaf_min": float(z.get("vaf_min", np.nan)),
                                "vaf_median": float(z.get("vaf_median", np.nan)),
                                "vaf_mean": float(z.get("vaf_mean", np.nan)),
                                "sse": float(z.get("sse", np.nan)),
                                "file_npz": str(out_path),
                                "file_x": file_x,
                                "muscles": ";".join(muscles),
                                "muscles_used": ";".join(muscles_keep),
                                "reference_speed_mps": float(z.get("reference_speed_mps", ref_sp)),
                                "is_reference": bool(z.get("is_reference", is_ref)),
                                "ok": True, "fail_reason": "", "skipped_existing": True,
                            })
                            continue
                except Exception:
                    pass

            # Run NMF with multiple random starts
            best = None
            best_W = None
            best_H = None

            for j in range(N_INIT):
                seed = stable_seed([dataset, subject, trial, sp_tag, k, j])
                W, H, sse, iters, converged = nmf_mu(X, k, seed, MAX_ITER, TOL, EPS)
                W, H = normalize_wh(W, H, EPS)
                Xhat = W @ H
                vaf_total, vaf_min, vaf_median, vaf_mean = vaf_stats(X, Xhat, EPS)

                cand = {
                    "seed": int(seed), "iters": int(iters), "converged": bool(converged),
                    "sse": float(sse), "vaf_total": float(vaf_total),
                    "vaf_min": float(vaf_min), "vaf_median": float(vaf_median),
                    "vaf_mean": float(vaf_mean),
                }

                if (best is None) or (cand["sse"] < best["sse"]):
                    best = cand
                    best_W = W
                    best_H = H

            assert best is not None and best_W is not None and best_H is not None

            # Save results
            W_save = best_W.astype(np.float32) if DTYPE == "float32" else best_W.astype(np.float64)
            H_save = best_H.astype(np.float32) if DTYPE == "float32" else best_H.astype(np.float64)

            np.savez_compressed(
                out_path,
                W=W_save,
                H=H_save if SAVE_H else np.empty((0, 0), dtype=np.float32),
                muscles=np.asarray(muscles, dtype=object),
                dataset=dataset, subject=subject, trial=trial,
                speed_mps=float(sp) if np.isfinite(sp) else np.nan,
                reference_speed_mps=float(ref_sp) if np.isfinite(ref_sp) else np.nan,
                is_reference=bool(is_ref),
                k=int(k), n_muscles=int(m), n_obs=int(n),
                n_strides=int(n_strides), n_points=int(n_points),
                n_init=int(N_INIT), max_iter=int(MAX_ITER),
                tol=float(TOL), eps=float(EPS),
                seed=int(best["seed"]), iters=int(best["iters"]),
                converged=bool(best["converged"]),
                vaf_total=float(best["vaf_total"]),
                vaf_min=float(best["vaf_min"]),
                vaf_median=float(best["vaf_median"]),
                vaf_mean=float(best["vaf_mean"]),
                sse=float(best["sse"]),
                file_x=str(x_path),
                muscles_used=";".join(muscles_keep),
            )

            summary_rows.append({
                "dataset": dataset, "subject": subject, "trial": trial,
                "speed_mps": sp, "reference_speed_mps": ref_sp,
                "is_reference": bool(is_ref), "k": int(k),
                "n_muscles": int(m), "n_obs": int(n),
                "n_strides": int(n_strides), "n_points": int(n_points),
                "n_init": int(N_INIT), "max_iter": int(MAX_ITER),
                "tol": float(TOL), "eps": float(EPS),
                "seed": int(best["seed"]), "iters": int(best["iters"]),
                "converged": bool(best["converged"]),
                "vaf_total": float(best["vaf_total"]),
                "vaf_min": float(best["vaf_min"]),
                "vaf_median": float(best["vaf_median"]),
                "vaf_mean": float(best["vaf_mean"]),
                "sse": float(best["sse"]),
                "file_npz": str(out_path),
                "file_x": file_x,
                "muscles": ";".join(muscles),
                "muscles_used": ";".join(muscles_keep),
                "ok": True, "fail_reason": "", "skipped_existing": False,
            })

        if ((i + 1) % 10) == 0 or (i + 1) == total_inputs:
            print(f"  Progress: {i + 1}/{total_inputs}")

    # Save summary
    out_csv = OUTPUT_DIR / "nmf_baseline_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    secs = float(time.time() - t0)
    print(f"\nWrote: {out_csv}")
    print(f"NMF outputs: {nmf_out_root}")
    print(f"Total inputs: {total_inputs}")
    print(f"Failed inputs: {n_inputs_fail}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Time: {secs:.1f} seconds")

    print("=" * 70)
    print("NMF BASELINE EXTRACTION COMPLETE")
    print("=" * 70)

    return 0 if n_inputs_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
