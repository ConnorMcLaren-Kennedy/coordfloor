# -*- coding: utf-8 -*-
"""
Script: 02c_nmf_stability.py
Purpose: Assess stability of chosen synergy rank via bootstrap resampling.

Inputs:
    - outputs/nmf_baseline_summary.csv (from 02a_nmf_baseline.py)
    - outputs/chosen_k.csv (from 02b_nmf_rankselect.py)

Outputs:
    - outputs/stability_summary.csv
    - outputs/final_synergies.csv

Stability criterion:
    - Bootstrap resampling over strides (n_boot=50)
    - Consensus matrix built from muscle co-clustering
    - Pass: cophenetic correlation >= 0.90 (for k>=2)
    - Pass: W cosine similarity >= 0.90 (for k=1)
"""

from __future__ import annotations

import os
import time
import warnings
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =========================
# PATH CONFIGURATION
# =========================
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))

RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"

# =========================
# STABILITY PARAMETERS 
# =========================
N_BOOT = 50              # Bootstrap replicates
N_INIT = 5               # Random starts per bootstrap NMF
MAX_ITER = 500           # Max iterations for bootstrap re-fits (not primary extraction)
TOL = 1e-4
EPS = 1e-9
SEED_BASE = 12345
MIN_COPHENETIC = 0.90    # Stability threshold
MIN_W_COSINE_K1 = 0.90   # Threshold for k=1 case

KEY_COLS: List[str] = ["dataset", "subject", "trial", "speed_mps"]


def compute_block_length(n_strides: int, min_b: int = 3, max_b_frac: int = 8) -> int:
    """Block length selection bounded to [min_b, n//max_b_frac], with sqrt(n) heuristic."""
    n_strides = int(n_strides)
    if n_strides <= 0:
        return 1
    max_b = max(min_b, n_strides // max_b_frac)
    b = max(min_b, min(max_b, int(np.sqrt(n_strides))))
    # ensure 1 <= b <= n_strides
    return int(max(1, min(b, n_strides)))


def blocked_bootstrap_indices(rng: np.random.Generator, n_strides: int, block_length: int) -> np.ndarray:
    """Non-circular moving-block bootstrap resampling of stride blocks (with replacement)."""
    n_strides = int(n_strides)
    b = int(block_length)
    if n_strides <= 0:
        return np.array([], dtype=int)

    # Fallback to iid stride sampling if blocks are ill-defined
    if b <= 1 or b >= n_strides:
        return rng.integers(0, n_strides, size=n_strides, endpoint=False)

    n_blocks_needed = (n_strides // b) + 1
    starts = rng.integers(0, n_strides - b + 1, size=n_blocks_needed, endpoint=False)

    idx: List[int] = []
    for s in starts.tolist():
        idx.extend(range(int(s), int(s) + b))
        if len(idx) >= n_strides:
            break

    return np.asarray(idx[:n_strides], dtype=int)


def stable_seed(parts: List[Any]) -> int:
    """Generate deterministic seed from parts."""
    s = "|".join([str(x) for x in parts])
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def speed_tag(speed_mps: float) -> str:
    """Convert speed to 4-digit tag."""
    if not np.isfinite(speed_mps):
        return "nan"
    return str(int(round(float(speed_mps) * 100.0))).zfill(4)


def _canon_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize key columns."""
    df = df.copy()
    for c in ("dataset", "subject", "trial"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "speed_mps" in df.columns:
        df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce").astype(float).round(6)
    return df


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_wh(W: np.ndarray, H: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize W columns to sum to 1."""
    scale = W.sum(axis=0) + eps
    return W / scale, H * scale.reshape(-1, 1)


def nmf_mu(
    X: np.ndarray, k: int, seed: int, max_iter: int, tol: float, eps: float
) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    """NMF via multiplicative updates."""
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


def _cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb + eps))


def _match_cosines(W_ref: np.ndarray, W_b: np.ndarray, eps: float) -> List[float]:
    """Match synergies via Hungarian algorithm and return cosine similarities."""
    k = W_ref.shape[1]
    S = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            S[i, j] = _cosine(W_ref[:, i], W_b[:, j], eps)

    S2 = np.where(np.isfinite(S), S, -1.0)

    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(1.0 - S2)
        return [float(S[i, j]) for i, j in zip(r.tolist(), c.tolist())]
    except Exception:
        # Greedy fallback
        sims: List[float] = []
        used_j = set()
        for i in range(k):
            j_best = None
            v_best = -1e9
            for j in range(k):
                if j in used_j:
                    continue
                v = S2[i, j]
                if v > v_best:
                    v_best = v
                    j_best = j
            if j_best is None:
                continue
            used_j.add(j_best)
            sims.append(float(S[i, j_best]))
        return sims


def _dispersion_from_consensus(consensus: np.ndarray) -> float:
    """Compute dispersion from consensus matrix."""
    C = np.asarray(consensus, dtype=float)
    m = int(C.shape[0])
    if m < 2:
        return float("nan")
    tri = C[np.triu_indices(m, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return float("nan")
    return float(np.mean(4.0 * np.square(tri - 0.5)))


def _cophenetic_from_consensus(consensus: np.ndarray, dispersion: float) -> float:
    """Compute cophenetic correlation from consensus matrix."""
    try:
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, cophenet

        C = np.asarray(consensus, dtype=float)
        m = int(C.shape[0])
        if m < 3:
            return float("nan")

        D = 1.0 - C
        np.fill_diagonal(D, 0.0)
        dvec = squareform(D, checks=False)
        dvec = np.asarray(dvec, dtype=float)

        if not np.isfinite(dvec).all():
            return float("nan")

        if float(np.std(dvec)) < 1e-12:
            if np.isfinite(dispersion) and dispersion > 0.95:
                return 1.0
            if np.isfinite(dispersion) and dispersion < 0.05:
                return 0.0
            return float("nan")

        Z = linkage(dvec, method="average")
        c, _ = cophenet(Z, dvec)
        return float(c)
    except Exception:
        return float("nan")


def main() -> int:
    print("=" * 70)
    print("NMF STABILITY GATING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find input files
    base_csv = UPSTREAM_PATH / "nmf_baseline_summary.csv"
    if not base_csv.exists():
        base_csv = OUTPUT_DIR / "nmf_baseline_summary.csv"

    chosen_csv = UPSTREAM_PATH / "chosen_k.csv"
    if not chosen_csv.exists():
        chosen_csv = OUTPUT_DIR / "chosen_k.csv"

    if not base_csv.exists():
        print(f"ERROR: Cannot find nmf_baseline_summary.csv")
        return 1
    if not chosen_csv.exists():
        print(f"ERROR: Cannot find chosen_k.csv")
        return 1

    print(f"Baseline summary: {base_csv}")
    print(f"Chosen k: {chosen_csv}")
    print(f"Parameters: n_boot={N_BOOT}, min_cophenetic={MIN_COPHENETIC}")

    base = _canon_keys(pd.read_csv(base_csv))
    chosen = _canon_keys(pd.read_csv(chosen_csv))

    # Find file path columns
    base_file_npz_col = _pick_col(base, ["file_npz", "npz_path", "path_npz"])
    base_file_x_col = _pick_col(base, ["file_x", "x_path", "file_X"])

    if base_file_npz_col is None or base_file_x_col is None:
        print("ERROR: Baseline summary missing file_npz or file_x columns")
        return 1

    chosen_k_col = _pick_col(chosen, ["k", "chosen_k", "k_chosen", "rank", "k_star"])
    if chosen_k_col is None:
        print("ERROR: chosen_k.csv missing k column")
        return 1

    if "k" not in base.columns:
        print("ERROR: Baseline summary missing k column")
        return 1

    base["k"] = pd.to_numeric(base["k"], errors="coerce").astype("Int64")
    base = base.drop_duplicates(subset=KEY_COLS + ["k"]).reset_index(drop=True)

    chosen = chosen.copy()
    chosen["k_chosen"] = pd.to_numeric(chosen[chosen_k_col], errors="coerce").astype("Int64")

    # Merge to get file paths for chosen k
    merged = chosen[KEY_COLS + ["k_chosen"]].merge(
        base,
        how="left",
        left_on=KEY_COLS + ["k_chosen"],
        right_on=KEY_COLS + ["k"],
        suffixes=("", "_base"),
    )

    total = len(merged)
    if total == 0:
        print("ERROR: No rows to process")
        return 1

    print(f"Processing {total} inputs")

    stability_rows: List[Dict[str, Any]] = []
    synergy_rows: List[Dict[str, Any]] = []
    pass_n = 0
    fail_n = 0
    t0 = time.time()

    for idx in range(total):
        r = merged.iloc[idx]
        dataset = str(r.get("dataset", "")).strip()
        subject = str(r.get("subject", "")).strip()
        trial = str(r.get("trial", "")).strip()
        sp = float(r.get("speed_mps", np.nan))
        k = int(r.get("k_chosen")) if pd.notna(r.get("k_chosen")) else -1

        rec: Dict[str, Any] = {
            "dataset": dataset, "subject": subject, "trial": trial,
            "speed_mps": sp, "k": k, "pass_stability": False,
            "fail_reason": "", "cophenetic": float("nan"),
            "dispersion": float("nan"), "w_cosine_median": float("nan"),
            "w_cosine_min": float("nan"), "n_strides": float("nan"),
            "n_points": float("nan"), "n_muscles": float("nan"),
            "muscles": "", "file_npz": "", "file_x": "",
        }

        # Load baseline W
        fp_npz = Path(str(r.get(base_file_npz_col, "")))
        if not fp_npz.exists():
            rec["fail_reason"] = "missing_baseline_npz_for_chosen_k"
            stability_rows.append(rec)
            fail_n += 1
            continue
        rec["file_npz"] = str(fp_npz)

        try:
            with np.load(fp_npz, allow_pickle=True) as z:
                W_ref = np.asarray(z["W"], dtype=float)
                muscles = [str(x) for x in z["muscles"].tolist()] if "muscles" in z.files else []
                k_ref = int(z.get("k", k))
                if k_ref != k:
                    k = k_ref
                    rec["k"] = k
        except Exception as e:
            rec["fail_reason"] = f"LOAD_FAIL baseline_npz: {type(e).__name__}: {str(e)[:160]}"
            stability_rows.append(rec)
            fail_n += 1
            continue

        if not muscles or W_ref.ndim != 2:
            rec["fail_reason"] = "baseline_npz_missing_W_or_muscles"
            stability_rows.append(rec)
            fail_n += 1
            continue

        m = int(W_ref.shape[0])
        if k <= 0 or k > int(W_ref.shape[1]):
            rec["fail_reason"] = f"k_out_of_range (k={k}, W_cols={W_ref.shape[1]})"
            rec["muscles"] = ";".join(muscles)
            rec["n_muscles"] = m
            stability_rows.append(rec)
            fail_n += 1
            continue

        rec["muscles"] = ";".join(muscles)
        rec["n_muscles"] = m

        # Load X for bootstrap
        fp_x = Path(str(r.get(base_file_x_col, "")))
        if not fp_x.exists():
            rec["fail_reason"] = "missing_file_x_for_bootstrap"
            stability_rows.append(rec)
            fail_n += 1
            continue
        rec["file_x"] = str(fp_x)

        try:
            with np.load(fp_x, allow_pickle=True) as zx:
                X = np.asarray(zx["X"], dtype=float)
                # Some upstream X files (older runs) may not include stride metadata.
                # Infer it from X if missing.
                n_points = int(zx.get("n_points", 200))
                if n_points <= 0:
                    n_points = 200

                n_strides = int(zx.get("n_strides", 0))
                if n_strides <= 0:
                    if X.ndim == 2 and (X.shape[1] % n_points == 0):
                        n_strides = int(X.shape[1] // n_points)
                    else:
                        n_strides = 0
        except Exception as e:
            rec["fail_reason"] = f"LOAD_FAIL X_npz: {type(e).__name__}: {str(e)[:160]}"
            stability_rows.append(rec)
            fail_n += 1
            continue

        if X.ndim != 2 or X.shape[0] != m:
            rec["fail_reason"] = f"BAD_X_SHAPE {X.shape} expected ({m}, n_obs)"
            stability_rows.append(rec)
            fail_n += 1
            continue

        if n_strides < 2 or n_points < 1:
            rec["fail_reason"] = f"n_strides<2 n_strides={n_strides} n_points={n_points}"
            rec["n_strides"] = n_strides
            rec["n_points"] = n_points
            stability_rows.append(rec)
            fail_n += 1
            continue

        n_obs = int(X.shape[1])
        if n_obs != (n_strides * n_points):
            rec["fail_reason"] = f"BAD_STRIDE_META n_obs={n_obs} != n_strides*n_points"
            rec["n_strides"] = n_strides
            rec["n_points"] = n_points
            stability_rows.append(rec)
            fail_n += 1
            continue

        rec["n_strides"] = n_strides
        rec["n_points"] = n_points

        # Bootstrap stability assessment
        X3 = X.reshape(m, n_strides, n_points)
        rng = np.random.default_rng(SEED_BASE + stable_seed([dataset, subject, trial, speed_tag(sp), k]))

        # Moving-block bootstrap over strides 
        b_len = compute_block_length(n_strides)

        cos_vals: List[float] = []
        conn_sum = np.zeros((m, m), dtype=float)

        for b in range(N_BOOT):
            idx_boot = blocked_bootstrap_indices(rng, n_strides, b_len)
            Xb = X3[:, idx_boot, :].reshape(m, n_strides * n_points)

            best_sse = None
            best_W = None
            best_H = None

            for j in range(N_INIT):
                seed = SEED_BASE + stable_seed([dataset, subject, trial, speed_tag(sp), k, "boot", b, j])
                W, H, sse, _, _ = nmf_mu(Xb, k, seed, MAX_ITER, TOL, EPS)
                W, H = normalize_wh(W, H, EPS)
                if (best_sse is None) or (float(sse) < float(best_sse)):
                    best_sse = float(sse)
                    best_W = W
                    best_H = H

            assert best_W is not None and best_H is not None

            sims = _match_cosines(W_ref[:, :k], best_W[:, :k], EPS)
            for v in sims:
                if np.isfinite(v):
                    cos_vals.append(float(v))

            labels = np.argmax(best_W[:, :k], axis=1)
            conn_sum += (labels[:, None] == labels[None, :]).astype(float)

        consensus = conn_sum / float(N_BOOT)
        np.fill_diagonal(consensus, 1.0)

        disp = _dispersion_from_consensus(consensus)
        coph = _cophenetic_from_consensus(consensus, disp)

        rec["dispersion"] = float(disp)
        rec["cophenetic"] = float(coph)

        if len(cos_vals) > 0:
            vv = np.asarray(cos_vals, dtype=float)
            rec["w_cosine_median"] = float(np.median(vv))
            rec["w_cosine_min"] = float(np.min(vv))

        # Apply stability criterion
        if k == 1:
            if np.isfinite(rec["w_cosine_median"]) and (rec["w_cosine_median"] >= MIN_W_COSINE_K1):
                rec["pass_stability"] = True
            else:
                rec["fail_reason"] = f"w_cosine_median<{MIN_W_COSINE_K1} (k==1) or NaN"
        else:
            if np.isfinite(coph) and (coph >= MIN_COPHENETIC):
                rec["pass_stability"] = True
            else:
                rec["fail_reason"] = f"cophenetic<{MIN_COPHENETIC} (or NaN)"

        stability_rows.append(rec)

        if not bool(rec["pass_stability"]):
            fail_n += 1
        else:
            pass_n += 1

        # Build synergy rows
        for s in range(k):
            sr: Dict[str, Any] = {
                "dataset": dataset, "subject": subject, "trial": trial,
                "speed_mps": sp, "k": k, "synergy": int(s + 1),
                "pass_stability": bool(rec["pass_stability"]),
                "fail_reason": str(rec["fail_reason"]),
                "cophenetic": float(rec["cophenetic"]),
                "dispersion": float(rec["dispersion"]),
                "w_cosine_median": float(rec["w_cosine_median"]),
                "w_cosine_min": float(rec["w_cosine_min"]),
                "muscles": ";".join(muscles),
                "n_muscles": int(m),
            }
            for mi, mn in enumerate(muscles):
                sr[mn] = float(W_ref[mi, s])
            synergy_rows.append(sr)

        if ((idx + 1) % 25) == 0 or ((idx + 1) == total):
            print(f"  Progress: {idx+1}/{total}, PASS={pass_n}")

    # Save outputs
    stab_df = pd.DataFrame(stability_rows)
    stab_path = OUTPUT_DIR / "stability_summary.csv"
    stab_df.to_csv(stab_path, index=False)

    syn_df = pd.DataFrame(synergy_rows)
    syn_path = OUTPUT_DIR / "final_synergies.csv"
    syn_df.to_csv(syn_path, index=False)

    secs = float(time.time() - t0)
    print(f"\nWrote: {stab_path}")
    print(f"Wrote: {syn_path}")
    print(f"Total: {total}, Pass: {pass_n}, Fail: {fail_n}")
    print(f"Time: {secs:.1f} seconds")

    print("=" * 70)
    print("NMF STABILITY GATING COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
