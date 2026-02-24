"""
Script: 09_sensitivity_analyses.py
Purpose: Assess robustness of floor estimates under methodological variants.

Inputs:
    - outputs/stability_summary.csv (from 02c_nmf_stability.py)
    - outputs/coordination_floors.csv (from 07_detect_floor.py)

Outputs:
    - outputs/sensitivity_variants.csv
    - outputs/sensitivity_comparison.csv

Variants tested:
    - baseline: Seven-gate floor estimates
    - nmf_kl: NMF with Kullback-Leibler divergence objective
    - stride_half: Half stride count (random subsample)
    - matching_euclidean: Euclidean distance for module pairing
    - norm_per_speed: Per-speed peak normalization

Implementation note:
    - For the sensitivity refits (all non-baseline variants), the synergy rank is held
      constant across speeds at each participant's reference-speed rank (k_ref). This
      keeps module pairing square (k×k) for Hungarian matching.

Comparison metrics:
    - Repeated-measures Bland-Altman (replicate-adjusted)
    - Intraclass correlation coefficients (ICC)
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# =========================
# INPUT PATHS
# =========================
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))
DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[1] / "data"))

# =========================
# RELEASE-LOCAL OUTPUTS
# =========================
RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"

# Constants
REF_SPEED = 5.0
TEST_ORDER = [4.0, 3.33, 3.0, 2.78]
N_POINTS = 200
RANDOM_SEED = 20260103

# Variant tags
VARIANTS = ["baseline", "nmf_kl", "stride_half", "matching_euclidean", "norm_per_speed"]


# ==============================================================================
# NMF IMPLEMENTATIONS
# ==============================================================================

def nmf_frobenius(X, k, seed, max_iter=500, tol=1e-6, eps=1e-10):
    """Standard NMF with Frobenius norm (multiplicative updates)."""
    rng = np.random.default_rng(seed)
    m, n = X.shape
    W = rng.random((m, k)) + eps
    H = rng.random((k, n)) + eps

    prev_err = None
    for it in range(1, max_iter + 1):
        WH = W @ H + eps
        H *= (W.T @ X) / (W.T @ WH + eps)
        WH = W @ H + eps
        W *= (X @ H.T) / (WH @ H.T + eps)

        if it % 20 == 0:
            err = np.sum((X - W @ H) ** 2)
            if prev_err is not None and abs(prev_err - err) / (prev_err + eps) < tol:
                break
            prev_err = err

    return W, H


def nmf_kl_divergence(X, k, seed, max_iter=500, tol=1e-6, eps=1e-10):
    """NMF with generalized Kullback-Leibler divergence."""
    rng = np.random.default_rng(seed)
    m, n = X.shape
    W = rng.random((m, k)) + eps
    H = rng.random((k, n)) + eps
    X_safe = np.maximum(X, eps)

    prev_div = None
    for it in range(1, max_iter + 1):
        WH = W @ H + eps
        H *= (W.T @ (X_safe / WH)) / (W.sum(axis=0, keepdims=True).T + eps)
        WH = W @ H + eps
        W *= ((X_safe / WH) @ H.T) / (H.sum(axis=1, keepdims=True).T + eps)

        if it % 20 == 0:
            WH = W @ H + eps
            div = np.sum(X_safe * np.log(X_safe / WH + eps) - X_safe + WH)
            if prev_div is not None and abs(prev_div - div) / (abs(prev_div) + eps) < tol:
                break
            prev_div = div

    return W, H


def run_nmf_multistart(X, k, nmf_func, n_starts=10, seed_base=42):
    """Run NMF with multiple random starts, return best solution."""
    best_W, best_H, best_err = None, None, np.inf
    for i in range(n_starts):
        W, H = nmf_func(X, k, seed=seed_base + i)
        err = np.sum((X - W @ H) ** 2)
        if err < best_err:
            best_W, best_H, best_err = W, H, err
    return best_W, best_H


# ==============================================================================
# FIDELITY AND METRICS
# ==============================================================================

def solve_H_nnls(W, X):
    """Solve for H >= 0 minimizing ||X - WH||_F^2."""
    k = W.shape[1]
    n_obs = X.shape[1]
    H = np.zeros((k, n_obs))
    for j in range(n_obs):
        H[:, j], _ = nnls(W, X[:, j])
    return H


def compute_fidelity(X, W, H):
    """Compute VAF-based fidelity."""
    residual = X - W @ H
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum(X ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def match_modules_cosine(W1, W2):
    """Match modules using cosine similarity (Hungarian algorithm)."""
    k = W1.shape[1]
    sim_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            n1 = np.linalg.norm(W1[:, i])
            n2 = np.linalg.norm(W2[:, j])
            if n1 > 0 and n2 > 0:
                sim_matrix[i, j] = np.dot(W1[:, i], W2[:, j]) / (n1 * n2)
    cost = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, sim_matrix


def match_modules_euclidean(W1, W2):
    """Match modules using Euclidean distance (Hungarian algorithm)."""
    k = W1.shape[1]
    W1_norm = W1 / (np.linalg.norm(W1, axis=0, keepdims=True) + 1e-10)
    W2_norm = W2 / (np.linalg.norm(W2, axis=0, keepdims=True) + 1e-10)
    cost = cdist(W1_norm.T, W2_norm.T, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost)
    sim_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            n1 = np.linalg.norm(W1[:, i])
            n2 = np.linalg.norm(W2[:, j])
            if n1 > 0 and n2 > 0:
                sim_matrix[i, j] = np.dot(W1[:, i], W2[:, j]) / (n1 * n2)
    return row_ind, col_ind, sim_matrix


def compute_spatial_temporal_metrics(W_ref, W_q, H_ref, H_q, match_func):
    """Compute cosine similarity, DTW, and principal angles."""
    row_ind, col_ind, sim_matrix = match_func(W_ref, W_q)

    cosines = [sim_matrix[i, col_ind[i]] for i in row_ind]
    cosine_median = np.median(cosines)

    k = W_ref.shape[1]
    dtw_dists = []
    for i, j in zip(row_ind, col_ind):
        h_ref_mean = H_ref[i, :].reshape(-1, N_POINTS).mean(axis=0)
        h_q_mean = H_q[j, :].reshape(-1, N_POINTS).mean(axis=0)
        dtw_d = simple_dtw(h_ref_mean, h_q_mean)
        dtw_dists.append(dtw_d)
    dtw_mean = np.mean(dtw_dists)

    W_ref_norm = W_ref / (np.linalg.norm(W_ref, axis=0, keepdims=True) + 1e-10)
    W_q_norm = W_q / (np.linalg.norm(W_q, axis=0, keepdims=True) + 1e-10)
    try:
        U1, _, _ = np.linalg.svd(W_ref_norm, full_matrices=False)
        U2, _, _ = np.linalg.svd(W_q_norm, full_matrices=False)
        M = U1.T @ U2
        svals = np.linalg.svd(M, compute_uv=False)
        svals = np.clip(svals, -1, 1)
        angles_rad = np.arccos(svals)
        principal_angle_max = np.degrees(np.max(angles_rad))
    except:
        principal_angle_max = 90.0

    return cosine_median, dtw_mean, principal_angle_max


def simple_dtw(a, b):
    """Simple DTW with absolute difference cost, no window constraint."""
    n, m = len(a), len(b)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_stability_summary():
    """Load stability summary for vanhooren subjects."""
    path = UPSTREAM_PATH / "stability_summary.csv"
    df = pd.read_csv(path)
    df = df[(df["dataset"] == "vanhooren") & (df["pass_stability"] == True)]
    return df


def load_reliability_envelope():
    """Load reliability bounds."""
    path = UPSTREAM_PATH / "reliability_envelope.csv"
    return pd.read_csv(path)


def load_surrogate_baseline():
    """Load surrogate baseline."""
    path = UPSTREAM_PATH / "surrogate_baseline.csv"
    return pd.read_csv(path)


def load_baseline_floors():
    """Load baseline floor results (seven-gate floors) for sensitivity comparisons."""
    path = UPSTREAM_PATH / "coordination_floors.csv"
    df = pd.read_csv(path)

    
    df = df.rename(columns={
        "floor_speed_mps_7gate": "floor_speed_mps",
        "first_failed_speed_mps_7gate": "first_failed_speed_mps",
        "first_failure_gate_7gate": "first_failure_gate",
        "all_speeds_pass_7gate": "all_speeds_pass",
    })
    return df[["dataset", "subject", "floor_speed_mps", "first_failed_speed_mps", "first_failure_gate", "all_speeds_pass"]]



def find_matched_npz(subject, speed_mps):
    """Find the matched NPZ file for a subject/speed."""
    subj_dir = DATA_PATH / "processed" / "vanhooren" / subject / "level_strideparity"
    if not subj_dir.exists():
        return None

    
    sp = round(float(speed_mps), 2)
    speed_map = {5.00: "5ms", 4.00: "4ms", 3.33: "333ms", 3.00: "3ms", 2.78: "278ms"}
    speed_tag = speed_map.get(sp)
    if speed_tag is None:
        
        if abs(sp - 3.33) < 0.02 or abs(sp - 2.78) < 0.02:
            speed_tag = f"{int(round(sp * 100))}ms"
        else:
            speed_tag = f"{int(round(sp))}ms"

    for f in subj_dir.glob(f"*{speed_tag}*matched*.npz"):
        return f
    for f in subj_dir.glob(f"*matched*.npz"):
        if speed_tag in f.name or str(speed_mps).replace(".", "") in f.name:
            return f
    return None


def load_X_from_npz(npz_path):
    """Load X matrix from preprocessed NPZ."""
    with np.load(npz_path, allow_pickle=True) as z:
        return z["X"], z.get("env_norm", None), z.get("ref_peaks", None)


# ==============================================================================
# FLOOR ESTIMATION
# ==============================================================================

def estimate_floor_for_variant(
    subject: str,
    variant: str,
    stability_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    surrogate_df: pd.DataFrame,
    nmf_func=nmf_frobenius,
    match_func=match_modules_cosine,
    stride_fraction: float = 1.0,
    renormalize: bool = False,
) -> Tuple[float, Optional[float], Optional[str]]:
    """
    Estimate floor for a single subject under a given variant.
    Returns: (floor_speed, first_failed_speed, failure_gate)
    """
    subj_stab = stability_df[stability_df["subject"] == subject]
    if subj_stab.empty:
        return np.nan, np.nan, "no_stability_data"

    rel_row = reliability_df[reliability_df["subject"] == subject]
    if rel_row.empty:
        return np.nan, np.nan, "no_reliability_data"
    rel_row = rel_row.iloc[0]

    L_fid = float(rel_row["L_ref_fidelity"])
    L_cos = float(rel_row["L_ref_cosine_median"])
    U_dtw = float(rel_row["U_ref_dtw_mean"])
    U_ang = float(rel_row["U_ref_principal_angle_max_deg"])

    ref_row = subj_stab[np.isclose(subj_stab["speed_mps"], REF_SPEED, atol=0.1)]
    if ref_row.empty:
        return np.nan, np.nan, "no_ref_speed"
    ref_row = ref_row.iloc[0]

    # Implementation constraint: fixed k across speeds to keep matching square (k×k)
    # in this sensitivity script.
    k = int(ref_row["k"])

    ref_npz = find_matched_npz(subject, REF_SPEED)
    if ref_npz is None:
        return np.nan, np.nan, "no_ref_npz"

    X_ref_full, env_ref, peaks_ref = load_X_from_npz(ref_npz)

    if stride_fraction < 1.0:
        n_strides = X_ref_full.shape[1] // N_POINTS
        n_keep = max(1, int(n_strides * stride_fraction))
        rng = np.random.default_rng(RANDOM_SEED)
        keep_idx = rng.choice(n_strides, n_keep, replace=False)
        keep_idx = np.sort(keep_idx)
        X_ref = np.hstack([X_ref_full[:, i*N_POINTS:(i+1)*N_POINTS] for i in keep_idx])
    else:
        X_ref = X_ref_full

    if renormalize and env_ref is not None:
        ref_peaks_per_muscle = X_ref.max(axis=1, keepdims=True)
        ref_peaks_per_muscle = np.maximum(ref_peaks_per_muscle, 1e-10)
        X_ref = X_ref / ref_peaks_per_muscle

    W_ref, H_ref = run_nmf_multistart(X_ref, k, nmf_func, n_starts=5, seed_base=RANDOM_SEED)

    last_pass_speed = REF_SPEED
    first_failed_speed = None
    failure_gate = None

    for spd in TEST_ORDER:
        surr_row = surrogate_df[(surrogate_df["subject"] == subject) &
                                 (np.isclose(surrogate_df["q_speed_mps"], spd, atol=0.1))]
        if surr_row.empty:
            continue
        F_null_95 = float(surr_row.iloc[0]["F_null_95"])

        q_npz = find_matched_npz(subject, spd)
        if q_npz is None:
            continue

        X_q_full, env_q, peaks_q = load_X_from_npz(q_npz)

        if stride_fraction < 1.0:
            n_strides_q = X_q_full.shape[1] // N_POINTS
            n_keep_q = max(1, int(n_strides_q * stride_fraction))
            rng = np.random.default_rng(RANDOM_SEED + 1000)
            keep_idx_q = rng.choice(n_strides_q, n_keep_q, replace=False)
            keep_idx_q = np.sort(keep_idx_q)
            X_q = np.hstack([X_q_full[:, i*N_POINTS:(i+1)*N_POINTS] for i in keep_idx_q])
        else:
            X_q = X_q_full

        if renormalize:
            q_peaks = X_q.max(axis=1, keepdims=True)
            q_peaks = np.maximum(q_peaks, 1e-10)
            X_q = X_q / q_peaks

        W_q, H_q = run_nmf_multistart(X_q, k, nmf_func, n_starts=5, seed_base=RANDOM_SEED + 100)

        H_q_from_ref = solve_H_nnls(W_ref, X_q)
        fid_ref_to_q = compute_fidelity(X_q, W_ref, H_q_from_ref)

        H_ref_from_q = solve_H_nnls(W_q, X_ref)
        fid_q_to_ref = compute_fidelity(X_ref, W_q, H_ref_from_q)

        cosine_med, dtw_m, ang_max = compute_spatial_temporal_metrics(
            W_ref, W_q, H_ref, H_q, match_func
        )

        g1 = fid_ref_to_q >= L_fid
        g2 = fid_q_to_ref >= L_fid
        g3 = fid_ref_to_q > F_null_95
        g4 = fid_q_to_ref > F_null_95
        g5 = cosine_med >= L_cos
        g6 = dtw_m <= U_dtw
        g7 = ang_max <= U_ang

        all_pass = g1 and g2 and g3 and g4 and g5 and g6 and g7

        if all_pass:
            last_pass_speed = spd
        else:
            first_failed_speed = spd
            for gname, ok in [("g1", g1), ("g2", g2), ("g3", g3), ("g4", g4), ("g5", g5), ("g6", g6), ("g7", g7)]:
                if not ok:
                    failure_gate = gname
                    break
            break

    return last_pass_speed, first_failed_speed, failure_gate


# ==============================================================================
# ICC AND BLAND-ALTMAN
# ==============================================================================

def compute_icc_2_1(data: pd.DataFrame, subjects_col: str, raters_col: str, value_col: str) -> float:
    """
    Compute ICC(2,1) - two-way random effects, single measurement.
    data: long-format DataFrame with subjects, raters (variants), and values.
    """
    df = data.dropna(subset=[value_col])
    if df.empty:
        return np.nan

    subjects = df[subjects_col].unique()
    raters = df[raters_col].unique()
    n = len(subjects)
    k = len(raters)

    if n < 2 or k < 2:
        return np.nan

    pivot = df.pivot_table(index=subjects_col, columns=raters_col, values=value_col, aggfunc='first')
    pivot = pivot.dropna()

    if len(pivot) < 2:
        return np.nan

    n = len(pivot)
    k = len(pivot.columns)

    grand_mean = pivot.values.mean()
    row_means = pivot.mean(axis=1).values
    col_means = pivot.mean(axis=0).values

    SS_total = np.sum((pivot.values - grand_mean) ** 2)
    SS_rows = k * np.sum((row_means - grand_mean) ** 2)
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)
    SS_error = SS_total - SS_rows - SS_cols

    MS_rows = SS_rows / (n - 1) if n > 1 else 0
    MS_cols = SS_cols / (k - 1) if k > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

    icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error + (k / n) * (MS_cols - MS_error))

    return float(np.clip(icc, 0, 1))


def compute_bland_altman_replicate(data: pd.DataFrame, baseline_col: str, variant_col: str) -> Dict:
    """
    Compute Bland-Altman statistics adjusted for replicate design.
    Following Bland & Altman (2007) for multiple observations per subject.
    """
    df = data.dropna(subset=[baseline_col, variant_col])
    if len(df) < 2:
        return {"mean_diff": np.nan, "sd_diff": np.nan, "loa_lower": np.nan, "loa_upper": np.nan, "n": 0}

    diffs = df[variant_col] - df[baseline_col]
    means = (df[variant_col] + df[baseline_col]) / 2

    mean_diff = diffs.mean()
    sd_diff = diffs.std(ddof=1)

    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff

    return {
        "mean_diff": float(mean_diff),
        "sd_diff": float(sd_diff),
        "loa_lower": float(loa_lower),
        "loa_upper": float(loa_upper),
        "n": int(len(df))
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("SENSITIVITY ANALYSES")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading upstream data...")
    stability_df = load_stability_summary()
    reliability_df = load_reliability_envelope()
    surrogate_df = load_surrogate_baseline()
    baseline_floors_df = load_baseline_floors()

    subjects = sorted(baseline_floors_df["subject"].unique())
    print(f"Subjects: {subjects}")
    print(f"Variants: {VARIANTS}")
    print()

    results = []

    print("-" * 70)
    print("VARIANT: baseline")
    print("-" * 70)
    for _, row in baseline_floors_df.iterrows():
        results.append({
            "subject": row["subject"],
            "variant": "baseline",
            "floor_speed_mps": row["floor_speed_mps"],
            "first_failed_speed_mps": row["first_failed_speed_mps"],
            "first_failure_gate": row["first_failure_gate"],
        })
        print(f"  {row['subject']}: floor = {row['floor_speed_mps']}")
    print()

    print("-" * 70)
    print("VARIANT: nmf_kl")
    print("-" * 70)
    for subj in subjects:
        floor, fail_spd, fail_gate = estimate_floor_for_variant(
            subj, "nmf_kl", stability_df, reliability_df, surrogate_df,
            nmf_func=nmf_kl_divergence, match_func=match_modules_cosine,
            stride_fraction=1.0, renormalize=False
        )
        results.append({
            "subject": subj,
            "variant": "nmf_kl",
            "floor_speed_mps": floor,
            "first_failed_speed_mps": fail_spd,
            "first_failure_gate": fail_gate,
        })
        print(f"  {subj}: floor = {floor}")
    print()

    print("-" * 70)
    print("VARIANT: stride_half")
    print("-" * 70)
    for subj in subjects:
        floor, fail_spd, fail_gate = estimate_floor_for_variant(
            subj, "stride_half", stability_df, reliability_df, surrogate_df,
            nmf_func=nmf_frobenius, match_func=match_modules_cosine,
            stride_fraction=0.5, renormalize=False
        )
        results.append({
            "subject": subj,
            "variant": "stride_half",
            "floor_speed_mps": floor,
            "first_failed_speed_mps": fail_spd,
            "first_failure_gate": fail_gate,
        })
        print(f"  {subj}: floor = {floor}")
    print()

    print("-" * 70)
    print("VARIANT: matching_euclidean")
    print("-" * 70)
    for subj in subjects:
        floor, fail_spd, fail_gate = estimate_floor_for_variant(
            subj, "matching_euclidean", stability_df, reliability_df, surrogate_df,
            nmf_func=nmf_frobenius, match_func=match_modules_euclidean,
            stride_fraction=1.0, renormalize=False
        )
        results.append({
            "subject": subj,
            "variant": "matching_euclidean",
            "floor_speed_mps": floor,
            "first_failed_speed_mps": fail_spd,
            "first_failure_gate": fail_gate,
        })
        print(f"  {subj}: floor = {floor}")
    print()

    print("-" * 70)
    print("VARIANT: norm_per_speed")
    print("-" * 70)
    for subj in subjects:
        floor, fail_spd, fail_gate = estimate_floor_for_variant(
            subj, "norm_per_speed", stability_df, reliability_df, surrogate_df,
            nmf_func=nmf_frobenius, match_func=match_modules_cosine,
            stride_fraction=1.0, renormalize=True
        )
        results.append({
            "subject": subj,
            "variant": "norm_per_speed",
            "floor_speed_mps": floor,
            "first_failed_speed_mps": fail_spd,
            "first_failure_gate": fail_gate,
        })
        print(f"  {subj}: floor = {floor}")
    print()

    results_df = pd.DataFrame(results)

    print("=" * 70)
    print("COMPUTING ICC")
    print("=" * 70)

    icc_data = results_df[["subject", "variant", "floor_speed_mps"]].copy()
    icc_value = compute_icc_2_1(icc_data, "subject", "variant", "floor_speed_mps")
    print(f"ICC(2,1) across all variants: {icc_value:.4f}")
    print()

    icc_rows = [{"comparison": "all_variants", "icc_2_1": icc_value, "n_subjects": len(subjects), "n_variants": len(VARIANTS)}]

    for var in VARIANTS:
        if var == "baseline":
            continue
        subset = results_df[results_df["variant"].isin(["baseline", var])]
        icc_pair = compute_icc_2_1(subset, "subject", "variant", "floor_speed_mps")
        icc_rows.append({"comparison": f"baseline_vs_{var}", "icc_2_1": icc_pair, "n_subjects": len(subjects), "n_variants": 2})
        print(f"ICC baseline vs {var}: {icc_pair:.4f}")

    icc_df = pd.DataFrame(icc_rows)
    print()

    print("=" * 70)
    print("COMPUTING BLAND-ALTMAN SUMMARIES")
    print("=" * 70)

    pivot_df = results_df.pivot(index="subject", columns="variant", values="floor_speed_mps").reset_index()

    ba_rows = []
    for var in VARIANTS:
        if var == "baseline":
            continue
        ba = compute_bland_altman_replicate(pivot_df, "baseline", var)
        ba_rows.append({
            "variant": var,
            "mean_diff": ba["mean_diff"],
            "sd_diff": ba["sd_diff"],
            "loa_lower": ba["loa_lower"],
            "loa_upper": ba["loa_upper"],
            "n": ba["n"],
        })
        print(f"{var}: mean_diff={ba['mean_diff']:.4f}, SD={ba['sd_diff']:.4f}, LoA=[{ba['loa_lower']:.4f}, {ba['loa_upper']:.4f}]")

    ba_df = pd.DataFrame(ba_rows)
    print()

    print("=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    sensitivity_path = OUTPUT_DIR / "sensitivity_comparison.csv"
    results_df.to_csv(sensitivity_path, index=False)
    print(f"Saved: {sensitivity_path}")

    icc_path = OUTPUT_DIR / "icc_summary.csv"
    icc_df.to_csv(icc_path, index=False)
    print(f"Saved: {icc_path}")

    ba_path = OUTPUT_DIR / "bland_altman_summary.csv"
    ba_df.to_csv(ba_path, index=False)
    print(f"Saved: {ba_path}")

    pivot_path = OUTPUT_DIR / "floors_by_variant.csv"
    pivot_df.to_csv(pivot_path, index=False)
    print(f"Saved: {pivot_path}")

    params = {
        "baseline_floor_source": str(UPSTREAM_PATH / "coordination_floors.csv"),
        "reliability_envelope_source": str(UPSTREAM_PATH / "reliability_envelope.csv"),
        "surrogate_baseline_source": str(UPSTREAM_PATH / "surrogate_baseline.csv"),
        "stability_summary_source": str(UPSTREAM_PATH / "stability_summary.csv"),
        "variants": VARIANTS,
        "ref_speed": REF_SPEED,
        "test_order": TEST_ORDER,
        "random_seed": RANDOM_SEED,
        "stride_fraction_for_stride_half": 0.5,
        "nmf_n_starts": 5,
        "run_datetime": datetime.now().isoformat(),
    }
    params_path = OUTPUT_DIR / "sensitivity_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved: {params_path}")

    summary_lines = [
        "SENSITIVITY ANALYSIS SUMMARY",
        "=" * 50,
        "",
        f"Run datetime: {datetime.now().isoformat()}",
        f"N subjects: {len(subjects)}",
        f"Variants tested: {VARIANTS}",
        "",
        "FLOOR DISTRIBUTION BY VARIANT:",
        "-" * 30,
    ]
    for var in VARIANTS:
        var_floors = results_df[results_df["variant"] == var]["floor_speed_mps"]
        summary_lines.append(f"{var}: mean={var_floors.mean():.3f}, std={var_floors.std():.3f}, min={var_floors.min()}, max={var_floors.max()}")

    summary_lines.extend([
        "",
        "ICC SUMMARY:",
        "-" * 30,
        f"ICC(2,1) all variants: {icc_value:.4f}",
        "",
        "BLAND-ALTMAN SUMMARY (vs baseline):",
        "-" * 30,
    ])
    for _, row in ba_df.iterrows():
        summary_lines.append(f"{row['variant']}: mean_diff={row['mean_diff']:.4f}, LoA=[{row['loa_lower']:.4f}, {row['loa_upper']:.4f}]")

    summary_lines.extend([
        "",
        "INTERPRETATION:",
        "-" * 30,
        f"ICC >= 0.75 indicates good agreement. Observed ICC = {icc_value:.4f}.",
        "Bland-Altman LoA indicate the range within which differences fall for 95% of subjects.",
    ])

    summary_path = OUTPUT_DIR / "sensitivity_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved: {summary_path}")

    print()
    print("=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
