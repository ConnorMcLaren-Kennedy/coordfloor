"""
Script: 04_compute_reliability.py
Purpose: Compute within-speed reliability bounds at reference speed via bootstrap.

Inputs:
    - outputs/stability_summary.csv (from 02c_nmf_stability.py)

Outputs:
    - outputs/reliability_envelope.csv (subject-specific bounds)
    - outputs/reliability_distributions/*.npz (raw bootstrap distributions)

Reliability bounds (per subject, at reference speed 5.0 m/s):
    Lower bounds (must be >= to pass):
        - Fidelity (10th percentile, 95% lower confidence bound)
        - Cosine median (10th percentile, 95% lower confidence bound)
    Upper bounds (must be <= to pass):
        - DTW mean (90th percentile, 95% upper confidence bound)
        - Principal angle max (90th percentile, 95% upper confidence bound)

Algorithm:
    Moving block bootstrap half-splits, 2000 replicates per subject.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

# =========================
# INPUT PATHS
# =========================
# Set via environment variable or command line
# Expects: stability_summary.csv from NMF extraction step
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))

# =========================
# RELEASE-LOCAL OUTPUT PATHS
# =========================
RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"
DIST_DIR = OUTPUT_DIR / "reliability_distributions"

# =========================
# CONSTANTS
# =========================
SAMPLES_PER_STRIDE = 200

P_COVERAGE = 0.90               # target: 90% coverage => 10th/90th percentiles
GAMMA_CONFIDENCE = 0.95         # target: one-sided 95% confidence bound
ALPHA = 1.0 - GAMMA_CONFIDENCE  # 0.05

# Bootstrap settings
N_BOOT = 2000   # number of half-split bootstrap replicates per subject
N_CI = 2000     # bootstrap replicates to form CI on the target quantile

# NMF settings
NMF_MAX_ITER = 500
NMF_N_INITS = 10
RANDOM_SEED = 42


# =========================
# UTILITIES
# =========================
def _require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}. Found columns: {list(df.columns)}")


def load_reference_rows() -> pd.DataFrame:
    """
    Loads stability_summary.csv and returns vanhooren reference-speed (5.0) rows that passed stability.
    Must contain at least: dataset, subject, speed_mps, pass_stability, k, file_x
    """
    p = UPSTREAM_PATH / "stability_summary.csv"
    df = pd.read_csv(p)

    _require_columns(df, ["dataset", "subject", "speed_mps", "pass_stability"], "stability_summary.csv")

    vh = df[(df["dataset"] == "vanhooren") & (df["speed_mps"] == 5.0) & (df["pass_stability"] == True)].copy()

    _require_columns(vh, ["subject"], "vanhooren reference slice")

    # These columns are required for computation
    _require_columns(vh, ["k", "file_x"], "vanhooren reference slice")

    vh = vh.sort_values(["subject"]).reset_index(drop=True)
    print(f"Found {len(vh)} vanhooren subjects at reference speed (5.0 m/s) with pass_stability=True")
    return vh


def load_X_matrix(file_x_path: str) -> np.ndarray:
    X = np.load(file_x_path)["X"]
    X = np.asarray(X, dtype=float)
    # enforce non-negativity (should already be non-negative)
    X[X < 0] = 0.0
    return X


def compute_optimal_block_length(X: np.ndarray, n_strides: int) -> int:
    """
    Patton-Politis-White automatic selection, bounded to [3, floor(n_strides/8)].
    """
    try:
        from arch.bootstrap import optimal_block_length
        stride_means = X[0, :].reshape(n_strides, SAMPLES_PER_STRIDE).mean(axis=1)
        result = optimal_block_length(stride_means)
        b_opt = int(np.round(result.iloc[0]["circular"]))
    except Exception as e:
        warnings.warn(f"optimal_block_length failed: {e}. Using fallback.")
        b_opt = max(3, n_strides // 10)

    b_min = 3
    b_max = max(3, n_strides // 8)
    b = max(b_min, min(b_opt, b_max))

    # prevent degeneracy if extremely short
    if n_strides < 4 * b:
        b = max(3, n_strides // 4)

    return int(b)


def mbb_sample_indices(n_strides: int, b: int, n_draw: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving-block bootstrap indices at stride level (circular wrap).
    Returns length n_draw indices in [0, n_strides-1].
    """
    n_blocks = int(np.ceil(n_draw / b))
    starts = rng.integers(0, n_strides, size=n_blocks)
    idx = []
    for s in starts:
        for j in range(b):
            idx.append((int(s) + j) % n_strides)
    return np.array(idx[:n_draw], dtype=int)


def run_nmf_best(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Fits NMF (solver='mu', init='nndsvda') and returns (W, H) in synergy form:
      X ≈ W @ H
    where W is (m x k) and H is (k x time).

    Note: 'nndsvda' initialisation is deterministic; the restart loop is mainly
    kept to support alternative initialisations (e.g., 'random'/'nndsvdar').
    """
    best_err = np.inf
    best_W = None
    best_H = None

    # X is (m x time). sklearn NMF expects (n_samples x n_features).
    X_t = X.T  # (time x m)

    for i in range(NMF_N_INITS):
        try:
            model = NMF(
                n_components=k,
                init="nndsvda",
                max_iter=NMF_MAX_ITER,
                random_state=int(seed + i),
                solver="mu",
            )
            W_time = model.fit_transform(X_t)   # (time x k)
            H_muscle = model.components_        # (k x m)

            W = H_muscle.T                      # (m x k)
            H = W_time.T                        # (k x time)

            recon = W @ H
            err = np.linalg.norm(X - recon, ord="fro")

            if err < best_err:
                best_err = err
                best_W = W
                best_H = H
        except Exception:
            continue

    return best_W, best_H


def compute_fidelity_nnls(X_test: np.ndarray, W_train: np.ndarray) -> float:
    """
    Fidelity = 1 - SSE/SSE0, where H is solved by NNLS at each time sample.
    """
    denom = float(np.sum(X_test ** 2))
    if denom <= 0:
        return np.nan

    n_muscles, n_samples = X_test.shape
    k = W_train.shape[1]
    H_proj = np.zeros((k, n_samples), dtype=float)

    for t in range(n_samples):
        H_proj[:, t], _ = nnls(W_train, X_test[:, t])

    recon = W_train @ H_proj
    num = float(np.sum((X_test - recon) ** 2))
    return 1.0 - (num / denom)


def cosine_hungarian_match(W_A: np.ndarray, W_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (row_ind, col_ind, matched_cosines) where matching maximizes cosine similarity.
    """
    eps = 1e-12
    WA = W_A / (np.linalg.norm(W_A, axis=0, keepdims=True) + eps)
    WB = W_B / (np.linalg.norm(W_B, axis=0, keepdims=True) + eps)

    sim = WA.T @ WB  # (k x k), cosine similarities
    cost = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost)
    matched = sim[row_ind, col_ind]
    return row_ind, col_ind, matched


def principal_angle_max_deg(W_A: np.ndarray, W_B: np.ndarray) -> float:
    """
    Björck-Golub principal angles: max angle between column spaces, in degrees.
    """
    # Orthonormal bases
    QA, _ = np.linalg.qr(W_A)
    QB, _ = np.linalg.qr(W_B)

    M = QA.T @ QB
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)

    angles = np.degrees(np.arccos(s))
    return float(np.max(angles))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Classic DTW distance (no window), absolute-difference cost.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n = a.size
    m = b.size
    D = np.full((n + 1, m + 1), np.inf, dtype=float)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[n, m])


def mean_activation_waveforms(H: np.ndarray, n_strides: int) -> np.ndarray:
    """
    H is (k x (n_strides*T)). Returns mean waveform per module: (k x T).
    """
    k, total = H.shape
    T = SAMPLES_PER_STRIDE
    if total != n_strides * T:
        return None
    H3 = H.reshape(k, n_strides, T)
    return H3.mean(axis=1)


def compute_halfsplit_metrics(W_A, H_A, W_B, H_B, n_half_strides: int) -> tuple[float, float, float]:
    """
    Returns (cosine_median, dtw_mean, principal_angle_max_deg).
    """
    row_ind, col_ind, matched_cos = cosine_hungarian_match(W_A, W_B)
    cosine_median = float(np.median(matched_cos))

    # Temporal (DTW) on mean activation waveforms, matched using the W matching
    HA_mean = mean_activation_waveforms(H_A, n_half_strides)
    HB_mean = mean_activation_waveforms(H_B, n_half_strides)
    if HA_mean is None or HB_mean is None:
        return np.nan, np.nan, np.nan

    dtw_vals = []
    for ia, ib in zip(row_ind, col_ind):
        dtw_vals.append(dtw_distance(HA_mean[ia, :], HB_mean[ib, :]))
    dtw_mean = float(np.mean(dtw_vals))

    # Subspace angles
    ang_max = principal_angle_max_deg(W_A, W_B)

    return cosine_median, dtw_mean, ang_max


def bootstrap_quantile_bound(values: np.ndarray, q: float, side: str, n_ci: int, seed: int) -> tuple[float, float]:
    """
    Returns (q_estimate, bound) where bound is a one-sided confidence bound on the quantile:
      - side='lower' => 95% lower bound => percentile(alpha)
      - side='upper' => 95% upper bound => percentile(1-alpha)
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return np.nan, np.nan

    q_est = float(np.quantile(values, q))

    rng = np.random.default_rng(seed)
    n = values.size
    qs = np.empty(n_ci, dtype=float)

    for i in range(n_ci):
        samp = rng.choice(values, size=n, replace=True)
        qs[i] = np.quantile(samp, q)

    if side == "lower":
        bound = float(np.quantile(qs, ALPHA))
    elif side == "upper":
        bound = float(np.quantile(qs, 1.0 - ALPHA))
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    return q_est, bound


def compute_subject_envelope(subject_row: pd.Series, seed: int) -> dict:
    subject = str(subject_row["subject"])
    k = int(subject_row["k"])
    file_x = subject_row["file_x"]

    X = load_X_matrix(file_x)
    m, n_samples = X.shape
    n_strides = int(n_samples // SAMPLES_PER_STRIDE)

    if n_strides * SAMPLES_PER_STRIDE != n_samples:
        raise RuntimeError(f"{subject}: X samples not divisible by SAMPLES_PER_STRIDE={SAMPLES_PER_STRIDE}.")

    b = compute_optimal_block_length(X, n_strides)

    # record disjoint K that would be available under original scheme (for transparency)
    K_disjoint = int(n_strides // (2 * b))

    half_n = int(n_strides // 2)
    if half_n < 2:
        return {
            "dataset": "vanhooren",
            "subject": subject,
            "trial": 0,
            "n_strides_ref": n_strides,
            "block_length_b": b,
            "K_disjoint": K_disjoint,
            "N_boot_target": N_BOOT,
            "N_boot_success": 0,
            "warning_flag": "too_few_strides",
        }

    # reshape to stride-wise for fast sampling
    X3 = X.reshape(m, n_strides, SAMPLES_PER_STRIDE)

    rng = np.random.default_rng(seed)

    fid_vals = []
    cos_vals = []
    dtw_vals = []
    ang_vals = []

    successes = 0
    attempts = 0
    max_attempts = int(N_BOOT * 20)

    while successes < N_BOOT and attempts < max_attempts:
        attempts += 1

        idxA = mbb_sample_indices(n_strides, b, half_n, rng)
        idxB = mbb_sample_indices(n_strides, b, half_n, rng)

        X_A = X3[:, idxA, :].reshape(m, half_n * SAMPLES_PER_STRIDE)
        X_B = X3[:, idxB, :].reshape(m, half_n * SAMPLES_PER_STRIDE)

        W_A, H_A = run_nmf_best(X_A, k, seed=seed + attempts * 2)
        W_B, H_B = run_nmf_best(X_B, k, seed=seed + attempts * 2 + 1)

        if W_A is None or H_A is None or W_B is None or H_B is None:
            continue

        f_ab = compute_fidelity_nnls(X_B, W_A)
        f_ba = compute_fidelity_nnls(X_A, W_B)

        cos_med, dtw_mean, ang_max = compute_halfsplit_metrics(W_A, H_A, W_B, H_B, half_n)

        if not (np.isfinite(f_ab) and np.isfinite(f_ba) and np.isfinite(cos_med) and np.isfinite(dtw_mean) and np.isfinite(ang_max)):
            continue

        fid_vals.append(float(f_ab))
        fid_vals.append(float(f_ba))
        cos_vals.append(float(cos_med))
        dtw_vals.append(float(dtw_mean))
        ang_vals.append(float(ang_max))

        successes += 1

        if successes % 100 == 0:
            print(f"  {subject}: bootstrap successes {successes}/{N_BOOT}")

    fid_vals = np.array(fid_vals, dtype=float)
    cos_vals = np.array(cos_vals, dtype=float)
    dtw_vals = np.array(dtw_vals, dtype=float)
    ang_vals = np.array(ang_vals, dtype=float)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DIST_DIR / f"vanhooren_{subject}_ref_boot.npz",
        dataset="vanhooren",
        subject=subject,
        k=k,
        n_strides_ref=n_strides,
        block_length_b=b,
        half_n=half_n,
        N_boot_target=N_BOOT,
        N_boot_success=successes,
        fid_dist=fid_vals,
        cosine_median_dist=cos_vals,
        dtw_mean_dist=dtw_vals,
        principal_angle_max_deg_dist=ang_vals,
    )

    # Quantile targets
    q_low = 1.0 - P_COVERAGE  # 0.10
    q_high = P_COVERAGE       # 0.90

    # Fidelity / Cosine: lower bound on 10th percentile
    fid_q10, L_ref_fid = bootstrap_quantile_bound(fid_vals, q=q_low, side="lower", n_ci=N_CI, seed=seed + 101)
    cos_q10, L_ref_cos = bootstrap_quantile_bound(cos_vals, q=q_low, side="lower", n_ci=N_CI, seed=seed + 102)

    # DTW / Angle: upper bound on 90th percentile
    dtw_q90, U_ref_dtw = bootstrap_quantile_bound(dtw_vals, q=q_high, side="upper", n_ci=N_CI, seed=seed + 103)
    ang_q90, U_ref_ang = bootstrap_quantile_bound(ang_vals, q=q_high, side="upper", n_ci=N_CI, seed=seed + 104)

    warn = ""
    if successes < N_BOOT:
        warn = "bootstrap_shortfall"
    if not np.isfinite(L_ref_fid) or not np.isfinite(L_ref_cos) or not np.isfinite(U_ref_dtw) or not np.isfinite(U_ref_ang):
        warn = (warn + ";invalid_bounds").strip(";")

    return {
        "dataset": "vanhooren",
        "subject": subject,
        "trial": 0,
        "k_ref": k,
        "n_strides_ref": n_strides,
        "block_length_b": b,
        "K_disjoint": K_disjoint,
        "half_n": half_n,
        "N_boot_target": N_BOOT,
        "N_boot_success": successes,
        "n_fidelity_values": int(fid_vals.size),
        "n_metric_values": int(cos_vals.size),

        # distribution summaries (optional but useful)
        "fid_min": float(np.min(fid_vals)) if fid_vals.size else np.nan,
        "fid_mean": float(np.mean(fid_vals)) if fid_vals.size else np.nan,
        "fid_max": float(np.max(fid_vals)) if fid_vals.size else np.nan,

        "cos_min": float(np.min(cos_vals)) if cos_vals.size else np.nan,
        "cos_mean": float(np.mean(cos_vals)) if cos_vals.size else np.nan,
        "cos_max": float(np.max(cos_vals)) if cos_vals.size else np.nan,

        "dtw_min": float(np.min(dtw_vals)) if dtw_vals.size else np.nan,
        "dtw_mean": float(np.mean(dtw_vals)) if dtw_vals.size else np.nan,
        "dtw_max": float(np.max(dtw_vals)) if dtw_vals.size else np.nan,

        "ang_min": float(np.min(ang_vals)) if ang_vals.size else np.nan,
        "ang_mean": float(np.mean(ang_vals)) if ang_vals.size else np.nan,
        "ang_max": float(np.max(ang_vals)) if ang_vals.size else np.nan,

        # quantile estimates + bounds
        "fid_q10_est": fid_q10,
        "L_ref_fidelity": L_ref_fid,

        "cos_q10_est": cos_q10,
        "L_ref_cosine_median": L_ref_cos,

        "dtw_q90_est": dtw_q90,
        "U_ref_dtw_mean": U_ref_dtw,

        "ang_q90_est": ang_q90,
        "U_ref_principal_angle_max_deg": U_ref_ang,

        # Backward-compatible alias (old name used by v1)
        "L_ref": L_ref_fid,

        "warning_flag": warn,
    }


def main() -> int:
    print("=" * 72)
    print("WITHIN-SPEED RELIABILITY ENVELOPE (BOOTSTRAP)")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vh_ref = load_reference_rows()
    if len(vh_ref) == 0:
        print("ERROR: no vanhooren reference rows found.")
        return 1

    rows = []
    for i, row in vh_ref.iterrows():
        subject = str(row["subject"])
        print(f"\nProcessing {subject} ({i+1}/{len(vh_ref)})")
        out = compute_subject_envelope(row, seed=RANDOM_SEED + i * 1000)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "reliability_envelope.csv"
    out_df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("Saved distributions to:", DIST_DIR)

    # quick summary
    print("\nSUMMARY (reliability bounds):")
    print("Rows:", len(out_df))
    print("Subjects with warnings:", int(out_df["warning_flag"].fillna("").astype(str).ne("").sum()))
    print("L_ref_fidelity range:", float(out_df["L_ref_fidelity"].min()), "-", float(out_df["L_ref_fidelity"].max()))
    print("L_ref_cosine_median range:", float(out_df["L_ref_cosine_median"].min()), "-", float(out_df["L_ref_cosine_median"].max()))
    print("U_ref_dtw_mean range:", float(out_df["U_ref_dtw_mean"].min()), "-", float(out_df["U_ref_dtw_mean"].max()))
    print("U_ref_principal_angle_max_deg range:", float(out_df["U_ref_principal_angle_max_deg"].min()), "-", float(out_df["U_ref_principal_angle_max_deg"].max()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
