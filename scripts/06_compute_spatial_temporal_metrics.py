"""
Script: 06_compute_spatial_temporal_metrics.py
Purpose: Compute spatial and temporal similarity metrics between speeds.

Inputs:
    - outputs/coordination_fidelity.csv (from 03_compute_fidelity.py)
    - outputs/stability_summary.csv (from 02c_nmf_stability.py)

Outputs:
    - outputs/spatial_temporal_metrics.csv

Metrics computed:
    - Cosine similarity (median, min, mean) between matched W columns
    - Principal angles (max, mean) between W subspaces
    - DTW distance (mean, max) between matched H activation waveforms
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment


# =========================
# INPUT PATHS
# =========================
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))

# =========================
# RELEASE-LOCAL OUTPUTS
# =========================
RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"

# =========================
# CONSTANTS
# =========================
SAMPLES_PER_STRIDE = 200
REF_SPEED = 5.0

# Keep consistent with the existing pipeline
NMF_MAX_ITER = 500
NMF_N_INITS = 10
RANDOM_SEED = 42


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns {missing}. Found: {list(df.columns)}")


def _speed_key(x: float) -> float:
    # Avoid float equality issues between 3.33 vs 3.3300000000000001, etc.
    return float(f"{float(x):.2f}")


def load_stability_lookup() -> dict[tuple[str, float], str]:
    p = UPSTREAM_PATH / "stability_summary.csv"
    df = pd.read_csv(p)

    _require_columns(df, ["dataset", "subject", "speed_mps", "file_x", "pass_stability"], "stability_summary.csv")
    vh = df[df["dataset"] == "vanhooren"].copy()

    # Robust boolean parse: accept True/False, 1/0, "true"/"false"
    ps = vh["pass_stability"]
    if ps.dtype != bool:
        vh["pass_stability"] = ps.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    
    vh = vh[vh["pass_stability"] == True].copy()

    lookup: dict[tuple[str, float], str] = {}
    for _, r in vh.iterrows():
        key = (str(r["subject"]), _speed_key(r["speed_mps"]))
        lookup[key] = str(r["file_x"]).strip()
    return lookup


def load_fidelity_pairs() -> pd.DataFrame:
    p = UPSTREAM_PATH / "coordination_fidelity.csv"
    df = pd.read_csv(p)

    _require_columns(
        df,
        ["dataset", "subject", "trial", "ref_speed_mps", "q_speed_mps", "k_ref", "k_q", "n_strides_ref", "n_strides_q"],
        "coordination_fidelity.csv",
    )

    vh = df[df["dataset"] == "vanhooren"].copy()
    vh = vh[vh["ref_speed_mps"] == REF_SPEED].copy()
    vh = vh.sort_values(["subject", "q_speed_mps"]).reset_index(drop=True)

    return vh


def load_X(file_x_path: str) -> np.ndarray:
    X = np.load(file_x_path)["X"]
    X = np.asarray(X, dtype=float)
    X[X < 0] = 0.0
    return X


def run_nmf_best(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits NMF and returns (W, H) in synergy form: X ≈ W @ H
      W: (m x k), H: (k x time)
    Selects the lowest Frobenius reconstruction error across NMF_N_INITS restarts.
    With init='nndsvda' the fit is typically deterministic; the restart loop is
    kept for symmetry/robustness and to support alternative inits.
    """
    X_t = X.T  # (time x m)

    best_err = np.inf
    best_W = None
    best_H = None

    for i in range(NMF_N_INITS):
        model = NMF(
            n_components=int(k),
            init="nndsvda",
            solver="mu",
            max_iter=int(NMF_MAX_ITER),
            random_state=int(seed + i),
        )
        W_time = model.fit_transform(X_t)   # (time x k)
        H_muscle = model.components_        # (k x m)

        W = H_muscle.T                      # (m x k)
        H = W_time.T                        # (k x time)

        recon = W @ H
        err = float(np.linalg.norm(X - recon, ord="fro"))

        if err < best_err:
            best_err = err
            best_W = W
            best_H = H

    if best_W is None or best_H is None:
        raise RuntimeError("NMF failed for all initializations.")

    return best_W, best_H


def cosine_hungarian_match(W_A: np.ndarray, W_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hungarian matching maximizing cosine similarity between columns of W_A and W_B.
    Works for rectangular matrices (matches min(kA, kB) pairs).

    Returns (row_ind, col_ind, matched_cosines).
    """
    eps = 1e-12
    WA = W_A / (np.linalg.norm(W_A, axis=0, keepdims=True) + eps)
    WB = W_B / (np.linalg.norm(W_B, axis=0, keepdims=True) + eps)

    sim = WA.T @ WB  # (kA x kB)
    cost = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost)
    matched = sim[row_ind, col_ind]
    return row_ind, col_ind, matched


def principal_angles_deg(W_A: np.ndarray, W_B: np.ndarray) -> np.ndarray:
    """
    Björck-Golub principal angles (in degrees) between column spaces of W_A and W_B.
    Returns an array of angles of length min(kA, kB).
    """
    QA, _ = np.linalg.qr(W_A, mode="reduced")
    QB, _ = np.linalg.qr(W_B, mode="reduced")

    M = QA.T @ QB
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)

    return np.degrees(np.arccos(s))


def mean_activation_waveforms(H: np.ndarray, n_strides: int) -> np.ndarray:
    """
    H is (k x (n_strides*T)). Returns mean waveform per module: (k x T).
    """
    k, total = H.shape
    T = SAMPLES_PER_STRIDE
    if total != n_strides * T:
        raise RuntimeError(f"H has length {total} but expected {n_strides}*{T}={n_strides*T}.")
    H3 = H.reshape(k, n_strides, T)
    return H3.mean(axis=1)


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


def main() -> int:
    print("=" * 70)
    print("SPATIAL AND TEMPORAL METRICS")
    print("=" * 70)

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_fidelity_pairs()
    lookup = load_stability_lookup()

    print(f"Processing {len(pairs)} speed pairs...")

    rows = []
    for idx, r in pairs.iterrows():
        subject = str(r["subject"])
        trial = r["trial"]

        ref_speed = _speed_key(r["ref_speed_mps"])
        q_speed = _speed_key(r["q_speed_mps"])

        k_ref = int(r["k_ref"])
        k_q = int(r["k_q"])

        expected_n_ref = int(r["n_strides_ref"])
        expected_n_q = int(r["n_strides_q"])

        key_ref = (subject, ref_speed)
        key_q = (subject, q_speed)

        if key_ref not in lookup:
            raise RuntimeError(f"Missing file_x for {subject} @ {ref_speed} m/s (pass_stability=True).")
        if key_q not in lookup:
            raise RuntimeError(f"Missing file_x for {subject} @ {q_speed} m/s (pass_stability=True).")

        X_ref = load_X(lookup[key_ref])
        X_q = load_X(lookup[key_q])

        if X_ref.shape[1] % SAMPLES_PER_STRIDE != 0:
            raise RuntimeError(f"{subject} ref X length not divisible by {SAMPLES_PER_STRIDE}.")
        if X_q.shape[1] % SAMPLES_PER_STRIDE != 0:
            raise RuntimeError(f"{subject} q X length not divisible by {SAMPLES_PER_STRIDE}.")

        n_ref = int(X_ref.shape[1] // SAMPLES_PER_STRIDE)
        n_q = int(X_q.shape[1] // SAMPLES_PER_STRIDE)

        if (n_ref != expected_n_ref) or (n_q != expected_n_q):
            print(
                f"WARNING: {subject} @ {q_speed} m/s stride mismatch. "
                f"Expected (ref,q)=({expected_n_ref},{expected_n_q}) but X gives ({n_ref},{n_q})."
            )

        seed_ref = RANDOM_SEED + idx * 10 + 1
        seed_q = RANDOM_SEED + idx * 10 + 2

        W_ref, H_ref = run_nmf_best(X_ref, k_ref, seed_ref)
        W_q, H_q = run_nmf_best(X_q, k_q, seed_q)

        row_ind, col_ind, cosines = cosine_hungarian_match(W_ref, W_q)

        cosine_median = float(np.median(cosines)) if cosines.size else np.nan
        cosine_min = float(np.min(cosines)) if cosines.size else np.nan
        cosine_mean = float(np.mean(cosines)) if cosines.size else np.nan

        angs = principal_angles_deg(W_ref, W_q)
        principal_angle_max = float(np.max(angs)) if angs.size else np.nan
        principal_angle_mean = float(np.mean(angs)) if angs.size else np.nan

        Href_mean = mean_activation_waveforms(H_ref, n_ref)
        Hq_mean = mean_activation_waveforms(H_q, n_q)

        dtw_vals = np.asarray(
            [dtw_distance(Href_mean[int(i), :], Hq_mean[int(j), :]) for i, j in zip(row_ind, col_ind)],
            dtype=float,
        )

        dtw_mean = float(np.mean(dtw_vals)) if dtw_vals.size else np.nan
        dtw_max = float(np.max(dtw_vals)) if dtw_vals.size else np.nan

        rows.append(
            {
                "dataset": "vanhooren",
                "subject": subject,
                "trial": trial,
                "q_speed_mps": float(q_speed),
                "k_ref": k_ref,
                "k_q": k_q,
                "cosine_median": cosine_median,
                "cosine_min": cosine_min,
                "cosine_mean": cosine_mean,
                "principal_angle_max_deg": principal_angle_max,
                "principal_angle_mean_deg": principal_angle_mean,
                "dtw_mean": dtw_mean,
                "dtw_max": dtw_max,
            }
        )

        print(
            f"  {subject} @ {q_speed} m/s: "
            f"cos={cosine_median:.3f}, angle={principal_angle_max:.1f}deg, dtw={dtw_mean:.3f}"
        )

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "spatial_temporal_metrics.csv"
    out_df.to_csv(out_path, index=False)

    dt = time.time() - t0
    print("\nSaved:", out_path)
    print(f"Total time: {dt/60.0:.2f} minutes")

    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)
    print("Rows:", len(out_df))
    print("Cosine median range:", float(out_df["cosine_median"].min()), "-", float(out_df["cosine_median"].max()))
    print("Principal angle max range:", float(out_df["principal_angle_max_deg"].min()), "-", float(out_df["principal_angle_max_deg"].max()))
    print("DTW mean range:", float(out_df["dtw_mean"].min()), "-", float(out_df["dtw_mean"].max()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
