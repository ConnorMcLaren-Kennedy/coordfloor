"""
Script: 05_compute_surrogates.py
Purpose: Generate IAAFT surrogate baseline for superiority testing.

Inputs:
    - outputs/coordination_fidelity.csv (from 03_compute_fidelity.py)
    - outputs/stability_summary.csv (from 02c_nmf_stability.py)

Outputs:
    - outputs/surrogate_baseline.csv

Algorithm:
    IAAFT (Iterative Amplitude-Adjusted Fourier Transform) phase randomization
    preserves amplitude distribution and power spectrum while destroying
    temporal structure. Establishes F_null_95 threshold for superiority check.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# =========================
# INPUT PATHS
# =========================
# Set via environment variables or defaults to outputs/ in repo
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))
RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"

SAMPLES_PER_STRIDE = 200
IAAFT_ITERATIONS = 20
RANDOM_SEED = 42


def load_fidelity_data():
    fid_df = pd.read_csv(UPSTREAM_PATH / "coordination_fidelity.csv")
    return fid_df


def load_stability_data():
    stab_df = pd.read_csv(UPSTREAM_PATH / "stability_summary.csv")
    return stab_df


def load_X_matrix(file_x_path):
    return np.load(str(file_x_path))["X"]


def load_W_matrix(file_npz_path):
    data = np.load(str(file_npz_path))
    return data["W"]


def iaaft_surrogate(x, n_iterations=IAAFT_ITERATIONS, rng=None):
    if rng is None:
        raise ValueError(
            "rng must be provided for deterministic IAAFT surrogates. "
            "Pass rng=np.random.default_rng(seed)."
        )
    n = len(x)
    x_sorted = np.sort(x)
    x_fft = np.fft.rfft(x)
    amplitudes = np.abs(x_fft)
    surrogate = rng.permutation(x).astype(np.float64)
    for _ in range(n_iterations):
        s_fft = np.fft.rfft(surrogate)
        phases = np.angle(s_fft)
        s_fft_new = amplitudes * np.exp(1j * phases)
        surrogate = np.fft.irfft(s_fft_new, n=n)
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = x_sorted[ranks]
    return surrogate


def generate_iaaft_X(X, seed=None):
    rng = np.random.default_rng(seed)
    n_muscles, n_samples = X.shape
    n_strides = n_samples // SAMPLES_PER_STRIDE
    X_surr = np.zeros_like(X)
    for m in range(n_muscles):
        for s in range(n_strides):
            start = s * SAMPLES_PER_STRIDE
            end = start + SAMPLES_PER_STRIDE
            stride_data = X[m, start:end]
            X_surr[m, start:end] = iaaft_surrogate(stride_data, rng=rng)
    return X_surr


def compute_fidelity_with_nnls(X_test, W_train):
    n_muscles, n_samples = X_test.shape
    k = W_train.shape[1]
    H_proj = np.zeros((k, n_samples))
    for t in range(n_samples):
        H_proj[:, t], _ = nnls(W_train, X_test[:, t])
    recon = W_train @ H_proj
    return 1.0 - np.sum((X_test - recon) ** 2) / np.sum(X_test**2)


def _get_stab_row(stab_df, subject, speed_mps):
    """Float-safe lookup for stability_summary rows by subject + speed."""
    d = stab_df[stab_df["subject"].astype(str) == str(subject)].copy()
    if d.empty:
        raise RuntimeError(f"stability_summary.csv: missing subject {subject}")

    spd = pd.to_numeric(d["speed_mps"], errors="coerce").to_numpy(dtype=float)
    mask = np.isclose(spd, float(speed_mps), rtol=0.0, atol=1e-3)

    n = int(mask.sum())
    if n != 1:
        avail = sorted(pd.to_numeric(d["speed_mps"], errors="coerce").dropna().unique().tolist())
        raise RuntimeError(
            f"stability_summary.csv: {subject} expected 1 row at speed_mps={speed_mps}, found {n}. "
            f"Available speeds: {avail}"
        )

    return d.iloc[int(np.where(mask)[0][0])]


def compute_surrogate_baseline(fid_row, stab_df, n_surrogates=1000, seed=RANDOM_SEED):
    subject = fid_row["subject"]
    q_speed = float(fid_row["q_speed_mps"])
    ref_speed = float(fid_row.get("ref_speed_mps", 5.0))

    ref_row = _get_stab_row(stab_df, subject, ref_speed)
    q_row = _get_stab_row(stab_df, subject, q_speed)

    W_ref = load_W_matrix(ref_row["file_npz"])
    X_q = load_X_matrix(q_row["file_x"])

    surrogate_fidelities = []
    for i in range(n_surrogates):
        X_surr = generate_iaaft_X(X_q, seed=seed + i)
        fid = compute_fidelity_with_nnls(X_surr, W_ref)
        surrogate_fidelities.append(fid)

    surrogate_fidelities = np.array(surrogate_fidelities, dtype=float)
    F_null_95 = np.percentile(surrogate_fidelities, 95)

    fid_ref_to_q = float(fid_row["fidelity_ref_to_q"])
    fid_q_to_ref = float(fid_row["fidelity_q_to_ref"])

    return {
        "dataset": "vanhooren",
        "subject": subject,
        "trial": 0,
        "q_speed_mps": q_speed,
        "F_null_mean": surrogate_fidelities.mean(),
        "F_null_std": surrogate_fidelities.std(),
        "F_null_95": float(F_null_95),
        "fidelity_ref_to_q": fid_ref_to_q,
        "fidelity_q_to_ref": fid_q_to_ref,
        "pass_surrogate_ref_to_q": fid_ref_to_q > F_null_95,
        "pass_surrogate_q_to_ref": fid_q_to_ref > F_null_95,
    }


def main(n_surrogates=1000, test_mode=False):
    print("=" * 70)
    print("IAAFT SURROGATE BASELINE")
    print(f"Surrogates per pair: {n_surrogates}")
    print("=" * 70)

    fid_df = load_fidelity_data()
    stab_df = load_stability_data()

    if test_mode:
        fid_df = fid_df.head(1)
        print("TEST MODE: Processing only 1 speed pair")

    print(f"Processing {len(fid_df)} speed pairs...")

    start_time = time.time()
    results = []

    for _, row in fid_df.iterrows():
        t0 = time.time()
        result = compute_surrogate_baseline(row, stab_df, n_surrogates=n_surrogates)
        results.append(result)
        elapsed = time.time() - t0

        print(
            f"  {result['subject']} @ {result['q_speed_mps']} m/s: "
            f"F_null_95={result['F_null_95']:.4f}, pass={result['pass_surrogate_ref_to_q']} ({elapsed:.1f}s)"
        )

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "surrogate_baseline.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    total_time = time.time() - start_time
    print(f"\nSaved: {output_path}")
    print(f"Total time: {total_time/60:.1f} minutes")

    print("\n" + "=" * 70)
    print("SURROGATE BASELINE SUMMARY")
    print("=" * 70)
    print(f"Pairs processed: {len(results_df)}")
    print(f"F_null_95 range: {results_df['F_null_95'].min():.4f} - {results_df['F_null_95'].max():.4f}")
    print(f"Pass rate (ref→q): {int(results_df['pass_surrogate_ref_to_q'].sum())}/{len(results_df)}")
    print(f"Pass rate (q→ref): {int(results_df['pass_surrogate_q_to_ref'].sum())}/{len(results_df)}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode: 1 pair, 10 surrogates")
    parser.add_argument("--n", type=int, default=1000, help="Number of surrogates")
    args = parser.parse_args()

    if args.test:
        main(n_surrogates=10, test_mode=True)
    else:
        main(n_surrogates=args.n)
