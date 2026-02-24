"""
Script: 10_santuz_validation.py
Purpose: External validation using Santuz dataset within-speed reliability.

Inputs:
    - Santuz EMG data files (data/santuz/*.dat)
    - outputs/reliability_envelope.csv (for comparison bounds)

Outputs:
    - outputs/santuz_reliability.csv
    - outputs/santuz_failed_subjects.csv

Validates that van Hooren reliability bounds are consistent with
within-speed reliability observed in the larger Santuz sample.
Uses the same blocked half-split bootstrap procedure.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

warnings.filterwarnings('ignore')

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

# Output files
RESULTS_FILE = OUTPUT_DIR / "santuz_reliability.csv"
FAILED_FILE = OUTPUT_DIR / "santuz_failed_subjects.csv"

# Parameters (reduced for validation purpose - see Methods)
N_BOOT = 50  # Number of bootstrap replicates (sufficient for population-level validation)
K_SYNERGIES = 5  # Fixed rank for comparability
N_NMF_STARTS = 10  # Random starts for NMF (standard in literature)
SAMPLES_PER_STRIDE = 200  # Time points per stride (from preprocessing)
MIN_STRIDES = 20  # Minimum strides required
BASE_SEED0 = 20260103  # Santuz deterministic base seed
SEED_STRIDE = 1000   # seed spacing per participant


def load_emg_data(filepath):
    """Load EMG data from .dat file."""
    df = pd.read_csv(filepath, sep='\t')
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    X = df.values.T
    return X


def segment_into_strides(X, samples_per_stride=200):
    """Segment concatenated EMG into strides."""
    n_muscles, n_total = X.shape
    n_strides = n_total // samples_per_stride
    if n_strides == 0:
        return None
    X_trimmed = X[:, :n_strides * samples_per_stride]
    X_strided = X_trimmed.reshape(n_muscles, n_strides, samples_per_stride)
    return X_strided


def nmf_frobenius(X, k, n_starts=10, max_iter=500, tol=1e-4, seed_base=None):
    """Run NMF with Frobenius norm objective, multiple random starts.

    Deterministic seeding (when seed_base is not None):
        start s uses seed = seed_base + s
    """
    m, n = X.shape
    best_W, best_H, best_err = None, None, np.inf

    for s in range(n_starts):
        # Use an isolated RNG per start for exact reproducibility.
        # If seed_base is None, fall back to a fixed seed (0) so that
        # results are still reproducible by default.
        sb = 0 if seed_base is None else int(seed_base)
        rng_init = np.random.RandomState(sb + int(s))
        W = rng_init.rand(m, k) + 0.01
        H = rng_init.rand(k, n) + 0.01

        prev_err = np.inf
        for iteration in range(max_iter):
            WtW = W.T @ W + 1e-10 * np.eye(k)
            WtX = W.T @ X
            H = np.maximum(H * (WtX / (WtW @ H + 1e-10)), 1e-10)

            HHt = H @ H.T + 1e-10 * np.eye(k)
            XHt = X @ H.T
            W = np.maximum(W * (XHt / (W @ HHt + 1e-10)), 1e-10)

            err = np.linalg.norm(X - W @ H, 'fro')
            if iteration > 0 and abs(prev_err - err) / (prev_err + 1e-10) < tol:
                break
            prev_err = err

        if err < best_err:
            best_err = err
            best_W = W.copy()
            best_H = H.copy()

    return best_W, best_H



def cross_reconstruct_fidelity(X_test, W_train):
    """Compute fidelity: how well W_train reconstructs X_test."""
    m, n = X_test.shape
    k = W_train.shape[1]

    H_opt = np.zeros((k, n))
    for j in range(n):
        H_opt[:, j], _ = nnls(W_train, X_test[:, j])

    X_recon = W_train @ H_opt
    ss_res = np.sum((X_test - X_recon) ** 2)
    ss_tot = np.sum(X_test ** 2)
    fidelity = 1 - ss_res / (ss_tot + 1e-10)

    return fidelity


def compute_block_length(n_strides, min_b=3, max_b_frac=8):
    """Simple block length selection bounded to [min_b, n//max_b_frac]."""
    max_b = max(min_b, n_strides // max_b_frac)
    b = max(min_b, min(max_b, int(np.sqrt(n_strides))))
    return b


def blocked_half_split_reliability(X_strided, k, n_boot=50, n_nmf_starts=10, base_seed=None):
    """Compute within-speed reliability using blocked half-split bootstrap.

    Deterministic seeding (when base_seed is not None):
      - All bootstrap resampling uses a local RandomState seeded with base_seed.
      - NMF start s uses seed = base_seed + s (handled inside nmf_frobenius).
    """
    n_muscles, n_strides, T = X_strided.shape

    # Local RNG for bootstrap sampling (isolated from global state)
    # Deterministic by default: if base_seed is None, fall back to 0.
    rng = np.random.RandomState(int(base_seed) if base_seed is not None else 0)

    b = compute_block_length(n_strides)
    half_n = n_strides // 2

    fidelities_fwd = []
    fidelities_rev = []

    for boot_idx in range(n_boot):
        n_blocks_needed = (half_n // b) + 1

        block_starts_A = rng.randint(0, n_strides - b + 1, size=n_blocks_needed)
        block_starts_B = rng.randint(0, n_strides - b + 1, size=n_blocks_needed)

        strides_A = []
        for start in block_starts_A:
            strides_A.extend(range(start, min(start + b, n_strides)))
            if len(strides_A) >= half_n:
                break
        strides_A = strides_A[:half_n]

        strides_B = []
        for start in block_starts_B:
            strides_B.extend(range(start, min(start + b, n_strides)))
            if len(strides_B) >= half_n:
                break
        strides_B = strides_B[:half_n]

        if len(strides_A) < half_n // 2 or len(strides_B) < half_n // 2:
            continue

        X_A = X_strided[:, strides_A, :].reshape(n_muscles, -1)
        X_B = X_strided[:, strides_B, :].reshape(n_muscles, -1)

        try:
            W_A, H_A = nmf_frobenius(X_A, k, n_starts=n_nmf_starts, seed_base=base_seed)
            W_B, H_B = nmf_frobenius(X_B, k, n_starts=n_nmf_starts, seed_base=base_seed)
        except Exception:
            continue

        fid_A_to_B = cross_reconstruct_fidelity(X_B, W_A)
        fid_B_to_A = cross_reconstruct_fidelity(X_A, W_B)

        if np.isfinite(fid_A_to_B) and np.isfinite(fid_B_to_A):
            fidelities_fwd.append(fid_A_to_B)
            fidelities_rev.append(fid_B_to_A)

    if len(fidelities_fwd) < 10:
        return None

    all_fidelities = np.array(fidelities_fwd + fidelities_rev)

    fid_mean = np.mean(all_fidelities)
    fid_std = np.std(all_fidelities)
    fid_min = np.min(all_fidelities)
    fid_max = np.max(all_fidelities)
    fid_q10 = np.percentile(all_fidelities, 10)
    fid_q50 = np.percentile(all_fidelities, 50)

    boot_q10s = []
    for _ in range(200):
        resample = rng.choice(all_fidelities, size=len(all_fidelities), replace=True)
        boot_q10s.append(np.percentile(resample, 10))
    se_q10 = np.std(boot_q10s)
    L_ref = fid_q10 - 1.645 * se_q10

    return {
        'n_boot_success': len(fidelities_fwd),
        'block_length_b': b,
        'half_n': half_n,
        'fid_mean': fid_mean,
        'fid_std': fid_std,
        'fid_min': fid_min,
        'fid_max': fid_max,
        'fid_q10': fid_q10,
        'fid_q50': fid_q50,
        'L_ref_fidelity': L_ref
    }



def load_completed_subjects():
    """Load list of already-completed subjects from existing results file."""
    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
        return set(df['subject'].tolist())
    return set()


def append_result(result):
    """Append a single result to the CSV file."""
    df = pd.DataFrame([result])
    if RESULTS_FILE.exists():
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)


def append_failed(failed):
    """Append a single failed subject to the failed CSV file."""
    df = pd.DataFrame([failed])
    if FAILED_FILE.exists():
        df.to_csv(FAILED_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(FAILED_FILE, index=False)


def main() -> int:
    print("=" * 70)
    print("SANTUZ EXTERNAL VALIDATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Santuz registry
    registry_path = UPSTREAM_PATH / "unified_registry.csv"
    if not registry_path.exists():
        # Try alternative location
        registry_path = DATA_PATH / "santuz_registry.csv"

    if not registry_path.exists():
        print(f"ERROR: Cannot find Santuz registry at {registry_path}")
        print("Please ensure the Santuz data registry is available.")
        return 1

    print(f"\n[1/3] Loading Santuz registry from {registry_path}...")
    registry = pd.read_csv(registry_path)
    santuz_rows = registry[registry['dataset'] == 'santuz'].copy()
    print(f"  Found {len(santuz_rows)} Santuz trials from {santuz_rows['subject'].nunique()} subjects")

    # Select the first available trial per subject
    if "trial" in santuz_rows.columns:
        santuz_rows = santuz_rows.copy()
        santuz_rows["_trial_num"] = pd.to_numeric(santuz_rows["trial"], errors="coerce")
        santuz_rows = santuz_rows.sort_values(["subject", "_trial_num"], ascending=[True, True])
        santuz_subjects = santuz_rows.groupby("subject").first().reset_index()
        santuz_subjects = santuz_subjects.drop(columns=["_trial_num"], errors="ignore")
    else:
        santuz_subjects = santuz_rows.groupby("subject").first().reset_index()

    santuz_subjects = santuz_subjects.sort_values("subject").reset_index(drop=True)
    print(f"  Total subjects to process: {len(santuz_subjects)}")

    # Check for already completed subjects
    completed = load_completed_subjects()
    if completed:
        print(f"  Found {len(completed)} already completed subjects - resuming...")

    # Process each subject
    print("\n[2/3] Computing within-speed reliability for each subject...")
    n_total = len(santuz_subjects)
    n_processed = 0
    n_skipped = len(completed)

    for idx, row in santuz_subjects.iterrows():
        subject = row['subject']
        filepath = row.get('file_matched', row.get('filepath', None))
        speed = row.get('speed_mps', np.nan)

        # Skip if already completed
        if subject in completed:
            continue

        n_processed += 1
        current_num = n_skipped + n_processed

        # Progress every subject
        print(f"  [{current_num}/{n_total}] {subject}...", end=" ", flush=True)
        start_time = datetime.now()

        if filepath is None or not Path(filepath).exists():
            append_failed({'subject': subject, 'reason': 'file_not_found'})
            print("SKIP (file not found)")
            continue

        try:
            X = load_emg_data(filepath)
            X_strided = segment_into_strides(X, samples_per_stride=SAMPLES_PER_STRIDE)

            if X_strided is None or X_strided.shape[1] < MIN_STRIDES:
                append_failed({'subject': subject, 'reason': 'insufficient_strides'})
                print("SKIP (insufficient strides)")
                continue

            base_seed = BASE_SEED0 + SEED_STRIDE * int(idx)
            rel = blocked_half_split_reliability(X_strided, K_SYNERGIES, n_boot=N_BOOT, n_nmf_starts=N_NMF_STARTS, base_seed=base_seed)

            if rel is None:
                append_failed({'subject': subject, 'reason': 'reliability_computation_failed'})
                print("SKIP (computation failed)")
                continue

            result = {
                'subject': subject,
                'dataset': 'santuz',
                'speed_mps': speed,
                'n_muscles': X.shape[0],
                'n_strides': X_strided.shape[1],
                **rel
            }
            append_result(result)

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"OK ({elapsed:.1f}s, L_ref={rel['L_ref_fidelity']:.3f})")

        except Exception as e:
            append_failed({'subject': subject, 'reason': str(e)[:50]})
            print(f"ERROR: {str(e)[:30]}")
            continue

    # Final summary
    print("\n[3/3] Generating summary...")

    if RESULTS_FILE.exists():
        results_df = pd.read_csv(RESULTS_FILE)
        print(f"  Total completed: {len(results_df)} subjects")

        # Load van Hooren for comparison
        vh_reliability_path = UPSTREAM_PATH / "reliability_envelope.csv"
        if vh_reliability_path.exists():
            vh_reliability = pd.read_csv(vh_reliability_path)

            print("\n" + "=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)

            print(f"\nSantuz within-speed reliability (N = {len(results_df)}):")
            print(f"  L_ref_fidelity: mean = {results_df['L_ref_fidelity'].mean():.3f}, SD = {results_df['L_ref_fidelity'].std():.3f}")
            print(f"                  min = {results_df['L_ref_fidelity'].min():.3f}, max = {results_df['L_ref_fidelity'].max():.3f}")
            print(f"  fid_q10 (10th %ile): mean = {results_df['fid_q10'].mean():.3f}, SD = {results_df['fid_q10'].std():.3f}")
            print(f"  fid_mean: mean = {results_df['fid_mean'].mean():.3f}, SD = {results_df['fid_mean'].std():.3f}")

            print(f"\nVan Hooren reference-speed reliability (N = {len(vh_reliability)}):")
            print(f"  L_ref_fidelity: mean = {vh_reliability['L_ref_fidelity'].mean():.3f}, SD = {vh_reliability['L_ref_fidelity'].std():.3f}")
            print(f"                  min = {vh_reliability['L_ref_fidelity'].min():.3f}, max = {vh_reliability['L_ref_fidelity'].max():.3f}")

            # Statistical comparison
            from scipy import stats
            if len(results_df) > 5:
                t_stat, p_val = stats.ttest_ind(results_df['L_ref_fidelity'].dropna(),
                                                  vh_reliability['L_ref_fidelity'].dropna())
                print(f"\nIndependent t-test (Santuz vs van Hooren L_ref):")
                print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")

                pooled_std = np.sqrt((results_df['L_ref_fidelity'].std()**2 + vh_reliability['L_ref_fidelity'].std()**2) / 2)
                cohens_d = (results_df['L_ref_fidelity'].mean() - vh_reliability['L_ref_fidelity'].mean()) / pooled_std
                print(f"  Cohen's d = {cohens_d:.3f}")

            # Save comparison
            comparison = {
                'dataset': ['santuz', 'vanhooren'],
                'n_subjects': [len(results_df), len(vh_reliability)],
                'L_ref_mean': [results_df['L_ref_fidelity'].mean(), vh_reliability['L_ref_fidelity'].mean()],
                'L_ref_sd': [results_df['L_ref_fidelity'].std(), vh_reliability['L_ref_fidelity'].std()],
                'L_ref_min': [results_df['L_ref_fidelity'].min(), vh_reliability['L_ref_fidelity'].min()],
                'L_ref_max': [results_df['L_ref_fidelity'].max(), vh_reliability['L_ref_fidelity'].max()]
            }
            comparison_df = pd.DataFrame(comparison)
            comparison_df.to_csv(OUTPUT_DIR / "reliability_comparison.csv", index=False)
            print(f"\n  Saved: reliability_comparison.csv")
        else:
            print(f"  Note: Van Hooren reliability file not found at {vh_reliability_path}")

    print(f"\nEnd time: {datetime.now().isoformat()}")
    print("=" * 70)
    print("SANTUZ VALIDATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
