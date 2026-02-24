# Technical Implementation Details

This document specifies the implementation details underlying the coordination floor analysis.

**Document scope**: This file contains information that is essential for reproducing results (rerunning the same code on the same data to verify outputs) but not required for understanding or replicating the methods (applying similar procedures to new data). The thesis text provides the latter; this document provides the former.

---

## 1. Deterministic Seeding

All stochastic procedures used in the thesis analysis pipeline are deterministically seeded to enable exact reproducibility. Many scripts derive per-condition seeds via CRC32 hashing of structured identifier strings; a few scripts instead use explicit fixed base seeds (passed via CLI defaults) that are documented below.

### 1.1 NMF Extraction Seeds

For NMF baseline extraction (`02a_nmf_baseline.py`), each random initialisation receives a seed computed as:

```python
seed = zlib.crc32(f"{dataset}|{subject}|{trial}|{speed_tag}|{k}|{init_idx}".encode()) & 0xFFFFFFFF
```

Where:
- `dataset`: "vanhooren" or "santuz"
- `subject`: subject identifier (e.g., "Sub_01")
- `trial`: trial identifier, or "00" if missing/NaN
- `speed_tag`: speed in cm/s, zero-padded to 4 digits (e.g., 5.00 m/s → "0500", 3.33 m/s → "0333")
- `k`: candidate synergy rank
- `init_idx`: random initialisation index (0 to n_init−1)

The bitwise AND with `0xFFFFFFFF` ensures the result fits in a 32-bit unsigned integer suitable for NumPy's random state.

### 1.2 Reliability Bootstrap Seeds

For reliability envelope estimation (`04_compute_reliability.py`), participants are processed in sorted order by subject identifier. Each participant receives a base seed:

```python
base_seed = RANDOM_SEED + participant_index * 1000
```

Where `RANDOM_SEED = 42` and `participant_index` is the 0-based position after sorting.

Within each participant's bootstrap procedure:
- Half-split attempt `a` uses seeds `base_seed + a*2` (half A) and `base_seed + a*2 + 1` (half B)
- NMF initialisations within each half use consecutive integers from the half's seed
- Quantile confidence interval bootstraps use `base_seed + 101` through `base_seed + 104` for the four metrics

### 1.3 Surrogate Generation Seeds

For IAAFT surrogate generation (`05_compute_surrogates.py`), surrogates are generated with sequential seeds starting from a base of 42:

```python
surrogate_seed = 42 + surrogate_index  # surrogate_index = 0 to n_surrogates-1
```

### 1.4 Stride Matching Seeds

For stride count equalisation across speeds (`01_preprocess.py`), a single NumPy generator is initialised with seed `20251214` and used in ascending speed order to draw stride subsamples without replacement.

### 1.5 Sensitivity Analysis Seeds

For sensitivity variants (`09_sensitivity_analyses.py`):
- Reference synergy extraction: seeds `20260103 + i` for `i = 0..4`
- Query synergy extraction: seeds `20260203 + i` for `i = 0..4`

For the `stride_half` variant specifically:
- Reference stride subsampling: seed `20260103`
- Query stride subsampling: seed `20260103 + 1000`

---

## 2. NMF Algorithm Parameters

### 2.1 Primary Extraction (`02a_nmf_baseline.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Algorithm | Multiplicative updates | Frobenius norm objective |
| `n_init` | 10 | Random initialisations per rank |
| `max_iter` | 2000 | Maximum iterations |
| `tol` | 1e-5 | Relative convergence tolerance |
| `eps` | 1e-8 | Numerical stabiliser for division |
| Candidate ranks (vanhooren) | k = 2 to min(7, m−1) | m = number of muscles |
| Candidate ranks (santuz) | k = 2 to min(9, m−1) | |

Convergence is checked every 10 iterations. The run terminates early if relative error change falls below `tol`.

### 2.2 Reliability Bootstrap NMF (`04_compute_reliability.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Solver | Multiplicative updates (`mu`) | Via sklearn NMF |
| Initialisation | NNDSVD-A | |
| `n_init` | 10 | |
| `max_iter` | 500 | Reduced for computational efficiency |

### 2.3 Sensitivity Variant NMF (`09_sensitivity_analyses.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_init` | 5 | Reduced for variant screening |
| Other parameters | As primary | |

**Rank note (sensitivity refits)**: In `09_sensitivity_analyses.py`, the synergy rank is held constant across speeds at each participant's reference-speed rank (k_ref) to keep module pairing square (k×k) for Hungarian matching.

### 2.4 W Normalisation

After extraction, W columns are normalised to sum to one:

```python
scale = W.sum(axis=0) + eps
W_norm = W / scale
H_norm = H * scale.reshape(-1, 1)
```

This ensures W values represent relative muscle contributions and the reconstruction `W @ H` is preserved.

---

## 3. Rank Selection Criteria

Synergy rank is selected as the smallest k satisfying both:

1. **VAF_total ≥ 0.90**: Total variance accounted for across all muscles and timepoints
2. **VAF_min ≥ 0.75**: Minimum single-muscle VAF

VAF is computed as:

```python
VAF_total = 1.0 - (SSE / SST)
VAF_muscle[i] = 1.0 - (SSE_i / SST_i)
VAF_min = min(VAF_muscle)
```

**Fallback rule**: If no rank meets both thresholds, select the smallest k within 0.01 of the maximum observed VAF_total.

---

## 4. Bootstrap Procedures

### 4.1 Block Length Estimation

Block length for moving-block bootstrap is estimated using the Patton–Politis–White automatic procedure (`arch.bootstrap.optimal_block_length`) applied to stride-mean series of the first muscle channel, using the circular block bootstrap estimate.

Bounds are applied:
- Minimum: 3 blocks
- Maximum: floor(n_strides / 8)
- Degeneracy check: if n_strides < 4b, reduce to max(3, n_strides // 4)

### 4.2 Moving-Block Bootstrap Sampling

Indices are drawn at the stride level with circular wrapping:

```python
def mbb_sample_indices(n_strides, b, n_draw, rng):
    n_blocks = ceil(n_draw / b)
    starts = rng.integers(0, n_strides, size=n_blocks)
    idx = []
    for s in starts:
        for j in range(b):
            idx.append((s + j) % n_strides)
    return idx[:n_draw]
```

### 4.3 Reliability Bootstrap Settings

| Parameter | Value |
|-----------|-------|
| `N_BOOT` | 2000 replicates per subject |
| `N_CI` | 2000 replicates for quantile CI |
| Half-split size | floor(n_strides / 2) |
| Max attempts | N_BOOT × 20 (to handle NMF failures) |

### 4.4 Quantile Bound Estimation

Reliability bounds use a two-stage bootstrap:
1. Compute target quantile (10th or 90th percentile) from the half-split distribution
2. Bootstrap the quantile estimate to obtain a one-sided 95% confidence bound

For lower bounds (fidelity, cosine): 5th percentile of bootstrap quantile distribution
For upper bounds (DTW, principal angle): 95th percentile of bootstrap quantile distribution


### 4.5 Within-Speed Reliability Across Speeds (Appendix D.4 / Appendix O)

This analysis computes within-speed (half-split) coordination fidelity independently at each tested speed, to quantify how reliable each speed is on its own (without any cross-speed comparison). This is distinct from the reference-speed reliability envelope in section 4.3/4.4, which uses a moving-block bootstrap and a one-sided confidence bound on the 10th percentile.

Implementation: `scripts/12_within_speed_reliability.py`

**Bootstrap / half-split settings (submission package defaults)**

| Parameter | Value |
|---|---:|
| `n_boot` | 100 replicates per condition |
| Split method | Random permutation of stride indices; two disjoint halves (drop 1 stride if odd) |
| NMF solver | scikit-learn `NMF(solver='cd', init='random')` |
| `n_init` | 20 random starts per half (keep best Frobenius error) |
| `max_iter` | 500 |
| `tol` | 1e-4 |
| `seed_base` | 20260103 |

**Seeding**

- Stride permutation uses `RandomState(seed=r)` for replicate index `r`.
- NMF initialisations use `random_state = seed_base + 100*r + s` for `s = 0..n_init-1`.

**Outputs**

`outputs/within_speed_reliability_all.csv` contains one row per analysed (dataset, subject, trial, speed) with:

- `within_fid_mean`: mean of the half-split fidelity distribution
- `within_fid_q10`: 10th percentile of the distribution (used as the speed-specific lower bound in Appendix D.4)

Increase `--n_boot` if you want tighter quantile estimates, at the cost of runtime.

---

## 5. Surrogate Generation (IAAFT)

Iterative Amplitude-Adjusted Fourier Transform surrogates are generated with:

| Parameter | Value |
|-----------|-------|
| Iterations | 20 per surrogate |
| Application | Per-muscle, per-stride (200 samples) |
| N surrogates | 1000 per participant × speed |

The 95th percentile of surrogate fidelity defines the null threshold.

---

## 6. Signal Processing Parameters

### 6.1 EMG Filtering (`params/base.yaml`)

| Parameter | Value |
|-----------|-------|
| Bandpass | 20–450 Hz |
| Bandpass order | 4th-order Butterworth, zero-phase |
| Envelope low-pass | 6 Hz |
| Envelope order | 4th-order Butterworth, zero-phase |
| Notch frequencies | 50 Hz, 60 Hz |
| Notch Q factor | 30 |
| Notch trigger threshold | ≥5% of 20–450 Hz band power |
| Post-notch residual threshold | ≥2% triggers channel exclusion |

### 6.2 GRF Processing

| Parameter | Value |
|-----------|-------|
| Low-pass filter | 20 Hz, 4th-order Butterworth, zero-phase |
| Contact onset threshold | ≥20 N |
| Contact offset threshold | ≤10 N |
| Minimum strike interval | 0.20 s |

### 6.3 Stride Screening

| Parameter | Value |
|-----------|-------|
| Duration range | 0.50–1.50 s |
| Outlier rule | median ± 5 × (1.4826 × MAD) |
| Baseline drift window | 0.05–0.01 s before contact |
| Drift threshold | robust z-score: z = (x − median)/(1.4826 × MAD); \|z\| > 3 |
| Channel failure threshold | >20% strides flagged |
| Time normalisation | 200 samples per stride |

### 6.4 Signal Clipping Detection

| Parameter | Value |
|-----------|-------|
| Rail definition | max(\|raw\|) per channel |
| Proximity criterion | within 0.05 V of rail |
| Exclusion threshold | ≥0.1% of samples |

---

## 7. Similarity Metrics

### 7.0 Source Synergy Solutions

For cross-speed fidelity computation, the primary multiplicative-update NMF solutions (from `02a_nmf_baseline.py`) are used.

For auxiliary metrics (cosine similarity, DTW distance, principal angles) computed in `06_compute_spatial_temporal_metrics.py`, W and H are re-estimated using scikit-learn NMF with solver `'mu'`, initialisation `'nndsvda'`, and deterministic `random_state`. This re-estimation ensures consistency with the reliability envelope computation, which also uses scikit-learn NMF.

### 7.1 Fidelity (NNLS Reconstruction)

```python
fidelity = 1.0 - (SSE / SST)
```

Where SSE is the squared Frobenius norm of (X − W @ H_nnls) and SST is the squared Frobenius norm of X. H_nnls is solved column-wise via `scipy.optimize.nnls`.

### 7.2 Cosine Similarity

Synergy columns are L2-normalised before computing cosine similarity:

```python
W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + eps)
similarity = W_A_norm.T @ W_B_norm  # k × k matrix
```

Matching uses the Hungarian algorithm to maximise total similarity.

### 7.3 Dynamic Time Warping

DTW distance is computed between stride-averaged activation profiles (H reshaped to n_strides × 200, averaged over strides) using dynamic programming with absolute-difference cost and no warping window constraint.

### 7.4 Principal Angles

Computed via Björck–Golub algorithm:
1. QR decomposition of each W matrix to obtain orthonormal bases
2. SVD of Q_A.T @ Q_B
3. Principal angles = arccos(singular values)

The maximum principal angle (in degrees) is reported.

---

## 8. Floor Detection

### 8.1 Primary Criterion

A speed passes if both directional fidelities meet the participant's reliability bound:

```python
pass = (F_ref_to_query >= L_ref_fidelity) and (F_query_to_ref >= L_ref_fidelity)
```

### 8.2 Seven-Gate Criterion

All seven gates must pass:

| Gate | Metric | Criterion |
|------|--------|-----------|
| G1 | Fidelity (ref→query) | ≥ L_ref_fidelity |
| G2 | Fidelity (query→ref) | ≥ L_ref_fidelity |
| G3 | Surrogate (ref→query) | > surrogate_95th |
| G4 | Surrogate (query→ref) | > surrogate_95th |
| G5 | Cosine similarity | ≥ L_ref_cosine_median |
| G6 | DTW distance | ≤ U_ref_dtw_mean |
| G7 | Principal angle | ≤ U_ref_principal_angle_max_deg |

### 8.3 Sequential Assignment

Floor assignment proceeds from fast to slow speeds:
1. Start at reference speed (5.0 m/s)
2. Test each slower speed in order
3. Floor = last speed to pass before first failure
4. If first transition fails, floor = reference speed

---

## 9. Changepoint Detection

### 9.1 Segmented Regression

Piecewise-linear (hinge) model:

```
y = a + b·x + c·max(0, bp − x)
```

Breakpoint estimated by grid search over 2000 candidate values spanning the query-speed range, minimising sum of squared errors.

### 9.2 PELT Mean-Shift

Single mean-shift changepoint via PELT algorithm with sum-of-squared-errors cost. With 4 query speeds, this evaluates all 3 possible single-split locations.

---

## 10. File Formats

### 10.1 NMF Output (.npz)

Each `nmf_*.npz` file contains:
- `W`: muscle weights (m × k), float32
- `H`: temporal activations (k × time), float32
- `muscles`: muscle names, object array
- `dataset`, `subject`, `trial`, `speed_mps`: identifiers
- `k`, `n_muscles`, `n_obs`, `n_strides`, `n_points`: dimensions
- `n_init`, `max_iter`, `tol`, `eps`: algorithm parameters
- `seed`, `iters`, `converged`: run metadata
- `vaf_total`, `vaf_min`, `vaf_median`, `vaf_mean`, `sse`: fit quality

### 10.2 Reliability Distribution (.npz)

Each `*_ref_boot.npz` file contains:
- `fid_dist`: fidelity values (2 × N_BOOT array, both directions)
- `cosine_median_dist`: cosine similarity values (N_BOOT array)
- `dtw_mean_dist`: DTW distance values (N_BOOT array)
- `principal_angle_max_deg_dist`: principal angle values (N_BOOT array)
- Metadata: dataset, subject, k, n_strides_ref, block_length_b, N_boot_success

---

## 11. Software Environment

See `environment.yml` for the complete conda environment specification. 

**Thesis results were generated under Python 3.12.** The codebase targets Python 3.10+ for broader compatibility, but exact numerical reproducibility requires Python 3.12.

Key dependencies:

- Python 3.10+
- NumPy, SciPy, pandas
- scikit-learn (NMF)
- arch (block length estimation)

---

## 12. Verification

### Folder integrity (SHA-256)

Generate a manifest for the current folder state and verify every file against it:

```bash
# Generate manifest (CSV)
python scripts/create_master_manifest.py --root . --output manifest_sha256.csv

# Verify all files
python scripts/verify_checksums.py --manifest manifest_sha256.csv --base-dir .
```

