"""scripts/11_reorganization_profile.py

Reorganization profile across adjacent speed transitions (Van Hooren primary).

This script implements the reorganization-profile analysis:

For each participant and each adjacent speed transition:

    within_avg_mean = 0.5*(within_fid_mean(speed_fast) + within_fid_mean(speed_slow))
    threshold_q10   = 0.5*(within_fid_q10(speed_fast) + within_fid_q10(speed_slow))
    cross_avg       = 0.5*(fidelity_fast->slow + fidelity_slow->fast)
    drop            = within_avg_mean - cross_avg
    drop_beyond     = max(0, threshold_q10 - cross_avg)


Required inputs
---------------
- outputs/within_speed_reliability_all.csv
  Produced by: scripts/12_within_speed_reliability.py
  Must contain: dataset, subject, trial, speed_mps, within_fid_mean, within_fid_q10

- outputs/full_pairwise_fidelity.csv
  Produced by: scripts/03b_full_pairwise_fidelity.py
  Must contain: dataset, subject, trial, speed_a_mps, speed_b_mps, fidelity_a_to_b

Statistics
----------
- Friedman test across the four transition drops
- Six paired t-tests between transitions, Holm–Bonferroni corrected
- Effect sizes as Cohen's d_z with 95% CI via noncentral-t inversion

Outputs
-------
- outputs/subject_transition_drops.csv
- outputs/subject_max_reorganization.csv
- outputs/group_reorganization_summary.csv
- outputs/transition_statistics.csv
- outputs/reorganization_summary.txt
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import nctdtrinc

# =========================
# INPUT PATHS
# =========================
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))

# =========================
# RELEASE-LOCAL OUTPUTS
# =========================
RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"

# Adjacent speed transitions (fast -> slow)
ADJACENT_PAIRS = [
    (5.0, 4.0),
    (4.0, 3.33),
    (3.33, 3.0),
    (3.0, 2.78),
]


def mps_to_min_per_km(mps: float) -> str:
    """Convert speed (m/s) to pace (min:sec per km), rounding to nearest second."""
    if not np.isfinite(mps) or mps <= 0:
        return "N/A"
    sec_per_km = 1000.0 / float(mps)
    sec = int(np.round(sec_per_km))
    minutes = sec // 60
    seconds = sec % 60
    return f"{minutes}:{seconds:02d}"


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load within-speed reliability and full pairwise fidelity."""
    within_path = UPSTREAM_PATH / "within_speed_reliability_all.csv"
    pairwise_path = UPSTREAM_PATH / "full_pairwise_fidelity.csv"

    if not within_path.exists():
        raise FileNotFoundError(
            f"Missing {within_path}. Run scripts/12_within_speed_reliability.py first."
        )
    if not pairwise_path.exists():
        raise FileNotFoundError(
            f"Missing {pairwise_path}. Run scripts/03b_full_pairwise_fidelity.py first."
        )

    within = pd.read_csv(within_path)
    pairwise = pd.read_csv(pairwise_path)

    _require_columns(within, ["dataset", "subject", "trial", "speed_mps", "within_fid_mean", "within_fid_q10"], within_path.name)
    _require_columns(
        pairwise,
        ["dataset", "subject", "trial", "speed_a_mps", "speed_b_mps", "fidelity_a_to_b"],
        pairwise_path.name,
    )

    # Canonical dtypes
    within = within.copy()
    within["speed_mps"] = pd.to_numeric(within["speed_mps"], errors="coerce").astype(float)
    within["trial"] = pd.to_numeric(within["trial"], errors="coerce").astype(float)
    within["within_fid_mean"] = pd.to_numeric(within["within_fid_mean"], errors="coerce").astype(float)
    within["within_fid_q10"] = pd.to_numeric(within["within_fid_q10"], errors="coerce").astype(float)

    pairwise = pairwise.copy()
    pairwise["speed_a_mps"] = pd.to_numeric(pairwise["speed_a_mps"], errors="coerce").astype(float)
    pairwise["speed_b_mps"] = pd.to_numeric(pairwise["speed_b_mps"], errors="coerce").astype(float)
    pairwise["trial"] = pd.to_numeric(pairwise["trial"], errors="coerce").astype(float)
    pairwise["fidelity_a_to_b"] = pd.to_numeric(pairwise["fidelity_a_to_b"], errors="coerce").astype(float)

    return within, pairwise


def _lookup_single(df: pd.DataFrame, query: dict, value_col: str) -> float | None:
    """Return a single scalar if exactly one row matches; else None."""
    mask = np.ones(len(df), dtype=bool)
    for k, v in query.items():
        if isinstance(v, float):
            mask &= np.isclose(df[k].values.astype(float), float(v), atol=1e-6)
        else:
            mask &= df[k].astype(str).values == str(v)

    sub = df.loc[mask, value_col]
    if len(sub) != 1:
        return None
    val = float(sub.iloc[0])
    if not np.isfinite(val):
        return None
    return val


def compute_subject_transition_drops(
    within_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    *,
    dataset: str = "vanhooren",
) -> pd.DataFrame:
    """Compute per-subject reorganization metrics for adjacent transitions.

    Definitions

    -----------------------------
    For each adjacent speed transition (fast -> slow):

        within_avg_mean  = 0.5*(within_fid_mean_fast + within_fid_mean_slow)
        threshold_q10    = 0.5*(within_fid_q10_fast  + within_fid_q10_slow)
        cross_avg        = 0.5*(fidelity_fast->slow + fidelity_slow->fast)

        drop             = within_avg_mean - cross_avg
        drop_beyond      = max(0, threshold_q10 - cross_avg)

    Notes
    -----
    - Group statistics (Friedman/t-tests) use `drop` (the mean-drop metric).
    - The per-participant "maximum-change transition" uses `drop_beyond`.
    """
    within = within_df[within_df["dataset"].astype(str) == dataset].copy()
    pairwise = pairwise_df[pairwise_df["dataset"].astype(str) == dataset].copy()

    # Only keep rows with finite within-speed mean and q10 bounds
    within = within[np.isfinite(within["within_fid_mean"]) & np.isfinite(within["within_fid_q10"])].copy()

    group_cols = ["dataset", "subject", "trial"]
    rows: list[dict] = []

    transition_order = {f"{sp_fast}->{sp_slow}": i for i, (sp_fast, sp_slow) in enumerate(ADJACENT_PAIRS)}

    for (ds, subj, trial), wgrp in within.groupby(group_cols, sort=True):
        pgrp = pairwise[(pairwise["dataset"] == ds) & (pairwise["subject"] == subj) & (pairwise["trial"] == trial)]
        if pgrp.empty:
            continue

        for sp_fast, sp_slow in ADJACENT_PAIRS:
            # Within-speed reliability (mean and q10) at the two speeds
            within_fast = _lookup_single(wgrp, {"speed_mps": sp_fast}, "within_fid_mean")
            within_slow = _lookup_single(wgrp, {"speed_mps": sp_slow}, "within_fid_mean")
            q10_fast = _lookup_single(wgrp, {"speed_mps": sp_fast}, "within_fid_q10")
            q10_slow = _lookup_single(wgrp, {"speed_mps": sp_slow}, "within_fid_q10")
            if within_fast is None or within_slow is None or q10_fast is None or q10_slow is None:
                continue

            # Directional cross-speed fidelities
            f_fast_to_slow = _lookup_single(
                pgrp,
                {"speed_a_mps": sp_fast, "speed_b_mps": sp_slow},
                "fidelity_a_to_b",
            )
            f_slow_to_fast = _lookup_single(
                pgrp,
                {"speed_a_mps": sp_slow, "speed_b_mps": sp_fast},
                "fidelity_a_to_b",
            )
            if f_fast_to_slow is None or f_slow_to_fast is None:
                continue

            within_avg_mean = 0.5 * (within_fast + within_slow)
            threshold_q10 = 0.5 * (q10_fast + q10_slow)
            cross_avg = 0.5 * (f_fast_to_slow + f_slow_to_fast)

            drop = within_avg_mean - cross_avg
            drop_beyond = max(0.0, threshold_q10 - cross_avg)

            trans = f"{sp_fast}->{sp_slow}"
            rows.append(
                {
                    "dataset": ds,
                    "subject": subj,
                    "trial": float(trial),
                    "transition": trans,
                    "transition_order": int(transition_order.get(trans, 999)),
                    "speed_fast_mps": float(sp_fast),
                    "speed_slow_mps": float(sp_slow),
                    "pace_fast_min_per_km": mps_to_min_per_km(sp_fast),
                    "pace_slow_min_per_km": mps_to_min_per_km(sp_slow),
                    # Within-speed reliability (means)
                    "within_fidelity_fast": float(within_fast),
                    "within_fidelity_slow": float(within_slow),
                    "within_fidelity_avg": float(within_avg_mean),
                    # Within-speed reliability (q10 threshold components)
                    "within_q10_fast": float(q10_fast),
                    "within_q10_slow": float(q10_slow),
                    "within_q10_threshold": float(threshold_q10),
                    # Cross-speed fidelity (directional + average)
                    "cross_fidelity_fast_to_slow": float(f_fast_to_slow),
                    "cross_fidelity_slow_to_fast": float(f_slow_to_fast),
                    "cross_fidelity_avg": float(cross_avg),
                    # Reorganization metrics
                    "drop": float(drop),
                    "drop_beyond": float(drop_beyond),
                    "pass_within_q10": bool(cross_avg >= threshold_q10),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dataset", "subject", "trial", "transition_order"], kind="mergesort").reset_index(drop=True)
        out = out.drop(columns=["transition_order"])
    return out


def identify_max_reorganization(drops_df: pd.DataFrame) -> pd.DataFrame:
    """Identify each participant's maximum-change transition (largest drop_beyond).

    The maximum-change transition is defined as the adjacent transition with the
    largest `drop_beyond` value. This differs from the transition
    with the largest mean-drop (`drop`).
    """
    if drops_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "subject",
                "trial",
                "max_reorg_transition",
                "max_reorg_speed_fast",
                "max_reorg_speed_slow",
                "max_drop_beyond",
                "cross_fidelity_avg_at_max",
                "within_q10_threshold_at_max",
                "drop_mean_at_max",
            ]
        )

    transition_order = {f"{sp_fast}->{sp_slow}": i for i, (sp_fast, sp_slow) in enumerate(ADJACENT_PAIRS)}

    rows = []
    group_cols = ["dataset", "subject", "trial"]
    for (ds, subj, trial), grp in drops_df.groupby(group_cols, sort=True):
        if grp.empty:
            continue

        g = grp.copy()
        g["transition_order"] = g["transition"].map(lambda t: int(transition_order.get(str(t), 999)))

        # Max drop_beyond; tie-break by earliest transition order
        max_val = float(pd.to_numeric(g["drop_beyond"], errors="coerce").max())
        cand = g[np.isclose(g["drop_beyond"].astype(float).values, max_val, rtol=0.0, atol=1e-12)]
        cand = cand.sort_values(["transition_order"], kind="mergesort").iloc[0]

        rows.append(
            {
                "dataset": ds,
                "subject": subj,
                "trial": float(trial),
                "max_reorg_transition": cand["transition"],
                "max_reorg_speed_fast": float(cand["speed_fast_mps"]),
                "max_reorg_speed_slow": float(cand["speed_slow_mps"]),
                "max_drop_beyond": float(cand["drop_beyond"]),
                "cross_fidelity_avg_at_max": float(cand["cross_fidelity_avg"]),
                "within_q10_threshold_at_max": float(cand["within_q10_threshold"]),
                "drop_mean_at_max": float(cand["drop"]),
            }
        )

    return pd.DataFrame(rows)


def compute_group_summary(drops_df: pd.DataFrame, max_df: pd.DataFrame) -> pd.DataFrame:
    """Group-level summary (matches the Results table structure)."""
    if drops_df.empty:
        return pd.DataFrame()

    # Mean/SD per transition
    g = (
        drops_df.groupby("transition", sort=False)["drop"]
        .agg([("drop_mean", "mean"), ("drop_sd", "std"), ("n", "count")])
        .reset_index()
    )

    # Counts of max-transition per subject
    max_counts = max_df["max_reorg_transition"].value_counts().to_dict() if not max_df.empty else {}
    n_subjects = int(max_df.shape[0]) if not max_df.empty else int(drops_df[["dataset", "subject", "trial"]].drop_duplicates().shape[0])

    g["n_subjects_max_here"] = g["transition"].map(lambda t: int(max_counts.get(t, 0)))
    g["pct_subjects_max_here"] = (g["n_subjects_max_here"] / max(n_subjects, 1) * 100.0)

    # Transition metadata
    meta = []
    for sp_fast, sp_slow in ADJACENT_PAIRS:
        meta.append(
            {
                "transition": f"{sp_fast}->{sp_slow}",
                "speed_fast_mps": float(sp_fast),
                "speed_slow_mps": float(sp_slow),
                "pace_fast_min_per_km": mps_to_min_per_km(sp_fast),
                "pace_slow_min_per_km": mps_to_min_per_km(sp_slow),
            }
        )
    meta_df = pd.DataFrame(meta)

    out = meta_df.merge(g, on="transition", how="left")

    # Ratio relative to first transition mean
    base_trans = f"{ADJACENT_PAIRS[0][0]}->{ADJACENT_PAIRS[0][1]}"
    base_mean = float(out.loc[out["transition"] == base_trans, "drop_mean"].iloc[0])
    if np.isfinite(base_mean) and abs(base_mean) > 1e-12:
        out["ratio_vs_5_to_4"] = out["drop_mean"] / base_mean
    else:
        out["ratio_vs_5_to_4"] = np.nan

    # Deterministic order
    order = [f"{a}->{b}" for a, b in ADJACENT_PAIRS]
    out["_order"] = out["transition"].apply(lambda t: order.index(t) if t in order else 999)
    out = out.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)

    # Round display columns
    for c in ["drop_mean", "drop_sd", "pct_subjects_max_here", "ratio_vs_5_to_4"]:
        if c in out.columns:
            out[c] = out[c].astype(float)

    return out


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p

    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m, dtype=float)

    for i in range(m):
        adj_sorted[i] = (m - i) * p_sorted[i]

    # Enforce monotonicity (nondecreasing)
    for i in range(1, m):
        if adj_sorted[i] < adj_sorted[i - 1]:
            adj_sorted[i] = adj_sorted[i - 1]

    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj


def dz_and_ci(diff: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Cohen's d_z for paired samples + 95% CI via noncentral-t inversion.

    CI method:
      - t = mean(diff) / (sd(diff)/sqrt(n))
      - noncentrality CI bounds: delta_L, delta_U such that
          P(T<=t | delta_L) = 1-alpha/2
          P(T<=t | delta_U) = alpha/2
      - convert delta bounds to d_z by dividing by sqrt(n)
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n < 2:
        return np.nan, np.nan, np.nan

    sd = float(np.std(diff, ddof=1))
    if sd <= 0:
        return np.nan, np.nan, np.nan

    dz = float(np.mean(diff) / sd)
    t_obs = float(dz * np.sqrt(n))
    df = n - 1

    # Invert noncentral t to get CI on delta (noncentrality parameter)
    delta_lo = float(nctdtrinc(df, 1.0 - alpha / 2.0, t_obs))
    delta_hi = float(nctdtrinc(df, alpha / 2.0, t_obs))
    dz_lo = float(delta_lo / np.sqrt(n))
    dz_hi = float(delta_hi / np.sqrt(n))

    lo = min(dz_lo, dz_hi)
    hi = max(dz_lo, dz_hi)
    return dz, lo, hi


def compute_transition_statistics(drops_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Friedman + Holm-corrected paired t-tests between transitions."""
    if drops_df.empty:
        return pd.DataFrame(), {"friedman_chi2": np.nan, "friedman_p": np.nan}

    # Wide matrix: subjects x transitions
    subj_cols = ["dataset", "subject", "trial"]
    wide = drops_df.pivot_table(index=subj_cols, columns="transition", values="drop", aggfunc="mean")

    # Keep only complete cases across all transitions
    transitions = [f"{a}->{b}" for a, b in ADJACENT_PAIRS]
    wide = wide.reindex(columns=transitions)
    wide_complete = wide.dropna(axis=0, how="any")

    n_subjects = int(wide_complete.shape[0])

    if n_subjects < 2:
        friedman_stat, friedman_p = np.nan, np.nan
    else:
        cols = [wide_complete[c].values for c in transitions]
        friedman_stat, friedman_p = stats.friedmanchisquare(*cols)

    # Pairwise paired t-tests
    results = []
    comps = list(combinations(transitions, 2))
    pvals = []

    for a, b in comps:
        da = wide_complete[a].values
        db = wide_complete[b].values
        diff = da - db
        t_stat, p_val = stats.ttest_rel(da, db)
        dz, dz_lo, dz_hi = dz_and_ci(diff, alpha=0.05)
        pvals.append(float(p_val))

        results.append(
            {
                "comparison": f"{a} vs {b}",
                "n": n_subjects,
                "df": n_subjects - 1,
                "mean_drop_a": float(np.mean(da)),
                "mean_drop_b": float(np.mean(db)),
                "mean_difference": float(np.mean(diff)),
                "t_statistic": float(t_stat),
                "p_uncorrected": float(p_val),
                "dz": float(dz),
                "dz_ci_low": float(dz_lo),
                "dz_ci_high": float(dz_hi),
            }
        )

    pvals = np.asarray(pvals, dtype=float)
    p_holm = holm_adjust(pvals)

    for i in range(len(results)):
        results[i]["p_holm"] = float(p_holm[i])
        results[i]["significant_holm_0p05"] = bool(p_holm[i] < 0.05)

    stats_df = pd.DataFrame(results)
    if not stats_df.empty:
        # Sort by uncorrected p (helpful for inspection)
        stats_df = stats_df.sort_values(["p_uncorrected"], kind="mergesort").reset_index(drop=True)

    return stats_df, {"friedman_chi2": float(friedman_stat), "friedman_p": float(friedman_p), "n": n_subjects}


def write_summary(
    drops_df: pd.DataFrame,
    max_df: pd.DataFrame,
    group_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    friedman: dict,
) -> str:
    """Human-readable summary."""
    if group_df.empty:
        return "No results."

    # Transition with max mean drop (group average)
    idx = group_df["drop_mean"].astype(float).idxmax()
    max_row = group_df.loc[idx]

    lines: list[str] = []
    lines.append("COORDINATION REORGANIZATION PROFILE")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("Definitions")
    lines.append("  within_avg_mean = mean(within_fid_mean at both speeds)")
    lines.append("  cross_avg       = mean(bidirectional cross-speed fidelity)")
    lines.append("  drop            = within_avg_mean - cross_avg")
    lines.append("  threshold_q10   = mean(within_fid_q10 at both speeds)")
    lines.append("  drop_beyond     = max(0, threshold_q10 - cross_avg)")
    lines.append("  (Maximum-change transition uses drop_beyond.)")
    lines.append("")

    lines.append("Key finding (mean-drop)")
    lines.append(
        f"  Largest mean drop at: {max_row['transition']} "
        f"(mean={float(max_row['drop_mean']):.4f}, SD={float(max_row['drop_sd']):.4f})"
    )
    lines.append(
        f"  Subjects whose max-change transition is here: {int(max_row['n_subjects_max_here'])}/{len(max_df)} "
        f"({float(max_row['pct_subjects_max_here']):.1f}%)"
    )
    lines.append("")

    lines.append("Transition summary")
    for _, r in group_df.iterrows():
        lines.append(
            f"  {r['transition']} ({r['pace_fast_min_per_km']}→{r['pace_slow_min_per_km']} min/km): "
            f"drop_mean={float(r['drop_mean']):.4f}, SD={float(r['drop_sd']):.4f}, "
            f"ratio={float(r['ratio_vs_5_to_4']):.2f}, "
            f"n_max_change={int(r['n_subjects_max_here'])} ({float(r['pct_subjects_max_here']):.1f}%)"
        )

    lines.append("")
    lines.append("Statistics (mean-drop)")
    lines.append(
        f"  Friedman test across transitions: chi2={friedman['friedman_chi2']:.3f}, "
        f"p={friedman['friedman_p']:.6g}, n={friedman.get('n', 'NA')}"
    )
    lines.append("  Paired t-tests (Holm-corrected):")
    if stats_df.empty:
        lines.append("    (none)")
    else:
        for _, r in stats_df.iterrows():
            sig = "*" if r["significant_holm_0p05"] else ""
            lines.append(
                f"    {r['comparison']}: t({int(r['df'])})={r['t_statistic']:.3f}, "
                f"p={r['p_uncorrected']:.4f}, p_holm={r['p_holm']:.4f}{sig}, "
                f"dz={r['dz']:.2f} [{r['dz_ci_low']:.2f},{r['dz_ci_high']:.2f}]"
            )

    lines.append("")
    lines.append("Outputs")
    lines.append("  - subject_transition_drops.csv")
    lines.append("  - subject_max_reorganization.csv")
    lines.append("  - group_reorganization_summary.csv")
    lines.append("  - transition_statistics.csv")
    lines.append("  - transition_drop_beyond_matrix.csv")
    lines.append("  - reorganization_summary.txt")

    return "\n".join(lines) + "\n"

def main() -> int:
    print("COORDINATION REORGANIZATION PROFILE")
    print("=" * 60)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    within_df, pairwise_df = load_data()

    # Compute drops
    drops_df = compute_subject_transition_drops(within_df, pairwise_df, dataset="vanhooren")
    if drops_df.empty:
        
        ds = "vanhooren"
        w = within_df[within_df["dataset"].astype(str) == ds].copy()
        p = pairwise_df[pairwise_df["dataset"].astype(str) == ds].copy()

        print("")
        print("DIAGNOSTICS: within_speed_reliability_all.csv")
        print(f"  Rows for {ds}: {len(w)}")
        if len(w):
            finite = w[np.isfinite(w["within_fid_mean"]) & np.isfinite(w["within_fid_q10"]) & (w["n_boot_success"] > 0)]
            print(f"  Rows with finite values and n_boot_success>0: {len(finite)}")
            print("  Counts by speed (all rows):")
            print(w.groupby("speed_mps").size().to_string())
            print("  Counts by speed (finite + success rows):")
            print(finite.groupby("speed_mps").size().to_string() if len(finite) else "  <none>")

        print("")
        print("DIAGNOSTICS: full_pairwise_fidelity.csv")
        print(f"  Rows for {ds}: {len(p)}")
        needed = [(5.0, 4.0), (4.0, 3.33), (3.33, 3.0), (3.0, 2.78)]
        for a, b in needed:
            n_ab = int(((p["speed_a_mps"] == a) & (p["speed_b_mps"] == b)).sum())
            n_ba = int(((p["speed_a_mps"] == b) & (p["speed_b_mps"] == a)).sum())
            print(f"  Pair {a}->{b}: {n_ab} rows; {b}->{a}: {n_ba} rows")

        raise RuntimeError(
            "No transition drops computed. Confirm that within_speed_reliability_all.csv and full_pairwise_fidelity.csv "
            "contain vanhooren rows for speeds 5.0,4.0,3.33,3.0,2.78."
        )

    # Max per subject
    max_df = identify_max_reorganization(drops_df)

    # Group summary
    group_df = compute_group_summary(drops_df, max_df)

    # Stats
    stats_df, friedman = compute_transition_statistics(drops_df)

    
    trans_cols = [f"{a}->{b}" for a, b in ADJACENT_PAIRS]
    mat = drops_df.pivot_table(index=["dataset", "subject", "trial"], columns="transition", values="drop_beyond", aggfunc="first")
    mat = mat.reset_index()
    # Ensure all transitions exist and ordered
    for c in trans_cols:
        if c not in mat.columns:
            mat[c] = np.nan
    mat = mat[["dataset", "subject", "trial"] + trans_cols]
    mat.to_csv(OUTPUT_DIR / "transition_drop_beyond_matrix.csv", index=False)

    # Save outputs
    drops_df.to_csv(OUTPUT_DIR / "subject_transition_drops.csv", index=False)
    max_df.to_csv(OUTPUT_DIR / "subject_max_reorganization.csv", index=False)
    group_df.to_csv(OUTPUT_DIR / "group_reorganization_summary.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "transition_statistics.csv", index=False)

    summary = write_summary(drops_df, max_df, group_df, stats_df, friedman)
    (OUTPUT_DIR / "reorganization_summary.txt").write_text(summary, encoding="utf-8", newline="\n")

    print("\nWrote outputs to:")
    for p in [
        "subject_transition_drops.csv",
        "subject_max_reorganization.csv",
        "group_reorganization_summary.csv",
        "transition_statistics.csv",
        "transition_drop_beyond_matrix.csv",
        "reorganization_summary.txt",
    ]:
        print(f"  - {OUTPUT_DIR / p}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
