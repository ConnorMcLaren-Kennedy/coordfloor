# -*- coding: utf-8 -*-
"""
Script: 02b_nmf_rankselect.py
Purpose: Select optimal number of synergies (k) based on VAF thresholds.

Inputs:
    - outputs/nmf_baseline_summary.csv (from 02a_nmf_baseline.py)

Outputs:
    - outputs/chosen_k.csv

Selection criteria:
    - Primary: smallest k where VAF_total >= 0.90 AND VAF_min >= 0.75
    - Fallback: smallest k within 0.01 of max VAF_total
    - Parsimony preference: smallest k that meets criteria
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

# =========================
# PATH CONFIGURATION
# =========================
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))

RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"

# =========================
# RANK SELECTION PARAMETERS
# =========================
VAF_TOTAL_TARGET = 0.90    # Minimum total VAF
VAF_MUSCLE_TARGET = 0.75   # Minimum per-muscle VAF
MAX_VAF_DROP = 0.01        # Fallback: within this of max VAF
PREFER_SMALLEST_K = True   # Parsimony preference


def main() -> int:
    print("=" * 70)
    print("NMF RANK SELECTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find baseline summary
    sum_csv = UPSTREAM_PATH / "nmf_baseline_summary.csv"
    if not sum_csv.exists():
        sum_csv = OUTPUT_DIR / "nmf_baseline_summary.csv"

    if not sum_csv.exists():
        print(f"ERROR: Cannot find nmf_baseline_summary.csv at {sum_csv}")
        return 1

    print(f"Loading baseline summary from {sum_csv}")
    df = pd.read_csv(sum_csv)

    # Required columns
    keys = ["dataset", "subject", "trial", "speed_mps"]
    required = keys + ["k", "vaf_total"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return 1

    has_vaf_min = "vaf_min" in df.columns
    has_ok = "ok" in df.columns
    has_converged = "converged" in df.columns

    print(f"Parameters: vaf_total >= {VAF_TOTAL_TARGET}, vaf_min >= {VAF_MUSCLE_TARGET}")
    print(f"Fallback: within {MAX_VAF_DROP} of max VAF")
    print(f"Prefer smallest k: {PREFER_SMALLEST_K}")

    work = df.copy()

    # Filter to usable rows
    if has_ok:
        work = work[work["ok"].astype(bool)]
    if has_converged:
        work = work[work["converged"].astype(bool)]

    # Ensure numeric
    work["k"] = pd.to_numeric(work["k"], errors="coerce")
    work["vaf_total"] = pd.to_numeric(work["vaf_total"], errors="coerce")
    if has_vaf_min:
        work["vaf_min"] = pd.to_numeric(work["vaf_min"], errors="coerce")

    work = work.dropna(subset=["k", "vaf_total"]).sort_values(keys + ["k"])

    chosen_rows = []
    failures = 0

    for _, g in work.groupby(keys, sort=False):
        g = g.sort_values("k")

        # Rule 1: smallest k meeting thresholds
        cond = g["vaf_total"] >= VAF_TOTAL_TARGET
        if has_vaf_min and (VAF_MUSCLE_TARGET is not None):
            cond = cond & (g["vaf_min"] >= float(VAF_MUSCLE_TARGET))

        if cond.any():
            k_sel = int(g.loc[cond, "k"].min() if PREFER_SMALLEST_K else g.loc[cond, "k"].max())
            rule = "thresholds"
        else:
            # Rule 2: smallest k within max_vaf_drop of max
            max_v = float(g["vaf_total"].max())
            within = g["vaf_total"] >= (max_v - MAX_VAF_DROP)
            k_sel = int(g.loc[within, "k"].min() if PREFER_SMALLEST_K else g.loc[within, "k"].max())
            rule = f"within_{MAX_VAF_DROP}_of_max_vaf"

        r = g[g["k"] == k_sel]
        if r.empty:
            failures += 1
            continue

        # Keep best VAF row if duplicates
        sort_cols = ["vaf_total"] + (["vaf_min"] if has_vaf_min else [])
        r = r.sort_values(sort_cols, ascending=False)

        row = r.iloc[0].to_dict()
        row["chosen_k"] = k_sel
        row["chosen_rule"] = rule
        row["max_vaf_total"] = float(g["vaf_total"].max())
        row["n_k_evaluated"] = int(g["k"].nunique())
        chosen_rows.append(row)

    chosen_df = pd.DataFrame(chosen_rows)
    chosen_df = chosen_df.drop_duplicates(subset=keys, keep="first").sort_values(keys)

    out_csv = OUTPUT_DIR / "chosen_k.csv"
    chosen_df.to_csv(out_csv, index=False)

    # Summary
    uniq_inputs = int(df[keys].drop_duplicates().shape[0])
    print(f"\nWrote: {out_csv}")
    print(f"Unique inputs: {uniq_inputs}")
    print(f"Chosen k rows: {len(chosen_df)}")
    print(f"Failed groups: {failures}")

    if len(chosen_df) > 0:
        print("\nChosen k distribution:")
        print(chosen_df["chosen_k"].value_counts().sort_index().to_string())

    print("=" * 70)
    print("NMF RANK SELECTION COMPLETE")
    print("=" * 70)

    return 0 if len(chosen_df) == uniq_inputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
