"""
Script: 08_changepoint_analysis.py
Purpose: Cross-validate floor estimates via changepoint detection methods.

Inputs:
    - outputs/coordination_floor_gates_by_speed.csv (from 07_detect_floor.py)
    - outputs/coordination_floors.csv (from 07_detect_floor.py)

Outputs:
    - outputs/coordination_floor_fidelity_curve_for_changepoint.csv
    - outputs/coordination_floor_changepoint_crosscheck.csv
    - outputs/changepoint_params.txt
    - outputs/changepoint_summary.txt

Methods:
    - Segmented regression: continuous hinge model with grid-search breakpoint
    - PELT cross-check: best single changepoint under SSE-to-mean cost
    - fidelity_min = min(fidelity_ref_to_q, fidelity_q_to_ref) per subject x speed
    - For descriptive counts at tested speeds, segmented breakpoints are rounded to the nearest tested speed.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =========================
# INPUT PATHS
# =========================
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))

# =========================
# RELEASE-LOCAL OUTPUTS
# =========================
RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"


def segmented_breakpoint_grid(
    x: np.ndarray,
    y: np.ndarray,
    n_grid: int = 2000,
) -> Optional[Dict[str, Any]]:
    """
    Continuous broken-line regression via grid search over breakpoint bp.

    Model: y = a + b*x + c*max(0, bp - x)
    Continuous at x=bp; allows slope change below bp.

    Returns dict with bp, sse, coef.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return None

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if (x_max - x_min) <= 1e-9:
        return None

    eps = 1e-9
    grid = np.linspace(x_min + eps, x_max - eps, int(n_grid))

    best: Optional[Dict[str, Any]] = None
    for bp in grid:
        h = np.clip(bp - x, 0.0, None)
        X = np.column_stack([np.ones_like(x), x, h])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        sse = float(np.sum((y - yhat) ** 2))
        if best is None or sse < best["sse"]:
            best = {"bp": float(bp), "sse": sse, "coef": coef}
    return best


def pelt_single_mean_shift(
    x_desc: np.ndarray,
    y_desc: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Single changepoint under SSE-to-segment-mean cost.
    Input must be ordered from fastest->slowest (descending speed).
    Breakpoint reported as midpoint between x[i-1] and x[i].
    """
    x = np.asarray(x_desc, dtype=float)
    y = np.asarray(y_desc, dtype=float)
    n = y.size
    if n < 2:
        return None

    best: Optional[Dict[str, Any]] = None
    for i in range(1, n):
        y1 = y[:i]
        y2 = y[i:]
        cost = float(np.sum((y1 - y1.mean()) ** 2) + np.sum((y2 - y2.mean()) ** 2))
        if best is None or cost < best["cost"]:
            high = float(x[i - 1])
            low = float(x[i])
            mid = (high + low) / 2.0
            best = {"i": int(i), "cost": cost, "high": high, "low": low, "mid": mid}
    return best


def _round_to_nearest_tested_speed(value: float, tested_speeds: np.ndarray) -> float:
    """Round a continuous estimate to the nearest tested speed (ties -> faster)."""
    if value is None or (not np.isfinite(value)):
        return float("nan")
    ts = np.asarray(tested_speeds, dtype=float)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        return float("nan")

    diffs = np.abs(ts - float(value))
    md = float(np.min(diffs))
    # ties -> choose the faster speed (larger m/s)
    tied = ts[np.isclose(diffs, md, rtol=0.0, atol=1e-12)]
    if tied.size > 0:
        return float(np.max(tied))
    return float(ts[int(np.argmin(diffs))])


def _try_make_plot(
    out_png: Path,
    x: np.ndarray,
    y: np.ndarray,
    primary_floor: float,
    seg_bp: Optional[float],
    pelt_mid: Optional[float],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    order = np.argsort(x)
    xa = x[order]
    ya = y[order]

    plt.figure()
    plt.plot(xa, ya, marker="o")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Fidelity min(ref→q, q→ref)")
    plt.title(title)

    if np.isfinite(primary_floor):
        plt.axvline(primary_floor, linestyle="-")
    if seg_bp is not None and np.isfinite(seg_bp):
        plt.axvline(seg_bp, linestyle="--")
    if pelt_mid is not None and np.isfinite(pelt_mid):
        plt.axvline(pelt_mid, linestyle=":")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Filter to specific dataset")
    ap.add_argument("--n_grid", type=int, default=2000, help="Grid resolution for segmented regression")
    ap.add_argument("--make_plots", action="store_true", help="Generate per-subject plots")
    args = ap.parse_args()

    fig_dir = OUTPUT_DIR / "figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gates_path = UPSTREAM_PATH / "coordination_floor_gates_by_speed.csv"
    floors_path = UPSTREAM_PATH / "coordination_floors.csv"
    if not gates_path.exists():
        raise FileNotFoundError(f"Missing: {gates_path}")
    if not floors_path.exists():
        raise FileNotFoundError(f"Missing: {floors_path}")

    gates = pd.read_csv(gates_path)
    floors = pd.read_csv(floors_path)

    if args.dataset is not None:
        gates = gates[gates["dataset"] == args.dataset].copy()
        floors = floors[floors["dataset"] == args.dataset].copy()

    gates["q_speed_mps"] = pd.to_numeric(gates["q_speed_mps"], errors="coerce")
    gates["trial"] = pd.to_numeric(gates["trial"], errors="coerce")
    floors["trial"] = pd.to_numeric(floors["trial"], errors="coerce")
    floors["floor_speed_mps"] = pd.to_numeric(floors["floor_speed_mps"], errors="coerce")

    gates["fidelity_min"] = gates[["fidelity_ref_to_q", "fidelity_q_to_ref"]].min(axis=1)

    curve_df = gates[["dataset", "subject", "trial", "q_speed_mps", "fidelity_min"]].copy()
    curve_df = curve_df.sort_values(["dataset", "subject", "trial", "q_speed_mps"], ascending=[True, True, True, False])
    curve_df.to_csv(OUTPUT_DIR / "coordination_floor_fidelity_curve_for_changepoint.csv", index=False)

    floors_key = floors.set_index(["dataset", "subject", "trial"], drop=False)

    rows = []
    for (dataset, subject, trial), gdf in gates.groupby(["dataset", "subject", "trial"], sort=True):
        gdf = gdf.dropna(subset=["q_speed_mps", "fidelity_min"]).copy()
        if gdf.shape[0] < 3:
            continue

        gdf = gdf.sort_values("q_speed_mps", ascending=False)
        x_desc = gdf["q_speed_mps"].to_numpy(dtype=float)
        y_desc = gdf["fidelity_min"].to_numpy(dtype=float)

        seg = segmented_breakpoint_grid(x_desc, y_desc, n_grid=args.n_grid)
        pelt = pelt_single_mean_shift(x_desc, y_desc)

        try:
            primary_floor = float(floors_key.loc[(dataset, subject, float(trial)), "floor_speed_mps"])
        except Exception:
            try:
                primary_floor = float(floors_key.loc[(dataset, subject, int(trial)), "floor_speed_mps"])
            except Exception:
                primary_floor = float("nan")

        seg_bp = float(seg["bp"]) if seg is not None else float("nan")
        seg_sse = float(seg["sse"]) if seg is not None else float("nan")

        # Round segmented breakpoint to nearest tested speed for descriptive counts
        tested_speeds = np.unique(x_desc[np.isfinite(x_desc)])
        seg_bp_nearest_tested = _round_to_nearest_tested_speed(seg_bp, tested_speeds)

        pelt_mid = float(pelt["mid"]) if pelt is not None else float("nan")
        pelt_low = float(pelt["low"]) if pelt is not None else float("nan")
        pelt_high = float(pelt["high"]) if pelt is not None else float("nan")

        notes = []
        if np.isfinite(primary_floor) and (np.isfinite(seg_bp) or np.isfinite(pelt_mid)):
            ests = [v for v in [seg_bp, pelt_mid] if np.isfinite(v)]
            slow_est = float(np.nanmin(ests)) if ests else float("nan")
            if np.isfinite(slow_est):
                if slow_est < primary_floor:
                    notes.append(f"crosscheck_slower_than_primary_by={primary_floor - slow_est:.3f}")
                elif slow_est > primary_floor:
                    notes.append(f"crosscheck_faster_than_primary_by={slow_est - primary_floor:.3f}")

        if np.isfinite(seg_bp) and np.isfinite(pelt_mid):
            notes.append(f"seg_vs_pelt_mid_absdiff={abs(seg_bp - pelt_mid):.3f}")

        if np.isfinite(pelt_low) and np.isfinite(pelt_high):
            notes.append(f"pelt_interval=[{pelt_low:.3f},{pelt_high:.3f}]")

        row = {
            "dataset": dataset,
            "subject": subject,
            "trial": int(trial) if pd.notna(trial) else trial,
            "primary_floor_speed_mps": primary_floor,
            "segmented_breakpoint_mps": seg_bp,
            "segmented_breakpoint_nearest_tested_speed_mps": seg_bp_nearest_tested,
            "segmented_sse": seg_sse,
            "pelt_breakpoint_mid_mps": pelt_mid,
            "pelt_break_low_mps": pelt_low,
            "pelt_break_high_mps": pelt_high,
            "n_speeds_used": int(gdf.shape[0]),
            "speeds_desc_mps": ",".join([f"{v:.3f}" for v in x_desc]),
            "notes": " | ".join(notes),
        }
        rows.append(row)

        if args.make_plots:
            title = f"{dataset} {subject} trial={int(trial)}"
            out_png = fig_dir / f"{dataset}_{subject}_trial{int(trial)}_changepoints.png"
            _try_make_plot(out_png, x_desc, y_desc, primary_floor,
                           seg_bp if np.isfinite(seg_bp) else None,
                           pelt_mid if np.isfinite(pelt_mid) else None,
                           title)

    out_df = pd.DataFrame(rows).sort_values(["dataset", "subject", "trial"])
    out_csv = OUTPUT_DIR / "coordination_floor_changepoint_crosscheck.csv"
    out_df.to_csv(out_csv, index=False)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    (OUTPUT_DIR / "changepoint_params.txt").write_text(
        "\n".join(
            [
                f"timestamp_utc: {now}",
                f"upstream_path: {UPSTREAM_PATH.resolve()}",
                f"dataset_filter: {args.dataset}",
                "fidelity_metric: fidelity_min = min(fidelity_ref_to_q, fidelity_q_to_ref)",
                f"segmented_regression: hinge model y=a+b*x+c*max(0,bp-x), bp via grid search (n_grid={args.n_grid})",
                "pelt_crosscheck: best single changepoint under SSE-to-mean cost; breakpoint is midpoint between adjacent tested speeds",
                "segmented_breakpoint_counts: rounded to nearest tested speed for descriptive summaries",
                f"plots_requested: {bool(args.make_plots)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with open(OUTPUT_DIR / "changepoint_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"timestamp_utc: {now}\n")
        f.write(f"rows_written: {out_df.shape[0]}\n")
        if out_df.shape[0] > 0:
            f.write("\nsegmented_breakpoint_nearest_tested_speed_mps counts:\n")
            f.write(out_df["segmented_breakpoint_nearest_tested_speed_mps"].value_counts(dropna=False).to_string() + "\n")
            f.write("\npelt_breakpoint_mid_mps (rounded 0.01) counts:\n")
            f.write(out_df["pelt_breakpoint_mid_mps"].round(2).value_counts(dropna=False).to_string() + "\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {OUTPUT_DIR / 'changepoint_params.txt'}")
    print(f"Wrote: {OUTPUT_DIR / 'changepoint_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
