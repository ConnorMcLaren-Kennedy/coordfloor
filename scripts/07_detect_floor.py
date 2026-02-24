"""
Script: 07_detect_floor.py
Purpose: Compute coordination floors across speeds.

Primary floor: sequential testing using bidirectional cross-reconstruction fidelity only (G1+G2)
    - floor is the slowest speed where min(F_ref→q, F_q→ref) >= L_ref_fidelity

Secondary (sensitivity) floor: seven-gate sequential testing (G1–G7)
    - floor_7gate is the slowest speed where all seven gates pass

Inputs:
    - outputs/reliability_envelope.csv (from 04_compute_reliability.py)
    - outputs/surrogate_baseline.csv (from 05_compute_surrogates.py)
    - outputs/spatial_temporal_metrics.csv (from 06_compute_spatial_temporal_metrics.py)
    - outputs/coordination_fidelity.csv (from 03_compute_fidelity.py)

Outputs:
    - outputs/coordination_floor_gates_by_speed.csv (all gates per subject x speed)
    - outputs/coordination_floors.csv (primary floor + seven-gate sensitivity floor)

Gates:
    g1: fidelity_ref_to_q >= L_ref_fidelity
    g2: fidelity_q_to_ref >= L_ref_fidelity
    g3: fidelity_ref_to_q > F_null_95
    g4: fidelity_q_to_ref > F_null_95
    g5: cosine_median >= L_ref_cosine_median
    g6: dtw_mean <= U_ref_dtw_mean
    g7: principal_angle_max_deg <= U_ref_principal_angle_max_deg

Sequential testing: 4.0 -> 3.33 -> 3.0 -> 2.78 (early stopping at first failure),
computed independently for (a) primary G1+G2 and (b) seven-gate G1–G7.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# INPUT PATHS
# =========================
# Set via environment variable or defaults to outputs/ in repo
UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", Path(__file__).resolve().parents[1] / "outputs"))


# =========================
# RELEASE-LOCAL OUTPUT PATHS
# =========================
RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RELEASE_ROOT / "outputs"


# =========================
# CONSTANTS
# =========================
REF_SPEED = 5.0
TEST_ORDER = [4.0, 3.33, 3.0, 2.78]  # sequential gatekeeping order (early stop)


def _require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}. Found: {list(df.columns)}")


def _get_row_by_subject_speed(df: pd.DataFrame, subject: str, spd: float, speed_col: str, name: str) -> pd.Series:
    d = df[df["subject"] == subject]
    if d.empty:
        raise RuntimeError(f"{name}: missing subject {subject}")

    vals = d[speed_col].astype(float).values
    mask = np.isclose(vals, float(spd), rtol=0.0, atol=1e-3)

    n = int(mask.sum())
    if n != 1:
        avail = sorted(d[speed_col].astype(float).unique().tolist())
        raise RuntimeError(f"{name}: {subject} expected 1 row at {speed_col}={spd}, found {n}. Available speeds: {avail}")

    return d.iloc[int(np.where(mask)[0][0])]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rel = pd.read_csv(OUTPUT_DIR / "reliability_envelope.csv")
    surr = pd.read_csv(OUTPUT_DIR / "surrogate_baseline.csv")
    met = pd.read_csv(OUTPUT_DIR / "spatial_temporal_metrics.csv")
    fid = pd.read_csv(UPSTREAM_PATH / "coordination_fidelity.csv")

    # Guards
    _require_columns(rel, ["subject", "L_ref_fidelity", "L_ref_cosine_median", "U_ref_dtw_mean", "U_ref_principal_angle_max_deg"], "reliability_envelope.csv")
    _require_columns(surr, ["subject", "q_speed_mps", "F_null_95"], "surrogate_baseline.csv")
    _require_columns(met, ["subject", "q_speed_mps", "cosine_median", "dtw_mean", "principal_angle_max_deg"], "spatial_temporal_metrics.csv")
    _require_columns(fid, ["subject", "q_speed_mps", "fidelity_ref_to_q", "fidelity_q_to_ref"], "coordination_fidelity.csv")

    subjects = sorted(rel["subject"].unique().tolist())

    gates_rows = []
    floors_rows = []

    for subject in subjects:
        r = rel[rel["subject"] == subject].iloc[0]
        L_fid = float(r["L_ref_fidelity"])
        L_cos = float(r["L_ref_cosine_median"])
        U_dtw = float(r["U_ref_dtw_mean"])
        U_ang = float(r["U_ref_principal_angle_max_deg"])

        # Primary: G1+G2 only
        last_pass_primary = REF_SPEED
        first_failed_speed_primary = np.nan
        first_failure_gate_primary = None
        failed_primary = False

        # Sensitivity: seven-gate (G1–G7)
        last_pass_7gate = REF_SPEED
        first_failed_speed_7gate = np.nan
        first_failure_gate_7gate = None
        failed_7gate = False

        # margins stored at PRIMARY first failure speed (kept for backwards compatibility)
        fail_margins = {
            "fail_margin_fid_ref": np.nan,
            "fail_margin_fid_q": np.nan,
            "fail_margin_surrogate_ref": np.nan,
            "fail_margin_surrogate_q": np.nan,
            "fail_margin_cos": np.nan,
            "fail_margin_dtw": np.nan,
            "fail_margin_angle": np.nan,
        }

        for seq_idx, spd in enumerate(TEST_ORDER):
            f = _get_row_by_subject_speed(fid, subject, spd, "q_speed_mps", "coordination_fidelity.csv")
            s = _get_row_by_subject_speed(surr, subject, spd, "q_speed_mps", "surrogate_baseline.csv")
            m = _get_row_by_subject_speed(met, subject, spd, "q_speed_mps", "spatial_temporal_metrics.csv")

            frq = float(f["fidelity_ref_to_q"])
            fqr = float(f["fidelity_q_to_ref"])
            Fnull = float(s["F_null_95"])
            cosmed = float(m["cosine_median"])
            dtwmean = float(m["dtw_mean"])
            angmax = float(m["principal_angle_max_deg"])

            g1 = frq >= L_fid
            g2 = fqr >= L_fid
            g3 = frq > Fnull
            g4 = fqr > Fnull
            g5 = cosmed >= L_cos
            g6 = dtwmean <= U_dtw
            g7 = angmax <= U_ang

            pass_primary = bool(g1 and g2)
            pass_7gate = bool(g1 and g2 and g3 and g4 and g5 and g6 and g7)

            margin_fid_ref = frq - L_fid
            margin_fid_q = fqr - L_fid
            margin_surrogate_ref = frq - Fnull
            margin_surrogate_q = fqr - Fnull
            margin_cos = cosmed - L_cos
            margin_dtw = U_dtw - dtwmean
            margin_angle = U_ang - angmax

            # Under early stopping, speeds are only "considered" until the first failure happens.
            seq_considered_primary = (not failed_primary)
            seq_considered_7gate = (not failed_7gate)

            # Primary early stopping (G1+G2)
            if not failed_primary:
                if pass_primary:
                    last_pass_primary = spd
                else:
                    failed_primary = True
                    first_failed_speed_primary = spd
                    first_failure_gate_primary = "g1" if not g1 else "g2"

                    fail_margins["fail_margin_fid_ref"] = margin_fid_ref
                    fail_margins["fail_margin_fid_q"] = margin_fid_q
                    fail_margins["fail_margin_surrogate_ref"] = margin_surrogate_ref
                    fail_margins["fail_margin_surrogate_q"] = margin_surrogate_q
                    fail_margins["fail_margin_cos"] = margin_cos
                    fail_margins["fail_margin_dtw"] = margin_dtw
                    fail_margins["fail_margin_angle"] = margin_angle

            # Seven-gate early stopping (G1–G7)
            if not failed_7gate:
                if pass_7gate:
                    last_pass_7gate = spd
                else:
                    failed_7gate = True
                    first_failed_speed_7gate = spd
                    for gname, ok in [("g1", g1), ("g2", g2), ("g3", g3), ("g4", g4), ("g5", g5), ("g6", g6), ("g7", g7)]:
                        if not ok:
                            first_failure_gate_7gate = gname
                            break

            gates_rows.append({
                "dataset": "vanhooren",
                "subject": subject,
                "trial": 0,
                "q_speed_mps": float(spd),
                "seq_index": int(seq_idx),

                # Primary early-stopping indicator
                "seq_considered": bool(seq_considered_primary),

                # Seven-gate early-stopping indicator (sensitivity)
                "seq_considered_7gate": bool(seq_considered_7gate),

                "fidelity_ref_to_q": frq,
                "fidelity_q_to_ref": fqr,
                "F_null_95": Fnull,
                "cosine_median": cosmed,
                "dtw_mean": dtwmean,
                "principal_angle_max_deg": angmax,

                "L_ref_fidelity": L_fid,
                "L_ref_cosine_median": L_cos,
                "U_ref_dtw_mean": U_dtw,
                "U_ref_principal_angle_max_deg": U_ang,

                "g1": bool(g1),
                "g2": bool(g2),
                "g3": bool(g3),
                "g4": bool(g4),
                "g5": bool(g5),
                "g6": bool(g6),
                "g7": bool(g7),

                # Pass flags (explicit)
                "pass_primary": bool(pass_primary),
                "pass_7gate": bool(pass_7gate),

                # Backwards-compatible name for seven-gate pass
                "all_gates_pass": bool(pass_7gate),

                "margin_fid_ref": margin_fid_ref,
                "margin_fid_q": margin_fid_q,
                "margin_surrogate_ref": margin_surrogate_ref,
                "margin_surrogate_q": margin_surrogate_q,
                "margin_cos": margin_cos,
                "margin_dtw": margin_dtw,
                "margin_angle": margin_angle,
            })

        all_speeds_pass_primary = (not failed_primary)
        all_speeds_pass_7gate = (not failed_7gate)

        floors_rows.append({
            "dataset": "vanhooren",
            "subject": subject,
            "trial": 0,
            "ref_speed_mps": float(REF_SPEED),

            # Primary output columns
            "floor_speed_mps": float(last_pass_primary),
            "first_failed_speed_mps": first_failed_speed_primary,
            "first_failure_gate": first_failure_gate_primary,
            "all_speeds_pass": bool(all_speeds_pass_primary),

            # Seven-gate sensitivity outputs
            "floor_speed_mps_7gate": float(last_pass_7gate),
            "first_failed_speed_mps_7gate": first_failed_speed_7gate,
            "first_failure_gate_7gate": first_failure_gate_7gate,
            "all_speeds_pass_7gate": bool(all_speeds_pass_7gate),

            # thresholds used
            "L_ref_fidelity": L_fid,
            "L_ref_cosine_median": L_cos,
            "U_ref_dtw_mean": U_dtw,
            "U_ref_principal_angle_max_deg": U_ang,

            # margins at PRIMARY first failure
            **fail_margins,
        })

    gates_df = pd.DataFrame(gates_rows)
    floors_df = pd.DataFrame(floors_rows)

    gates_path = OUTPUT_DIR / "coordination_floor_gates_by_speed.csv"
    floors_path = OUTPUT_DIR / "coordination_floors.csv"

    gates_df.to_csv(gates_path, index=False)
    floors_df.to_csv(floors_path, index=False)

    print("Saved:", gates_path)
    print("Saved:", floors_path)

    print("\nPrimary floor distribution (G1+G2):")
    print(floors_df["floor_speed_mps"].value_counts().sort_index(ascending=False))
    print("All speeds pass (primary):", int(floors_df["all_speeds_pass"].sum()), "/", len(floors_df))

    print("\nSeven-gate floor distribution (G1–G7):")
    print(floors_df["floor_speed_mps_7gate"].value_counts().sort_index(ascending=False))
    print("All speeds pass (7gate):", int(floors_df["all_speeds_pass_7gate"].sum()), "/", len(floors_df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
