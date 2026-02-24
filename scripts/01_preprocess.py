"""
Script: 01_preprocess.py
Purpose: Preprocess raw EMG data and segment into time-normalized strides.

Inputs:
    - Raw C3D files containing EMG and GRF data (van Hooren dataset)
    - Excel file mapping EMG channels to muscles
    - YAML parameter file (params/base.yaml)

Outputs:
    - data/processed/vanhooren/Sub_XX/level_strideparity/*_full.npz
    - data/processed/vanhooren/Sub_XX/level_strideparity/*_matched.npz
    - metadata/qc/vanhooren_level_strideparity_qc.csv

Processing steps:
    1. Bandpass filter (20-450 Hz), optional notch filter for mains interference
    2. Full-wave rectification and lowpass envelope (6 Hz)
    3. GRF-based stride segmentation using Schmitt trigger
    4. Time normalization to 200 samples per stride
    5. Peak normalization to reference speed (5.0 m/s)
    6. QC: clipping detection, baseline drift, SNR estimation
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import yaml
import numpy as np
import pandas as pd
import ezc3d
from scipy.signal import butter, filtfilt, iirnotch, welch


LEVEL_SPEEDS = [2.78, 3.00, 3.33, 4.00, 5.00]

# Minimum number of muscles required in intersection for a subject to be usable
MIN_INTERSECTION_MUSCLES = 4


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analog_channels_by_samples(c3d_obj) -> np.ndarray:
    a = np.asarray(c3d_obj["data"]["analogs"])
    if a.ndim == 3:
        subframes, n_ch, n_frames = a.shape
        return np.transpose(a, (1, 0, 2)).reshape(n_ch, subframes * n_frames)
    if a.ndim == 2:
        return a
    raise ValueError(f"Unexpected analog array shape: {a.shape}")


def find_label_index(labels: list[str], target: str) -> int:
    # exact match preferred; else case-insensitive
    if target in labels:
        return labels.index(target)
    lower = [str(x).lower() for x in labels]
    t = target.lower()
    if t in lower:
        return lower.index(t)
    raise ValueError(f"Analog label not found: {target}. Available example: {labels[:10]}...")


def standardize_vertical_force(fz: np.ndarray) -> np.ndarray:
    fz = np.asarray(fz).astype(float)
    # Some systems store vertical force negative; standardise to positive peaks.
    if np.nanmax(fz) < abs(np.nanmin(fz)):
        fz = -fz
    return fz


def butter_filtfilt(x: np.ndarray, fs: float, btype: str, cutoff, order: int = 4) -> np.ndarray:
    b, a = butter(order, np.asarray(cutoff) / (fs / 2.0), btype=btype)
    return filtfilt(b, a, x)


def mains_ratio(x: np.ndarray, fs: float, mains_hz: float) -> float:
    x = np.asarray(x)
    if x.size < 2048:
        return 0.0
    f, pxx = welch(x, fs=fs, nperseg=min(8192, x.size))
    band = (f >= 20) & (f <= 450)
    total = np.trapz(pxx[band], f[band])
    if total <= 0:
        return 0.0
    narrow = (f >= mains_hz - 1.0) & (f <= mains_hz + 1.0)
    p_narrow = np.trapz(pxx[narrow], f[narrow])
    return float(p_narrow / total)


def clipping_fraction(raw_v: np.ndarray, clip_abs_v: float | None, clip_tol_v: float) -> float:
    x = np.asarray(raw_v).astype(float)
    if x.size == 0:
        return 0.0
    # If clip_abs_v isn't set, estimate the rail from the observed max(|x|)
    rail = float(np.nanmax(np.abs(x))) if (clip_abs_v is None) else float(clip_abs_v)
    if not np.isfinite(rail) or rail <= 0:
        return 0.0
    thr = max(0.0, rail - float(clip_tol_v))
    return float(np.mean(np.abs(x) >= thr))


def preprocess_emg(
    raw_v: np.ndarray,
    fs: float,
    bandpass_hz: tuple[float, float],
    bandpass_order: int,
    envelope_lowpass_hz: float,
    envelope_order: int,
    notch_enabled: bool,
    notch_freqs_hz: list[float],
    notch_q: float,
    notch_ratio_threshold: float,
    notch_ratio_threshold_post: float,
    clip_abs_v: float | None,
    clip_tol_v: float,
    clip_frac_max: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
      env: rectified + lowpass envelope (nonnegative)
      x_filt: demeaned + bandpass (+ optional notch) signal BEFORE rectification
      meta: QC + notch metadata
    """
    meta = {
        "clip_frac": None,
        "clip_fail": False,
        "notch_applied": False,
        "notch_freq": None,
        "mains_ratios_pre": {},
        "mains_ratios_post": {},
        "ratio_pre_max": 0.0,
        "ratio_post_max": 0.0,
        "residual_mains_fail": False,
    }

    raw = np.asarray(raw_v).astype(float)

    # ---- Clipping QC on raw voltage ----
    cf = clipping_fraction(raw, clip_abs_v=clip_abs_v, clip_tol_v=clip_tol_v)
    meta["clip_frac"] = float(cf)
    meta["clip_fail"] = bool(cf >= float(clip_frac_max))

    # ---- Demean + bandpass ----
    x = raw - np.nanmean(raw)
    x = butter_filtfilt(x, fs=fs, btype="bandpass", cutoff=list(bandpass_hz), order=bandpass_order)

    # ---- Mains check (pre) ----
    ratios_pre = {float(f0): mains_ratio(x, fs, float(f0)) for f0 in notch_freqs_hz}
    meta["mains_ratios_pre"] = {str(k): float(v) for k, v in ratios_pre.items()}
    best_f0 = max(ratios_pre, key=ratios_pre.get) if len(ratios_pre) else None
    ratio_pre_max = float(ratios_pre[best_f0]) if best_f0 is not None else 0.0
    meta["ratio_pre_max"] = ratio_pre_max

    # ---- Optional notch ----
    if notch_enabled and (best_f0 is not None) and (ratio_pre_max >= float(notch_ratio_threshold)):
        b, a = iirnotch(w0=float(best_f0), Q=float(notch_q), fs=fs)
        x = filtfilt(b, a, x)
        meta["notch_applied"] = True
        meta["notch_freq"] = float(best_f0)

    # ---- Mains check (post) ----
    ratios_post = {float(f0): mains_ratio(x, fs, float(f0)) for f0 in notch_freqs_hz}
    meta["mains_ratios_post"] = {str(k): float(v) for k, v in ratios_post.items()}
    ratio_post_max = float(max(ratios_post.values())) if len(ratios_post) else 0.0
    meta["ratio_post_max"] = ratio_post_max

    # Residual mains only matters if we decided mains was strong enough to notch
    if meta["notch_applied"] and (ratio_post_max >= float(notch_ratio_threshold_post)):
        meta["residual_mains_fail"] = True

    # ---- Rectify + envelope ----
    env = np.abs(x)
    env = butter_filtfilt(env, fs=fs, btype="lowpass", cutoff=float(envelope_lowpass_hz), order=envelope_order)
    env = np.maximum(env, 0.0)

    return env, x, meta


def parse_speed_from_condition(cond: str) -> float | None:
    s = str(cond).strip()
    if not s.endswith("m/s"):
        return None
    s = s.replace("m/s", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def find_mapping_row(df_map: pd.DataFrame, subject: int, speed: float, tol: float = 1e-3) -> pd.Series:
    """
    Mapping sheet may omit intermediate speeds (e.g., 3.33). Fall back to nearest available speed.
    """
    sub = df_map[df_map["Subject"].astype(int) == int(subject)].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows in mapping sheet for Subject={subject}")

    sub["__speed__"] = sub["Condition"].apply(parse_speed_from_condition)
    sub = sub[sub["__speed__"].notna()]
    sub["__speed__"] = sub["__speed__"].astype(float)

    exact = sub[np.isclose(sub["__speed__"].values, float(speed), atol=tol, rtol=0)]
    if len(exact) > 0:
        return exact.iloc[0]

    # fallback to nearest
    avail = np.sort(sub["__speed__"].unique())
    nearest = float(avail[np.argmin(np.abs(avail - float(speed)))])
    print(
        f"[WARN] No mapping row for Subject={subject} speed={speed}. "
        f"Falling back to nearest available speed={nearest}. Available speeds={avail.tolist()}"
    )
    return sub[np.isclose(sub["__speed__"].values, nearest, atol=tol, rtol=0)].iloc[0]

def adjust_chan_nums_for_file(
    chan_nums: dict[str, int],
    n_analogs: int,
    labels: list[str],
    subject: int,
    speed: float,
) -> dict[str, int]:
    """
    If the mapping sheet uses channel numbers that exceed what the current C3D contains,
    apply a constant offset so max(channel) == n_analogs.

    This is only safe when the missing channels are all BEFORE the EMG block (i.e., a constant shift).
    """
    bad = [(m, ch) for m, ch in chan_nums.items() if (ch < 1) or (ch > n_analogs)]
    if not bad:
        return chan_nums

    max_ch = max(chan_nums.values())
    if max_ch > n_analogs:
        offset = max_ch - n_analogs
        shifted = {m: (ch - offset) for m, ch in chan_nums.items()}
        bad2 = [(m, ch) for m, ch in shifted.items() if (ch < 1) or (ch > n_analogs)]
        if not bad2:
            print(
                f"[WARN] Subject={subject} speed={speed}: EMG mapping channel numbers exceed "
                f"available analog channels (n_analogs={n_analogs}, max_mapped={max_ch}). "
                f"Applying offset={offset} to ALL EMG channels."
            )
            for m in chan_nums:
                ch0 = chan_nums[m]
                ch1 = shifted[m]
                lbl0 = labels[ch0 - 1] if (1 <= ch0 <= n_analogs) else "<out of range>"
                lbl1 = labels[ch1 - 1] if (1 <= ch1 <= n_analogs) else "<out of range>"
                print(f"    {m:16s}: {ch0:>2} ({lbl0}) -> {ch1:>2} ({lbl1})")
            return shifted

    raise ValueError(
        f"Subject={subject} speed={speed}: EMG mapping out of range for n_analogs={n_analogs}. bad={bad}"
    )


def speed_to_suffix(speed: float) -> str:
    if abs(speed - 2.78) < 1e-6:
        return "278ms"
    if abs(speed - 3.00) < 1e-6:
        return "3ms"
    if abs(speed - 3.33) < 1e-6:
        return "333ms"
    if abs(speed - 4.00) < 1e-6:
        return "4ms"
    if abs(speed - 5.00) < 1e-6:
        return "5ms"
    raise ValueError(f"Unexpected speed: {speed}")


def resolve_c3d_path(c3d_dir: Path, subject: int, suffix: str, speed: float) -> Path:
    """
    Return an existing C3D path. Handles a known filename inconsistency:
    some subjects use _333.c3d instead of _333ms.c3d for 3.33 m/s.
    """
    p = c3d_dir / f"Sub_{subject:02d}_{suffix}.c3d"
    if p.exists():
        return p

    # common alternate for 3.33 m/s
    if abs(speed - 3.33) < 1e-6:
        alt = c3d_dir / f"Sub_{subject:02d}_333.c3d"
        if alt.exists():
            return alt

        # last-resort: any 333* file without incline naming
        cands = sorted(c3d_dir.glob(f"Sub_{subject:02d}_333*.c3d"))
        for c in cands:
            name = c.name.lower()
            if ("deg" not in name) and ("down" not in name) and ("up" not in name):
                return c

        if cands:
            return cands[0]

    raise FileNotFoundError(
        f"Could not find C3D for subject={subject}, speed={speed}. "
        f"Tried suffix '{suffix}' and known alternates."
    )


def detect_step_contacts_schmitt(
    fz_total: np.ndarray,
    fs: float,
    thr_on: float,
    min_interval_s: float,
    thr_off_ratio: float = 0.5,
    lowpass_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust step-contact detector:
    - lowpass filter force
    - Schmitt trigger (on threshold + off threshold) to get BOTH contact-start and toe-off indices
    - min interval between contact-starts
    Returns:
      starts: indices where contact begins (>= thr_on)
      ends:   indices where contact ends (<= thr_off)
    """
    fz = np.asarray(fz_total).astype(float)
    fz = butter_filtfilt(fz, fs=fs, btype="lowpass", cutoff=float(lowpass_hz), order=4)

    # standardise sign (just in case)
    if np.nanmax(fz) < abs(np.nanmin(fz)):
        fz = -fz

    thr_off = float(thr_on) * float(thr_off_ratio)

    in_contact = False
    starts: list[int] = []
    ends: list[int] = []

    for i, v in enumerate(fz):
        if (not in_contact) and (v >= thr_on):
            starts.append(int(i))
            in_contact = True
        elif in_contact and (v <= thr_off):
            ends.append(int(i))
            in_contact = False

    # drop incomplete trailing contact
    n = min(len(starts), len(ends))
    starts = starts[:n]
    ends = ends[:n]

    if n == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    # enforce min interval on starts (keep matched ends)
    min_samp = int(round(float(min_interval_s) * fs))
    keep_i: list[int] = []
    last = -10**12
    for i, s in enumerate(starts):
        if s - last >= min_samp:
            keep_i.append(i)
            last = s

    starts_k = np.asarray([starts[i] for i in keep_i], dtype=int)
    ends_k = np.asarray([ends[i] for i in keep_i], dtype=int)

    return starts_k, ends_k


def stride_windows_from_contacts(contacts: np.ndarray, parity: int) -> list[tuple[int, int]]:
    """
    contacts = step contacts (alternating feet)
    parity = 0 or 1
    strides for one leg are every second contact -> windows between successive contacts in that subsequence
    """
    strikes = contacts[parity::2]
    windows = []
    for i in range(len(strikes) - 1):
        a = int(strikes[i])
        b = int(strikes[i + 1])
        if b > a:
            windows.append((a, b))
    return windows


def filter_windows_by_duration(
    windows: list[tuple[int, int]],
    fs: float,
    dur_min_s: float,
    dur_max_s: float,
    mad_k: float,
) -> tuple[list[tuple[int, int]], dict]:
    if len(windows) == 0:
        return [], {"kept": 0, "raw": 0}

    durs = np.array([(b - a) / fs for a, b in windows], dtype=float)

    med = float(np.median(durs))
    mad = float(np.median(np.abs(durs - med)))
    mad_s = 1.4826 * mad  # robust SD-ish scale

    lo = max(float(dur_min_s), med - float(mad_k) * mad_s)
    hi = min(float(dur_max_s), med + float(mad_k) * mad_s)

    keep = [(a, b) for (a, b), d in zip(windows, durs) if (d >= lo) and (d <= hi)]
    kept_durs = np.array([(b - a) / fs for a, b in keep], dtype=float)

    info = {
        "raw": int(durs.size),
        "kept": int(kept_durs.size),
        "dur_median_s": med,
        "dur_mad_scaled_s": float(mad_s),
        "dur_lo_s": float(lo),
        "dur_hi_s": float(hi),
        "dur_min_s": float(np.min(durs)),
        "dur_max_s": float(np.max(durs)),
        "kept_q05_s": float(np.quantile(kept_durs, 0.05)) if kept_durs.size else None,
        "kept_q50_s": float(np.quantile(kept_durs, 0.50)) if kept_durs.size else None,
        "kept_q95_s": float(np.quantile(kept_durs, 0.95)) if kept_durs.size else None,
    }
    return keep, info


def time_normalise_stride(x: np.ndarray, a: int, b: int, T: int) -> np.ndarray:
    seg = x[a:b]
    if seg.size <= 1:
        return np.full((T,), np.nan, dtype=float)
    src = np.linspace(0.0, 1.0, seg.size)
    dst = np.linspace(0.0, 1.0, T)
    return np.interp(dst, src, seg)


def score_windows_by_emg_timing(env_by_muscle: dict[str, np.ndarray], windows: list[tuple[int, int]], T: int) -> float:
    # Heuristic: correct parity should give early TA burst and later SOL burst
    needed = ["Tibialis", "Sol"]
    for m in needed:
        if m not in env_by_muscle:
            return 1e9

    if len(windows) < 5:
        return 1e9

    take = windows[: min(30, len(windows))]
    ta_peaks = []
    sol_peaks = []
    for (a, b) in take:
        ta = time_normalise_stride(env_by_muscle["Tibialis"], a, b, T)
        sol = time_normalise_stride(env_by_muscle["Sol"], a, b, T)
        ta_peaks.append(int(np.nanargmax(ta)))
        sol_peaks.append(int(np.nanargmax(sol)))

    # dispersion + ordering penalty if SOL doesn't peak after TA
    ta_std = float(np.std(ta_peaks))
    sol_std = float(np.std(sol_peaks))
    order_pen = float(np.mean(np.array(sol_peaks) <= np.array(ta_peaks))) * 50.0
    return ta_std + sol_std + order_pen


def filter_windows_by_toeoff(
    windows: list[tuple[int, int]],
    toeoff_by_contact: dict[int, int],
) -> tuple[list[tuple[int, int]], int]:
    kept: list[tuple[int, int]] = []
    dropped = 0
    for (a, b) in windows:
        toe = toeoff_by_contact.get(int(a), None)
        if toe is None or toe <= a or toe >= b:
            dropped += 1
            continue
        kept.append((a, b))
    return kept, dropped


def baseline_drift_qc(
    x_filt_by_muscle: dict[str, np.ndarray],
    windows: list[tuple[int, int]],
    muscles: list[str],
    fs: float,
    baseline_pre_s: float,
    baseline_end_s: float,
    drift_mad_z: float,
    drift_channel_fail_frac: float,
) -> tuple[list[tuple[int, int]], dict[str, dict], dict]:
    """
    Implements baseline-drift QC as in the written spec.

    Returns:
      windows_kept: windows after dropping union-of-outlier strides (computed across muscles)
      drift_meta_by_muscle: {muscle: {outlier_frac, drift_fail, baseline_med, baseline_mad_scaled}}
      info: summary dict
    """
    info = {
        "n_in": int(len(windows)),
        "n_dropped_invalid_baseline": 0,
        "n_dropped_union_outliers": 0,
        "n_out": 0,
        "muscles_checked": list(muscles),
        "muscles_failed": [],
    }

    if len(windows) == 0 or len(muscles) == 0:
        info["n_out"] = 0
        return [], {}, info

    pre_samp = int(round(float(baseline_pre_s) * fs))
    end_samp = int(round(float(baseline_end_s) * fs))
    if pre_samp <= end_samp:
        raise ValueError("baseline_pre_s must be > baseline_end_s")

    # drop strides where baseline window would be invalid (same for all muscles)
    valid_idx = []
    for i, (a, _b) in enumerate(windows):
        a = int(a)
        i0 = a - pre_samp
        i1 = a - end_samp
        if i0 < 0 or i1 <= i0:
            continue
        valid_idx.append(i)

    if len(valid_idx) != len(windows):
        info["n_dropped_invalid_baseline"] = int(len(windows) - len(valid_idx))
        windows = [windows[i] for i in valid_idx]

    if len(windows) == 0:
        info["n_out"] = 0
        return [], {}, info

    n = len(windows)

    # compute per-muscle baseline medians and robust z-scores
    drift_meta: dict[str, dict] = {}
    outlier_masks: dict[str, np.ndarray] = {}

    for m in muscles:
        x = np.asarray(x_filt_by_muscle[m]).astype(float)
        meds = np.zeros((n,), dtype=float)
        for i, (a, _b) in enumerate(windows):
            a = int(a)
            seg = x[(a - pre_samp) : (a - end_samp)]
            meds[i] = float(np.nanmedian(seg))

        med = float(np.median(meds))
        mad = float(np.median(np.abs(meds - med)))
        mad_s = 1.4826 * mad
        denom = mad_s if mad_s > 0 else 1e-12
        z = np.abs(meds - med) / denom
        out = z > float(drift_mad_z)

        frac = float(np.mean(out)) if out.size else 0.0
        fail = bool(frac > float(drift_channel_fail_frac))

        drift_meta[m] = {
            "baseline_med": med,
            "baseline_mad_scaled": float(mad_s),
            "outlier_frac": frac,
            "drift_fail": fail,
        }
        outlier_masks[m] = out

    muscles_failed = [m for m in muscles if drift_meta[m]["drift_fail"]]
    info["muscles_failed"] = muscles_failed

    # union outliers across muscles (so all muscles retain the same stride set)
    union = np.zeros((n,), dtype=bool)
    for m in muscles:
        union |= outlier_masks[m]

    keep_idx = [i for i, bad in enumerate(union) if not bad]
    info["n_dropped_union_outliers"] = int(len(union) - len(keep_idx))
    windows_kept = [windows[i] for i in keep_idx]
    info["n_out"] = int(len(windows_kept))

    return windows_kept, drift_meta, info


def snr_by_muscle(
    x_filt_by_muscle: dict[str, np.ndarray],
    windows: list[tuple[int, int]],
    toeoff_by_contact: dict[int, int],
    fs: float,
    baseline_pre_s: float,
    baseline_end_s: float,
    eps: float,
) -> dict[str, dict]:
    """
    SNR per the spec:
      SNR_stride = RMS_active / RMS_baseline, using x_filt (before rectification)
      active = contact -> toe-off, baseline = pre-contact window
    Returns {muscle: {snr_median, snr_n}}
    """
    pre_samp = int(round(float(baseline_pre_s) * fs))
    end_samp = int(round(float(baseline_end_s) * fs))
    out: dict[str, dict] = {}

    if len(windows) == 0:
        for m in x_filt_by_muscle:
            out[m] = {"snr_median": np.nan, "snr_n": 0}
        return out

    stride_starts = [int(a) for (a, _b) in windows]
    toeoffs = [int(toeoff_by_contact.get(a, -1)) for a in stride_starts]

    valid = []
    for i, (a, toe) in enumerate(zip(stride_starts, toeoffs)):
        i0 = a - pre_samp
        i1 = a - end_samp
        if toe <= a or i0 < 0 or i1 <= i0:
            continue
        valid.append(i)

    if len(valid) == 0:
        for m in x_filt_by_muscle:
            out[m] = {"snr_median": np.nan, "snr_n": 0}
        return out

    for m, x in x_filt_by_muscle.items():
        x = np.asarray(x).astype(float)
        snrs = []
        for i in valid:
            a = stride_starts[i]
            toe = toeoffs[i]
            base = x[(a - pre_samp) : (a - end_samp)]
            act = x[a:toe]
            if base.size == 0 or act.size == 0:
                continue
            rms_base = float(np.sqrt(np.mean(base**2)))
            rms_act = float(np.sqrt(np.mean(act**2)))
            snrs.append(rms_act / (rms_base + float(eps)))

        snrs = np.asarray(snrs, dtype=float)
        out[m] = {"snr_median": float(np.nanmedian(snrs)) if snrs.size else np.nan, "snr_n": int(snrs.size)}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params/base.yaml")
    ap.add_argument("--subject", type=int, required=True)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.params))
    subject = int(args.subject)

    van = cfg.get("vanhooren", {})
    c3d_dir = Path(van.get("c3d_dir", "data/raw/vanhooren/0. C3D files"))
    xlsx = Path(van.get("emg_map_xlsx", "data/raw/vanhooren/Excel file with EMG channels.xlsx"))

    # -------------------- Params --------------------
    pp = cfg.get("preprocess", {})
    bandpass_hz = tuple(pp.get("bandpass_hz", [20, 450]))
    bandpass_order = int(pp.get("bandpass_order", 4))
    envelope_lowpass_hz = float(pp.get("envelope_lowpass_hz", 6))
    envelope_order = int(pp.get("envelope_order", 4))

    notch_enabled = bool(pp.get("notch_enabled", True))
    notch_freqs_hz = list(pp.get("notch_freqs_hz", [50, 60]))
    notch_q = float(pp.get("notch_q", 30))
    notch_ratio_threshold = float(pp.get("notch_ratio_threshold", 0.05))
    notch_ratio_threshold_post = float(pp.get("notch_ratio_threshold_post", 0.02))

    # clipping thresholds (raw voltage)
    clip_abs_v = pp.get("clip_abs_v", None)
    if clip_abs_v is not None:
        clip_abs_v = float(clip_abs_v)
    clip_tol_v = float(pp.get("clip_tol_v", 0.05))
    clip_frac_max = float(pp.get("clip_frac_max", 0.001))

    # SNR reporting
    snr_floor = float(pp.get("snr_floor", 3.0))
    snr_eps = float(pp.get("snr_eps", 1e-8))

    st = cfg.get("stride", {})
    grf_threshold_n = float(st.get("grf_threshold_n", 20))
    min_strike_interval_s = float(st.get("min_strike_interval_s", 0.20))
    dur_min_s = float(st.get("stride_duration_min_s", 0.50))
    dur_max_s = float(st.get("stride_duration_max_s", 1.50))
    mad_k = float(st.get("stride_duration_mad_k", 5))
    T = int(st.get("time_norm_len", 200))
    match_seed = int(st.get("match_seed", 20251214))

    thr_off_ratio = float(st.get("thr_off_ratio", 0.5))
    force_lowpass_hz = float(st.get("force_lowpass_hz", 20.0))

    baseline_pre_s = float(st.get("baseline_pre_s", 0.05))
    baseline_end_s = float(st.get("baseline_end_s", 0.01))
    drift_mad_z = float(st.get("drift_mad_z", 3.0))
    drift_channel_fail_frac = float(st.get("drift_channel_fail_frac", 0.20))

    muscles_cfg = cfg.get("muscles", {})
    keep_muscles = list(muscles_cfg.get("vanhooren_keep", []))
    if not keep_muscles:
        raise ValueError("No muscles listed in params under muscles: vanhooren_keep")

    rng = np.random.default_rng(match_seed)

    # -------------------- Outputs --------------------
    out_dir = Path("data/processed/vanhooren") / f"Sub_{subject:02d}" / "level_strideparity"
    out_dir.mkdir(parents=True, exist_ok=True)

    Path("metadata").mkdir(exist_ok=True)
    qc_path = Path("metadata/qc/vanhooren_level_strideparity_qc.csv")
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    print("SUBJECT:", subject)
    print("C3D DIR:", c3d_dir)
    print("XLSX:", xlsx)
    print("OUTPUT:", out_dir)
    print("KEEP MUSCLES:", keep_muscles)

    print("PARAMS:", {
        "bandpass_hz": bandpass_hz,
        "bandpass_order": bandpass_order,
        "envelope_lowpass_hz": envelope_lowpass_hz,
        "envelope_order": envelope_order,
        "notch_enabled": notch_enabled,
        "notch_freqs_hz": notch_freqs_hz,
        "notch_q": notch_q,
        "notch_ratio_threshold": notch_ratio_threshold,
        "notch_ratio_threshold_post": notch_ratio_threshold_post,
        "clip_abs_v": clip_abs_v,
        "clip_tol_v": clip_tol_v,
        "clip_frac_max": clip_frac_max,
        "snr_floor": snr_floor,
        "snr_eps": snr_eps,
    })
    print("STRIDE PARAMS:", {
        "grf_threshold_n": grf_threshold_n,
        "min_strike_interval_s": min_strike_interval_s,
        "dur_min_s": dur_min_s,
        "dur_max_s": dur_max_s,
        "mad_k": mad_k,
        "T": T,
        "match_seed": match_seed,
        "thr_off_ratio": thr_off_ratio,
        "force_lowpass_hz": force_lowpass_hz,
        "baseline_pre_s": baseline_pre_s,
        "baseline_end_s": baseline_end_s,
        "drift_mad_z": drift_mad_z,
        "drift_channel_fail_frac": drift_channel_fail_frac,
    })

    df_map = pd.read_excel(xlsx, sheet_name="Corresponding analogue channel")

    # -------------------- Reference peaks at 5.0 m/s --------------------
    ref_speed = 5.00
    ref_suffix = speed_to_suffix(ref_speed)
    ref_c3d = resolve_c3d_path(c3d_dir, subject, ref_suffix, ref_speed)
    print("\nREFERENCE SPEED:", ref_speed, "FILE:", ref_c3d)

    c3d = ezc3d.c3d(str(ref_c3d))
    labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
    fs = float(c3d["header"]["analogs"]["frame_rate"])
    A = analog_channels_by_samples(c3d)

    row_ref = find_mapping_row(df_map, subject, ref_speed)
    chan_nums_ref = {m: int(row_ref[m]) for m in keep_muscles}
    chan_nums_ref = adjust_chan_nums_for_file(chan_nums_ref, n_analogs=A.shape[0], labels=labels, subject=subject, speed=ref_speed)


    env_ref: dict[str, np.ndarray] = {}
    meta_ref: dict[str, dict] = {}
    ref_ok: dict[str, bool] = {}

    n_analogs = A.shape[0]
    for m in keep_muscles:
        ch_num = int(chan_nums_ref[m])
        ch_idx = ch_num - 1

        if ch_idx < 0 or ch_idx >= n_analogs:
            print(
                f"  [MISSING CHANNEL @ REF] {m} expected channel {ch_num} "
                f"but file only has {n_analogs} analog channels. Excluding this muscle for this subject."
            )
            # Placeholder so ref_peaks + logging don't crash later
            env_ref[m] = np.array([np.nan], dtype=float)
            meta_ref[m] = {
                "clip_frac": np.nan,
                "clip_fail": True,
                "notch_applied": False,
                "notch_freq": None,
                "ratio_pre_max": np.nan,
                "ratio_post_max": np.nan,
                "residual_mains_fail": True,
            }
            ref_ok[m] = False
            continue

        raw = A[ch_idx, :]

        env, x_filt, meta = preprocess_emg(
            raw_v=raw,
            fs=fs,
            bandpass_hz=bandpass_hz,
            bandpass_order=bandpass_order,
            envelope_lowpass_hz=envelope_lowpass_hz,
            envelope_order=envelope_order,
            notch_enabled=notch_enabled,
            notch_freqs_hz=notch_freqs_hz,
            notch_q=notch_q,
            notch_ratio_threshold=notch_ratio_threshold,
            notch_ratio_threshold_post=notch_ratio_threshold_post,
            clip_abs_v=clip_abs_v,
            clip_tol_v=clip_tol_v,
            clip_frac_max=clip_frac_max,
        )
        env_ref[m] = env
        meta_ref[m] = meta

        ok = (not meta["clip_fail"]) and (not meta["residual_mains_fail"])
        ref_ok[m] = bool(ok)

    # NOTE: Reference peaks are computed from the full reference-speed envelope trace
    # (trial-level) and are not restricted to the subset of strides retained after
    # stride-level screening (duration/toe-off/baseline-drift QC).
    ref_peaks = {m: float(np.nanmax(env_ref[m])) for m in keep_muscles}

    print("\nREFERENCE PEAKS (used for scaling all speeds)")
    for m in keep_muscles:
        print(
            f"  {m:16s}: peak={ref_peaks[m]:.6g} | ref_ok={ref_ok[m]} | "
            f"clip_frac={meta_ref[m]['clip_frac']:.6g} | clip_fail={meta_ref[m]['clip_fail']} | "
            f"notch={meta_ref[m]['notch_applied']}@{meta_ref[m]['notch_freq']} | "
            f"ratio_pre_max={meta_ref[m]['ratio_pre_max']:.4g} | ratio_post_max={meta_ref[m]['ratio_post_max']:.4g} | "
            f"residual_fail={meta_ref[m]['residual_mains_fail']}"
        )

    # -------------------- Process each speed (DO NOT SAVE YET) --------------------
    per_speed: dict[float, dict] = {}
    qc_rows: list[dict] = []
    subject_all_speeds_ok = True

    min_keep = 20

    for speed in LEVEL_SPEEDS:
        suffix = speed_to_suffix(speed)
        c3d_path = resolve_c3d_path(c3d_dir, subject, suffix, speed)
        print("\n" + "-" * 80)
        print("SPEED:", speed, "FILE:", c3d_path)

        c3d = ezc3d.c3d(str(c3d_path))
        labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
        fs = float(c3d["header"]["analogs"]["frame_rate"])
        A = analog_channels_by_samples(c3d)

        # total vertical force
        idx_fz1 = find_label_index(labels, "Force.Fz1")
        idx_fz2 = find_label_index(labels, "Force.Fz2")
        fz_total = standardize_vertical_force(A[idx_fz1, :]) + standardize_vertical_force(A[idx_fz2, :])

        # EMG mapping row (fallback if missing)
        row = find_mapping_row(df_map, subject, speed)
        chan_nums = {m: int(row[m]) for m in keep_muscles}
        # Apply channel-number offset correction if needed (see Appendix B.7)
        chan_nums = adjust_chan_nums_for_file(chan_nums, n_analogs=A.shape[0], labels=labels, subject=subject, speed=speed)


        # preprocess EMG per muscle
        env_by_muscle: dict[str, np.ndarray] = {}
        x_by_muscle: dict[str, np.ndarray] = {}
        meta_by_muscle: dict[str, dict] = {}

        pre_fail: dict[str, bool] = {}  # clipping or residual mains OR excluded due to ref fail

        n_analogs = A.shape[0]

        for m in keep_muscles:
            ch_num = int(chan_nums[m])
            ch_idx = ch_num - 1

            if ch_idx < 0 or ch_idx >= n_analogs:
                print(
                    f"  [MISSING CHANNEL] {m:16s} | ch={ch_num:2d} | "
                    f"file has only {n_analogs} analog channels -> marking as failed/excluded."
                )
                meta_by_muscle[m] = {
                    "clip_frac": np.nan,
                    "clip_fail": True,
                    "notch_applied": False,
                    "notch_freq": None,
                    "ratio_pre_max": np.nan,
                    "ratio_post_max": np.nan,
                    "residual_mains_fail": True,
                }
                pre_fail[m] = True
                continue

            raw = A[ch_idx, :]


            env, x_filt, meta = preprocess_emg(
                raw_v=raw,
                fs=fs,
                bandpass_hz=bandpass_hz,
                bandpass_order=bandpass_order,
                envelope_lowpass_hz=envelope_lowpass_hz,
                envelope_order=envelope_order,
                notch_enabled=notch_enabled,
                notch_freqs_hz=notch_freqs_hz,
                notch_q=notch_q,
                notch_ratio_threshold=notch_ratio_threshold,
                notch_ratio_threshold_post=notch_ratio_threshold_post,
                clip_abs_v=clip_abs_v,
                clip_tol_v=clip_tol_v,
                clip_frac_max=clip_frac_max,
            )

            env_by_muscle[m] = env
            x_by_muscle[m] = x_filt
            meta_by_muscle[m] = meta

            pf = bool(meta["clip_fail"] or meta["residual_mains_fail"] or (not ref_ok.get(m, True)))
            pre_fail[m] = pf

            print(
                f"  {m:16s} | ch={chan_nums[m]:2d} | "
                f"clip_frac={meta['clip_frac']:.6g} | clip_fail={meta['clip_fail']} | "
                f"notch={meta['notch_applied']}@{meta['notch_freq']} | "
                f"ratio_pre_max={meta['ratio_pre_max']:.4g} | ratio_post_max={meta['ratio_post_max']:.4g} | "
                f"residual_fail={meta['residual_mains_fail']} | "
                f"excluded_ref={not ref_ok.get(m, True)}"
            )

        # step contacts (and toe-offs)
        contacts, toe_offs = detect_step_contacts_schmitt(
            fz_total=fz_total,
            fs=fs,
            thr_on=grf_threshold_n,
            min_interval_s=min_strike_interval_s,
            thr_off_ratio=thr_off_ratio,
            lowpass_hz=force_lowpass_hz,
        )
        print("STEP CONTACTS DETECTED:", int(len(contacts)))

        toeoff_by_contact = {int(c): int(o) for c, o in zip(contacts.tolist(), toe_offs.tolist())}

        # parity 0 and 1 windows + stride-duration QC
        w0_raw = stride_windows_from_contacts(contacts, parity=0)
        w0, info0 = filter_windows_by_duration(w0_raw, fs, dur_min_s, dur_max_s, mad_k)
        s0 = score_windows_by_emg_timing(env_by_muscle, w0, T)

        w1_raw = stride_windows_from_contacts(contacts, parity=1)
        w1, info1 = filter_windows_by_duration(w1_raw, fs, dur_min_s, dur_max_s, mad_k)
        s1 = score_windows_by_emg_timing(env_by_muscle, w1, T)

        print("PARITY 0 QC:", info0, "score:", s0)
        print("PARITY 1 QC:", info1, "score:", s1)

        # choose parity: prefer enough strides, then best score
        candidates = []
        if info0.get("kept", 0) > 0:
            candidates.append((0, s0, info0.get("kept", 0), w0))
        if info1.get("kept", 0) > 0:
            candidates.append((1, s1, info1.get("kept", 0), w1))

        if len(candidates) == 0:
            print("[FAIL] No strides after duration QC at speed", speed)
            subject_all_speeds_ok = False
            parity = None
            windows = []
        else:
            # If one parity meets min_keep and the other doesn't, take the one that meets it.
            if info0.get("kept", 0) >= min_keep and info1.get("kept", 0) < min_keep:
                best = (0, s0, info0.get("kept", 0), w0)
            elif info1.get("kept", 0) >= min_keep and info0.get("kept", 0) < min_keep:
                best = (1, s1, info1.get("kept", 0), w1)
            else:
                best = sorted(candidates, key=lambda t: t[1])[0]  # lowest score

            parity, score, n_kept, windows = best
            print("CHOSEN PARITY:", parity, "| kept strides:", int(n_kept), "| score:", float(score))

        # toe-off validity filter (needed for SNR)
        n_before_toe = len(windows)
        windows, dropped_toe = filter_windows_by_toeoff(windows, toeoff_by_contact)
        if dropped_toe:
            print(f"TOE-OFF FILTER: dropped {dropped_toe} strides (missing/invalid toe-off).")
        n_after_toe = len(windows)

        # baseline drift QC (stride-level), after parity selection
        drift_meta_by_muscle: dict[str, dict] = {}
        drift_info = {"n_in": n_after_toe, "n_out": n_after_toe}

        muscles_for_drift = [m for m in keep_muscles if not pre_fail.get(m, True)]
        if len(windows) > 0 and len(muscles_for_drift) > 0:
            windows_drifted, drift_meta_by_muscle, drift_info = baseline_drift_qc(
                x_filt_by_muscle=x_by_muscle,
                windows=windows,
                muscles=muscles_for_drift,
                fs=fs,
                baseline_pre_s=baseline_pre_s,
                baseline_end_s=baseline_end_s,
                drift_mad_z=drift_mad_z,
                drift_channel_fail_frac=drift_channel_fail_frac,
            )
            print("BASELINE DRIFT QC:", drift_info)
            if drift_info.get("muscles_failed"):
                print("  Drift-failed muscles:", drift_info["muscles_failed"])
            windows = windows_drifted
        else:
            if len(muscles_for_drift) == 0:
                print("[WARN] All muscles failed pre-QC; skipping drift QC.")
            if len(windows) == 0:
                print("[WARN] No windows to apply drift QC.")

        n_final = len(windows)

        # determine muscles for output at this speed (BEFORE intersection)
        drift_fail = {m: bool(drift_meta_by_muscle.get(m, {}).get("drift_fail", False)) for m in keep_muscles}
        muscles_out = [m for m in keep_muscles if (not pre_fail.get(m, True)) and (not drift_fail[m])]

        # compute SNR (reported)
        snr_meta = snr_by_muscle(
            x_filt_by_muscle={m: x_by_muscle[m] for m in muscles_out if m in x_by_muscle},
            windows=windows,
            toeoff_by_contact=toeoff_by_contact,
            fs=fs,
            baseline_pre_s=baseline_pre_s,
            baseline_end_s=baseline_end_s,
            eps=snr_eps,
        )
        snr_medians = [snr_meta[m]["snr_median"] for m in muscles_out if m in snr_meta and np.isfinite(snr_meta[m]["snr_median"])]
        speed_snr_median = float(np.nanmedian(snr_medians)) if len(snr_medians) else np.nan
        low_snr_flag = bool(np.isfinite(speed_snr_median) and (speed_snr_median < snr_floor))

        print(f"SNR (median across muscles) @ {speed} m/s:", speed_snr_median, "| low_snr_flag:", low_snr_flag)

        # speed viability check (after ALL QC)
        speed_ok = (n_final >= min_keep) and (parity is not None) and (len(muscles_out) >= 2)
        if not speed_ok:
            print(
                f"[FAIL] Speed {speed} unusable after QC: "
                f"strides={n_final} (min_keep={min_keep}), muscles_out={len(muscles_out)}, parity={parity}"
            )
            subject_all_speeds_ok = False

        # write per-muscle QC rows (always, even if speed fails)
        for m in keep_muscles:
            meta = meta_by_muscle.get(m, {
                "clip_frac": np.nan,
                "clip_fail": True,
                "notch_applied": False,
                "notch_freq": None,
                "ratio_pre_max": np.nan,
                "ratio_post_max": np.nan,
                "residual_mains_fail": True,
            })
            dmeta = drift_meta_by_muscle.get(m, {})
            qc_rows.append(
                {
                    "subject": subject,
                    "speed": float(speed),
                    "suffix": suffix,
                    "muscle": m,
                    "channel_num": int(chan_nums[m]),
                    "passed_speed_qc": bool(m in muscles_out and speed_ok),
                    "ref_ok": bool(ref_ok.get(m, True)),
                    "clip_frac": float(meta["clip_frac"]) if meta.get("clip_frac") is not None else np.nan,
                    "clip_fail": bool(meta.get("clip_fail", True)),
                    "notch_applied": bool(meta.get("notch_applied", False)),
                    "notch_freq": float(meta["notch_freq"]) if meta.get("notch_freq") is not None else np.nan,
                    "ratio_pre_max": float(meta.get("ratio_pre_max", np.nan)),
                    "ratio_post_max": float(meta.get("ratio_post_max", np.nan)),
                    "residual_mains_fail": bool(meta.get("residual_mains_fail", True)),
                    "drift_outlier_frac": float(dmeta.get("outlier_frac", np.nan)),
                    "drift_fail": bool(dmeta.get("drift_fail", False)),
                    "snr_median": float(snr_meta[m]["snr_median"]) if (m in snr_meta) else np.nan,
                    "low_snr_flag_speed": bool(low_snr_flag),
                    "n_step_contacts": int(len(contacts)),
                    "parity0_kept": int(info0.get("kept", 0)),
                    "parity1_kept": int(info1.get("kept", 0)),
                    "parity_chosen": int(parity) if parity is not None else np.nan,
                    "n_strides_pre_toeoff": int(n_before_toe),
                    "n_strides_post_toeoff": int(n_after_toe),
                    "n_strides_post_drift": int(n_final),
                    "speed_ok": bool(speed_ok),
                }
            )

        
        if speed_ok:
            per_speed[float(speed)] = {
                "suffix": suffix,
                "fs": fs,
                "parity": parity,
                "windows": windows,
                "env_by_muscle": env_by_muscle,
                "muscles_out": muscles_out,
                "meta_by_muscle": meta_by_muscle,
                "drift_meta_by_muscle": drift_meta_by_muscle,
                "drift_info": drift_info,
                "snr_meta": snr_meta,
                "low_snr_flag": low_snr_flag,
                "info0": info0,
                "info1": info1,
            }

    # -------------------- COMPUTE MUSCLE INTERSECTION ACROSS ALL SPEEDS --------------------
    print("\n" + "=" * 80)
    print("COMPUTING MUSCLE INTERSECTION ACROSS ALL SPEEDS")
    print("=" * 80)

    if (len(per_speed) != len(LEVEL_SPEEDS)) or (not subject_all_speeds_ok):
        print(
            "\n[FAIL] Cannot compute intersection: not all speeds usable. "
            f"Speeds saved={sorted(per_speed.keys())}."
        )
        
        df_qc = pd.DataFrame(qc_rows)
        if qc_path.exists():
            try:
                df_prev = pd.read_csv(qc_path)
                df_prev = df_prev[df_prev["subject"].astype(int) != int(subject)]
                df_out = pd.concat([df_prev, df_qc], ignore_index=True)
            except Exception as e:
                print("[WARN] Could not read existing QC CSV, overwriting. Error:", e)
                df_out = df_qc
        else:
            df_out = df_qc
        df_out.to_csv(qc_path, index=False)
        print("\nQC CSV WRITTEN:", qc_path, "| rows:", len(df_qc))
        return

    # Compute intersection: muscles that passed QC at ALL speeds
    muscles_per_speed = [set(per_speed[spd]["muscles_out"]) for spd in LEVEL_SPEEDS]
    muscles_intersection = set.intersection(*muscles_per_speed)
    muscles_intersection = sorted(muscles_intersection, key=lambda m: keep_muscles.index(m))

    print(f"\nMuscles per speed:")
    for spd in LEVEL_SPEEDS:
        print(f"  {spd} m/s: {sorted(per_speed[spd]['muscles_out'])}")
    print(f"\nINTERSECTION ({len(muscles_intersection)} muscles): {muscles_intersection}")

    if len(muscles_intersection) < MIN_INTERSECTION_MUSCLES:
        print(
            f"\n[FAIL] Insufficient muscles in intersection: {len(muscles_intersection)} < {MIN_INTERSECTION_MUSCLES}. "
            f"Subject {subject} will not have usable outputs."
        )
        # Update QC rows to mark all as not included
        for row in qc_rows:
            row["included_in_output"] = False
        
        df_qc = pd.DataFrame(qc_rows)
        if qc_path.exists():
            try:
                df_prev = pd.read_csv(qc_path)
                df_prev = df_prev[df_prev["subject"].astype(int) != int(subject)]
                df_out = pd.concat([df_prev, df_qc], ignore_index=True)
            except Exception as e:
                print("[WARN] Could not read existing QC CSV, overwriting. Error:", e)
                df_out = df_qc
        else:
            df_out = df_qc
        df_out.to_csv(qc_path, index=False)
        print("\nQC CSV WRITTEN:", qc_path, "| rows:", len(df_qc))
        return

    # Update QC rows with final included_in_output based on intersection
    for row in qc_rows:
        m = row["muscle"]
        spd_ok = row["speed_ok"]
        row["included_in_output"] = bool(m in muscles_intersection and spd_ok)

    # -------------------- SAVE FULL NPZ FILES (with intersection muscles only) --------------------
    print("\n" + "-" * 80)
    print("SAVING FULL NPZ FILES (intersection muscles only)")
    print("-" * 80)

    for speed in LEVEL_SPEEDS:
        data = per_speed[float(speed)]
        suffix = data["suffix"]
        fs = data["fs"]
        parity = data["parity"]
        windows = data["windows"]
        env_by_muscle = data["env_by_muscle"]

        n = len(windows)
        M = len(muscles_intersection)
        env_norm = np.zeros((M, n, T), dtype=float)

        for mi, m in enumerate(muscles_intersection):
            peak = float(ref_peaks[m])
            for si, (a, b) in enumerate(windows):
                s = time_normalise_stride(env_by_muscle[m], a, b, T)
                env_norm[mi, si, :] = s / peak if (np.isfinite(peak) and peak > 0) else np.nan

        X = env_norm.reshape(M, n * T)

        out_npz = out_dir / f"Sub_{subject:02d}_{suffix}_level_strideparity_full.npz"
        np.savez_compressed(
            out_npz,
            speed=float(speed),
            fs=float(fs),
            muscles=np.array(muscles_intersection, dtype=object),
            ref_peaks=np.array([ref_peaks[m] for m in muscles_intersection], dtype=float),
            parity=int(parity),
            windows=np.array(windows, dtype=int),
            X=X,
            env_norm=env_norm,
            n_strides=int(env_norm.shape[1]),
            n_points=int(env_norm.shape[2]),
            meta_json=json.dumps(
                {
                    "parity0": data["info0"],
                    "parity1": data["info1"],
                    "notch_meta": {m: data["meta_by_muscle"].get(m, {}) for m in muscles_intersection},
                    "drift_meta": {m: data["drift_meta_by_muscle"].get(m, {}) for m in muscles_intersection},
                    "drift_info": data["drift_info"],
                    "snr_meta": {m: data["snr_meta"].get(m, {}) for m in muscles_intersection},
                    "low_snr_flag_speed": data["low_snr_flag"],
                    "muscles_intersection": muscles_intersection,
                }
            ),
        )
        print("SAVED:", out_npz)

        # Update per_speed with intersection-filtered data for stride matching
        per_speed[float(speed)]["env_norm"] = env_norm
        per_speed[float(speed)]["muscles_final"] = muscles_intersection

    # -------------------- Write QC CSV --------------------
    df_qc = pd.DataFrame(qc_rows)

    if qc_path.exists():
        try:
            df_prev = pd.read_csv(qc_path)
            df_prev = df_prev[df_prev["subject"].astype(int) != int(subject)]
            df_out = pd.concat([df_prev, df_qc], ignore_index=True)
        except Exception as e:
            print("[WARN] Could not read existing QC CSV, overwriting. Error:", e)
            df_out = df_qc
    else:
        df_out = df_qc

    df_out.to_csv(qc_path, index=False)
    print("\nQC CSV WRITTEN:", qc_path, "| rows:", len(df_qc))

    # -------------------- Match stride counts across speeds --------------------
    print("\n" + "-" * 80)
    print("MATCHING STRIDE COUNTS ACROSS SPEEDS")
    print("-" * 80)

    counts = {speed: per_speed[speed]["env_norm"].shape[1] for speed in per_speed}
    n_match = min(counts.values())
    print("MATCHED STRIDE COUNT ACROSS SPEEDS:", int(n_match))
    print("COUNTS BEFORE MATCHING:", counts)

    for speed in sorted(per_speed.keys()):
        data = per_speed[speed]
        suffix = data["suffix"]
        env = data["env_norm"]
        n = env.shape[1]

        idx = rng.choice(n, size=n_match, replace=False)
        idx.sort()

        env_m = env[:, idx, :]
        X_m = env_m.reshape(env_m.shape[0], int(n_match) * T)

        out_npz = out_dir / f"Sub_{subject:02d}_{suffix}_level_strideparity_matched.npz"
        np.savez_compressed(
            out_npz,
            speed=float(speed),
            muscles=np.array(muscles_intersection, dtype=object),
            matched_indices=idx.astype(int),
            X=X_m,
            env_norm=env_m,
            n_strides=int(env_m.shape[1]),
            n_points=int(env_m.shape[2]),
        )
        print("SAVED MATCHED:", out_npz)

    print("\n" + "=" * 80)
    print(f"SUBJECT {subject} COMPLETE")
    print(f"  Intersection muscles: {muscles_intersection}")
    print(f"  Matched stride count: {n_match}")
    print("=" * 80)


if __name__ == "__main__":
    main()
