from __future__ import annotations
import numpy as np


def classify_run(history, burn_in_frac=0.5, runaway_threshold=0.45,
                 osc_amp_threshold=0.15, osc_std_threshold=0.06):
    r = np.asarray(history.get('R', []), dtype=float)
    if len(r) == 0:
        return 'unknown'
    burn = int(len(r) * burn_in_frac)
    tail = r[burn:] if burn < len(r) else r
    max_r = float(r.max())
    amp_tail = float(tail.max() - tail.min()) if len(tail) else 0.0
    std_tail = float(tail.std()) if len(tail) else 0.0
    if max_r > runaway_threshold:
        return 'runaway'
    if amp_tail > osc_amp_threshold or std_tail > osc_std_threshold:
        return 'oscillatory'
    return 'stable'


def classify_run_adaptive(history, burn_in_frac=0.5):
    """Regime classifier using spectral analysis and adaptive thresholds."""
    r = np.asarray(history.get('R', []), dtype=float)
    if len(r) < 10:
        return 'unknown'
    burn = int(len(r) * burn_in_frac)
    tail = r[burn:] if burn < len(r) else r
    max_r = float(r.max())
    mean_tail = float(tail.mean())
    std_tail = float(tail.std()) if len(tail) else 0.0
    amp_tail = float(tail.max() - tail.min()) if len(tail) else 0.0

    if max_r > 0.40:
        return 'runaway'

    if len(tail) >= 16:
        detrended = tail - np.mean(tail)
        fft = np.fft.rfft(detrended)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(detrended))
        if len(power) > 1:
            dc_removed = power[1:]
            peak_power = float(dc_removed.max()) if len(dc_removed) > 0 else 0.0
            total_power = float(dc_removed.sum()) if len(dc_removed) > 0 else 1.0
            spectral_ratio = peak_power / total_power if total_power > 0 else 0.0
            if spectral_ratio > 0.25 and amp_tail > 0.08:
                return 'oscillatory'

    if amp_tail > 0.12 or std_tail > 0.04:
        return 'oscillatory'

    return 'stable'


def summarize_history(history):
    out = {}
    for key, vals in history.items():
        arr = np.asarray(vals, dtype=float) if len(vals) else np.asarray([])
        if arr.size == 0:
            continue
        burn = int(0.5 * len(arr))
        tail = arr[burn:]
        out[f'{key}_mean_tail'] = float(tail.mean())
        out[f'{key}_max'] = float(arr.max())
        out[f'{key}_amp_tail'] = float(tail.max() - tail.min())
        out[f'{key}_std_tail'] = float(tail.std())
    out['regime_fixed'] = classify_run(history)
    out['regime'] = classify_run_adaptive(history)
    return out
