"""Window-level proxy labels with exact top-fraction selection.

Unlike baseline `np.quantile` + `>=` (which inflates the positive rate under
ties), these helpers mark exactly k = max(1, round(n * rate)) windows, with
deterministic tie-breaking: higher amplitude first, then lower window index.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


AmplitudeKind = Literal[
    "close_range",
    "feature_norm_range",
    "abs_log_return",
    "realized_vol",
]


def window_amplitudes(
    x: np.ndarray,
    kind: AmplitudeKind,
    window: int = 30,
    close_col: int = 3,
) -> np.ndarray:
    """Compute one amplitude score per window start index."""
    n = len(x) - window + 1
    if n <= 0:
        raise ValueError(f"Need >= {window} rows, got {len(x)}")
    amps = np.zeros(n, dtype=np.float64)

    if kind == "close_range":
        close = x[:, close_col]
        for i in range(n):
            seg = close[i : i + window]
            amps[i] = float(seg.max() - seg.min())
        return amps

    if kind == "feature_norm_range":
        norms = np.linalg.norm(x, axis=1)
        for i in range(n):
            seg = norms[i : i + window]
            amps[i] = float(seg.max() - seg.min())
        return amps

    if kind == "abs_log_return":
        # Uses close levels; works on raw or positively scaled series if >0.
        close = np.asarray(x[:, close_col], dtype=np.float64)
        # Guard non-positive after min-max: fall back to absolute change.
        for i in range(n):
            seg = close[i : i + window]
            if np.all(seg > 0):
                rets = np.diff(np.log(seg))
                amps[i] = float(np.sum(np.abs(rets)))
            else:
                amps[i] = float(np.sum(np.abs(np.diff(seg))))
        return amps

    if kind == "realized_vol":
        close = np.asarray(x[:, close_col], dtype=np.float64)
        for i in range(n):
            seg = close[i : i + window]
            if np.all(seg > 0):
                rets = np.diff(np.log(seg))
            else:
                rets = np.diff(seg)
            amps[i] = float(np.sqrt(np.sum(rets**2)))
        return amps

    raise ValueError(f"Unknown amplitude kind {kind!r}")


def top_fraction_labels(
    amplitudes: np.ndarray,
    rate: float = 0.01,
) -> Tuple[np.ndarray, dict]:
    """Exact top-`rate` labels with deterministic tie-break.

    Returns
    -------
    labels : bool array
    meta : dict with k, n, base_rate, threshold_amp, n_tied_at_cutoff
    """
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    n = len(amplitudes)
    k = max(1, int(round(n * rate)))
    # lexsort: last key is primary. We want amp desc, then index asc.
    idx = np.arange(n)
    order = np.lexsort((idx, -amplitudes))  # amp high→low, then smaller index
    chosen = order[:k]
    labels = np.zeros(n, dtype=bool)
    labels[chosen] = True

    cutoff_amp = float(amplitudes[chosen[-1]])
    n_tied = int(np.sum(amplitudes == cutoff_amp))
    meta = {
        "n_windows": int(n),
        "k_positives": int(k),
        "base_rate": float(k / n),
        "target_rate": float(rate),
        "cutoff_amplitude": cutoff_amp,
        "n_tied_at_cutoff_amp": n_tied,
        "tie_break": "amplitude_desc_then_index_asc",
    }
    return labels, meta


def baseline_quantile_labels(
    amplitudes: np.ndarray,
    q: float = 0.99,
) -> Tuple[np.ndarray, dict]:
    """Reproduce baseline_v1 rule: amps >= quantile(amps, q)."""
    thr = float(np.quantile(amplitudes, q))
    labels = amplitudes >= thr
    n = len(amplitudes)
    k = int(labels.sum())
    meta = {
        "n_windows": int(n),
        "k_positives": int(k),
        "base_rate": float(k / n) if n else float("nan"),
        "target_rate": float(1.0 - q),
        "cutoff_amplitude": thr,
        "n_tied_at_cutoff_amp": int(np.sum(amplitudes == thr)),
        "tie_break": "quantile_inclusive_geq",
    }
    return labels, meta


def ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    flag_frac: float = 0.05,
) -> dict:
    """Precision, recall, lift, and max achievable precision at a flag budget."""
    n = min(len(scores), len(labels))
    scores = scores[:n]
    labels = labels[:n].astype(bool)
    base_rate = float(labels.mean()) if n else float("nan")
    k = max(1, int(round(n * flag_frac)))
    # Deterministic top-k on scores: score desc, index asc.
    idx = np.arange(n)
    order = np.lexsort((idx, -scores))[:k]
    pred = np.zeros(n, dtype=bool)
    pred[order] = True
    tp = int(np.logical_and(pred, labels).sum())
    precision = tp / k
    denom = int(labels.sum())
    recall = (tp / denom) if denom > 0 else 0.0
    lift = (precision / base_rate) if base_rate > 0 else float("nan")
    # With k flags and P positives, max precision is min(1, P/k).
    max_precision = min(1.0, denom / k) if k > 0 else float("nan")
    return {
        "n": int(n),
        "flag_frac": float(flag_frac),
        "k_flags": int(k),
        "n_positives": int(denom),
        "base_rate": base_rate,
        "precision": float(precision),
        "recall": float(recall),
        "lift": float(lift),
        "max_achievable_precision": float(max_precision),
        "tp": int(tp),
    }
