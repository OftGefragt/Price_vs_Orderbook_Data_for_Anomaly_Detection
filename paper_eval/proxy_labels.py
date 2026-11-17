"""Weak proxy stress labels derived from the test series itself.

These are not verified manipulation labels. They mark extreme market
moves so that ranking quality can be reported with Precision@k and
Recall@k in addition to unsupervised agreement metrics.
"""

from __future__ import annotations

import numpy as np


def level1_stress_flags(
    x_test: np.ndarray,
    window: int = 30,
    q: float = 0.99,
) -> np.ndarray:
    """Flag windows with extreme Close log-return amplitude.

    Assumes columns are [Open, High, Low, Close, Volume].
    """
    close = x_test[:, 3]
    # Avoid log of zero after min-max scaling by using absolute diffs on scaled close.
    # Use relative step size on Close within each window.
    n = len(close) - window + 1
    amps = np.zeros(n, dtype=np.float64)
    for i in range(n):
        seg = close[i : i + window]
        amps[i] = float(np.max(seg) - np.min(seg))
    thr = np.quantile(amps, q)
    return amps >= thr


def level2_stress_flags(
    x_test: np.ndarray,
    window: int = 30,
    q: float = 0.99,
) -> np.ndarray:
    """Flag windows with extreme multivariate amplitude on order-book features."""
    # Row-wise L2 norm of feature vector, then amplitude inside the window.
    norms = np.linalg.norm(x_test, axis=1)
    n = len(norms) - window + 1
    amps = np.zeros(n, dtype=np.float64)
    for i in range(n):
        seg = norms[i : i + window]
        amps[i] = float(np.max(seg) - np.min(seg))
    thr = np.quantile(amps, q)
    return amps >= thr


def precision_recall_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k_frac: float = 0.05,
) -> tuple[float, float, int]:
    """Precision/Recall when predicting the top k_frac scores as anomalies."""
    n = min(len(scores), len(labels))
    scores = scores[:n]
    labels = labels[:n].astype(bool)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(scores)[::-1][:k]
    pred = np.zeros(n, dtype=bool)
    pred[order] = True
    tp = int(np.logical_and(pred, labels).sum())
    precision = tp / k
    denom = int(labels.sum())
    recall = (tp / denom) if denom > 0 else 0.0
    return float(precision), float(recall), k
