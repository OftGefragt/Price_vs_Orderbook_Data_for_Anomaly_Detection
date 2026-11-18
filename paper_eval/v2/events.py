"""Merge overlapping windows into events; event-level metrics + blocked bootstrap."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def merge_windows_to_events(
    flags: np.ndarray,
    gap: int = 30,
) -> List[Tuple[int, int]]:
    """Merge True windows into [start, end] inclusive index events.

    Windows i and j merge if they overlap or the gap between them is < `gap`
    (distance between starts <= window_len-1 is overlap; we parameterize by
    start-index gap tolerance: merge if next_start <= prev_end + gap).
    Here flags are on window-start indices; `gap` is in window-start units.
    Default gap=30 matches "within ~one window length".
    """
    idx = np.flatnonzero(np.asarray(flags, dtype=bool))
    if len(idx) == 0:
        return []
    events: List[Tuple[int, int]] = []
    start = end = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i <= end + gap:
            end = i
        else:
            events.append((start, end))
            start = end = i
    events.append((start, end))
    return events


def event_sets(flags: np.ndarray, gap: int = 30) -> set:
    """Represent events by frozenset of covered window indices for Jaccard."""
    evs = merge_windows_to_events(flags, gap=gap)
    # For set metrics we use the interval midpoint or the whole span as identity.
    # Conservative event identity: (start, end) tuples.
    return set(evs)


def event_precision_recall(
    score_flags: np.ndarray,
    label_flags: np.ndarray,
    gap: int = 30,
) -> Dict[str, float]:
    """Event-level precision/recall via interval overlap.

    A predicted event hits a label event if their [start,end] intervals overlap
    (inclusive) after expanding each by 0 (windows already overlap-aware).
    """
    pred = merge_windows_to_events(score_flags, gap=gap)
    truth = merge_windows_to_events(label_flags, gap=gap)

    def overlaps(a, b) -> bool:
        return not (a[1] < b[0] or b[1] < a[0])

    tp = 0
    matched_truth = set()
    for p in pred:
        for ti, t in enumerate(truth):
            if ti in matched_truth:
                continue
            if overlaps(p, t):
                tp += 1
                matched_truth.add(ti)
                break
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    # Event Jaccard on matched coverage of truth∪pred using overlap graph is awkward;
    # report interval-set Jaccard on expanded window membership instead.
    def cover(evs):
        s = set()
        for a, b in evs:
            s.update(range(a, b + 1))
        return s

    pc, tc = cover(pred), cover(truth)
    inter = len(pc & tc)
    union = len(pc | tc)
    jacc = inter / union if union else 0.0
    return {
        "n_pred_events": float(len(pred)),
        "n_label_events": float(len(truth)),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_jaccard_windowcover": float(jacc),
        "tp_events": float(tp),
    }


def blocked_bootstrap_ci(
    values_by_block: Sequence[float],
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Mean and percentile CI from resampling block-level summaries."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values_by_block, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(arr)), float(lo), float(hi)


def event_level_bootstrap(
    score_flags: np.ndarray,
    label_flags: np.ndarray,
    gap: int = 30,
    block_len: int = 30,
    n_boot: int = 400,
    seed: int = 42,
) -> Dict[str, float]:
    """Blocked bootstrap on window axis for event precision/recall."""
    n = min(len(score_flags), len(label_flags))
    score_flags = score_flags[:n]
    label_flags = label_flags[:n]
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_len)
    # Pre-split into contiguous blocks
    blocks_s = [
        score_flags[i * block_len : (i + 1) * block_len] for i in range(n_blocks)
    ]
    blocks_l = [
        label_flags[i * block_len : (i + 1) * block_len] for i in range(n_blocks)
    ]
    # drop leftover tail for clean blocks (conservative)
    precs, recalls, jaccs = [], [], []
    for _ in range(n_boot):
        choice = rng.integers(0, n_blocks, size=n_blocks)
        sf = np.concatenate([blocks_s[i] for i in choice])
        lf = np.concatenate([blocks_l[i] for i in choice])
        m = event_precision_recall(sf, lf, gap=gap)
        precs.append(m["event_precision"])
        recalls.append(m["event_recall"])
        jaccs.append(m["event_jaccard_windowcover"])
    point = event_precision_recall(score_flags, label_flags, gap=gap)
    out = dict(point)
    for name, arr in [
        ("event_precision", precs),
        ("event_recall", recalls),
        ("event_jaccard_windowcover", jaccs),
    ]:
        lo, hi = np.quantile(arr, [0.025, 0.975])
        out[f"{name}_ci95_lo"] = float(lo)
        out[f"{name}_ci95_hi"] = float(hi)
    out["n_boot"] = float(n_boot)
    out["block_len"] = float(block_len)
    out["gap"] = float(gap)
    return out
