"""Shared scoring protocol used to build the main comparison table."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def flag_top_quantile(scores: np.ndarray, q: float) -> np.ndarray:
    """Flag the highest scoring fraction q of observations (q in (0, 1])."""
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1].")
    threshold = np.quantile(scores, 1.0 - q)
    return scores >= threshold


def anomaly_rate(flags: np.ndarray) -> float:
    return float(np.mean(flags.astype(np.float64)))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def summarize_run(
    model: str,
    level: str,
    scores: np.ndarray,
    train_seconds: float,
    infer_seconds: float,
    reference_flags_5: np.ndarray | None = None,
) -> Dict[str, float | str]:
    flags_5 = flag_top_quantile(scores, 0.05)
    flags_10 = flag_top_quantile(scores, 0.10)
    row: Dict[str, float | str] = {
        "model": model,
        "data_level": level,
        "n_windows": int(len(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "flag_rate_5pct": anomaly_rate(flags_5),
        "flag_rate_10pct": anomaly_rate(flags_10),
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }
    if reference_flags_5 is not None:
        # Align by truncating to the shorter length when window counts differ.
        n = min(len(flags_5), len(reference_flags_5))
        row["jaccard_vs_ref_5pct"] = jaccard(flags_5[:n], reference_flags_5[:n])
    else:
        row["jaccard_vs_ref_5pct"] = 1.0
    return row


def rows_to_table(rows: Iterable[Dict[str, float | str]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    preferred = [
        "model",
        "data_level",
        "n_windows",
        "flag_rate_5pct",
        "flag_rate_10pct",
        "jaccard_vs_ref_5pct",
        "score_mean",
        "score_std",
        "train_seconds",
        "infer_seconds",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[cols]


def markdown_table(df: pd.DataFrame, float_cols: List[str] | None = None) -> str:
    out = df.copy()
    float_cols = float_cols or [
        c for c in out.columns if out[c].dtype.kind in "fc"
    ]
    for c in float_cols:
        out[c] = out[c].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in out.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])
