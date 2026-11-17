"""Shared scoring protocol used to build the main comparison tables."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

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
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if n < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_run(
    model: str,
    level: str,
    scores: np.ndarray,
    train_seconds: float,
    infer_seconds: float,
    seed: int,
    precision_5: Optional[float] = None,
    recall_5: Optional[float] = None,
    precision_10: Optional[float] = None,
    recall_10: Optional[float] = None,
    backend: Optional[str] = None,
) -> Dict[str, float | str | int]:
    flags_5 = flag_top_quantile(scores, 0.05)
    flags_10 = flag_top_quantile(scores, 0.10)
    row: Dict[str, float | str | int] = {
        "model": model,
        "data_level": level,
        "seed": int(seed),
        "n_windows": int(len(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "flag_rate_5pct": anomaly_rate(flags_5),
        "flag_rate_10pct": anomaly_rate(flags_10),
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }
    if precision_5 is not None:
        row["precision_at_5pct"] = float(precision_5)
        row["recall_at_5pct"] = float(recall_5 if recall_5 is not None else np.nan)
    if precision_10 is not None:
        row["precision_at_10pct"] = float(precision_10)
        row["recall_at_10pct"] = float(recall_10 if recall_10 is not None else np.nan)
    if backend is not None:
        row["backend"] = backend
    return row


def aggregate_seed_rows(rows: List[Dict]) -> pd.DataFrame:
    """Mean and std across seeds for numeric columns, grouped by model/level."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keys = ["model", "data_level"]
    num_cols = [
        c
        for c in df.columns
        if c not in keys + ["seed", "backend"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    out_rows = []
    for (model, level), g in df.groupby(keys, sort=False):
        item = {"model": model, "data_level": level, "n_seeds": len(g)}
        if "backend" in g.columns:
            item["backend"] = ",".join(sorted(set(map(str, g["backend"].dropna()))))
        for c in num_cols:
            item[f"{c}_mean"] = float(g[c].mean())
            item[f"{c}_std"] = float(g[c].std(ddof=0))
        out_rows.append(item)
    return pd.DataFrame(out_rows)


def pairwise_jaccard_table(
    flag_map: Dict[str, np.ndarray],
) -> pd.DataFrame:
    names = list(flag_map.keys())
    mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = jaccard(flag_map[a], flag_map[b])
    return pd.DataFrame(mat, index=names, columns=names)


def rows_to_table(rows: Iterable[Dict]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def markdown_table(df: pd.DataFrame, float_cols: Optional[List[str]] = None) -> str:
    out = df.copy()
    if float_cols is None:
        float_cols = [c for c in out.columns if out[c].dtype.kind in "fc"]
    for c in float_cols:
        out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in out.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])
