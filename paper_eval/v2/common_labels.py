"""Task 2: non-circular price-based labels shared in spirit across L1 and L2.

L2 GA-EIF feature matrices have no price column. The AnoGAN ETH_5min file
contains `midpoint` and the same 30 feature columns with values identical to
`eth_testdata.csv` (verified max abs diff ~1e-14). BTC AnoGAN train file is
empty in this repo, so train-side mid-price is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from paper_eval.data import REPO_ROOT, load_level1, load_level2, scale_fit_transform
from paper_eval.v2.labels import top_fraction_labels, window_amplitudes


ANOGAN_ETH = (
    REPO_ROOT
    / "Level 2 Analysis"
    / "Anomaly-Detection-in-Cryptocurrency-Markets-using-1D-Convolutional-AnoGAN-main"
    / "data"
    / "test"
    / "ETH_5min.csv"
)


def load_l2_with_midpoint() -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Return (l2_features, midpoint, timestamps) for ETH test, row-aligned."""
    l2_tr, l2_te = load_level2()
    an = pd.read_csv(ANOGAN_ETH)
    feats = list(l2_te.columns)
    missing = [c for c in feats if c not in an.columns]
    if missing:
        raise RuntimeError(f"AnoGAN file missing L2 features: {missing}")
    a = an[feats].astype(float).to_numpy()
    b = l2_te[feats].astype(float).to_numpy()
    if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
        raise RuntimeError("AnoGAN ETH_5min features are not identical to L2 test matrix")
    mid = an["midpoint"].astype(float)
    ts = pd.to_datetime(an["system_time"], utc=True)
    return l2_te, mid, pd.DatetimeIndex(ts)


def price_panel_from_series(price: np.ndarray) -> np.ndarray:
    """Fake OHLCV-shaped panel with Close=price for amplitude helpers."""
    # columns Open High Low Close Volume — only Close used for price labels
    n = len(price)
    x = np.zeros((n, 5), dtype=np.float64)
    x[:, 3] = price
    return x


def build_task2_label_bundle(rate: float = 0.01, window: int = 30) -> Dict:
    """Build primary / secondary / alternative labels for L1 (5-min) and L2."""
    # --- L1 5-min Close (raw levels for price amplitudes; scaling not required) ---
    l1_tr, l1_te = load_level1(bar="5min")
    l1_close = l1_te["Close"].to_numpy(dtype=np.float64)
    l1_price_x = price_panel_from_series(l1_close)

    primary_l1_amps = window_amplitudes(l1_price_x, "close_range", window=window)
    alt_l1_amps = window_amplitudes(l1_price_x, "abs_log_return", window=window)
    # secondary L1: baseline-style on scaled features (legacy)
    _, l1_scaled, _ = scale_fit_transform(l1_tr, l1_te)
    secondary_l1_amps = window_amplitudes(l1_scaled, "close_range", window=window)

    # --- L2 midpoint from AnoGAN ---
    l2_te, mid, ts = load_l2_with_midpoint()
    l2_tr, _ = load_level2()
    l2_price_x = price_panel_from_series(mid.to_numpy())
    primary_l2_amps = window_amplitudes(l2_price_x, "close_range", window=window)
    alt_l2_amps = window_amplitudes(l2_price_x, "abs_log_return", window=window)
    _, l2_scaled, _ = scale_fit_transform(l2_tr, l2_te)
    secondary_l2_amps = window_amplitudes(l2_scaled, "feature_norm_range", window=window)

    out = {
        "window": window,
        "rate": rate,
        "l2_midpoint_source": str(ANOGAN_ETH.relative_to(REPO_ROOT)),
        "l2_btc_midpoint_available": False,
        "notes": [
            "Primary label uses raw Close (L1) or mid-price (L2); not model input features for L2.",
            "L2 primary is non-circular w.r.t. the 30 book features.",
            "Strict bar-by-bar timestamp alignment between L1 5-min aggregate and L2 system_time is partial; see alignment table.",
        ],
        "labels": {},
    }

    for level, amps_map in {
        "L1": {
            "primary_price_range": primary_l1_amps,
            "secondary_legacy": secondary_l1_amps,
            "alternative_abs_log_return": alt_l1_amps,
        },
        "L2": {
            "primary_price_range": primary_l2_amps,
            "secondary_legacy": secondary_l2_amps,
            "alternative_abs_log_return": alt_l2_amps,
        },
    }.items():
        out["labels"][level] = {}
        for name, amps in amps_map.items():
            lab, meta = top_fraction_labels(amps, rate=rate)
            out["labels"][level][name] = {"labels": lab, "amplitudes": amps, "meta": meta}

    # Timestamp alignment attempt: floor L2 times to 5-min and intersect with L1 index
    l1_idx = l1_te.index
    l2_floor = ts.floor("5min")
    common = l1_idx.intersection(pd.DatetimeIndex(l2_floor.unique()))
    out["alignment"] = {
        "l1_5min_bars": int(len(l1_idx)),
        "l2_bars": int(len(ts)),
        "l2_unique_5min_floors": int(l2_floor.nunique()),
        "intersection_5min_timestamps": int(len(common)),
        "l2_time_start": str(ts[0]),
        "l2_time_end": str(ts[-1]),
        "l1_time_start": str(l1_idx[0]),
        "l1_time_end": str(l1_idx[-1]),
    }
    return out
