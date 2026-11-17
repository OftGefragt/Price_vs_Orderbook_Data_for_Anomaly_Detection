"""Dataset loaders for Level-1 OHLCV and Level-2 order-book inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


REPO_ROOT = Path(__file__).resolve().parents[1]

L1_TRAIN = REPO_ROOT / "Level 1 Analysis" / "data" / "train" / "BTC_1min.csv"
L1_TEST = REPO_ROOT / "Level 1 Analysis" / "data" / "test" / "ETH_1min.csv"
L2_TRAIN = (
    REPO_ROOT
    / "Level 2 Analysis"
    / "GA-optimized-EIF-SVM-for-anomaly-detection-main"
    / "data"
    / "train"
    / "btc_traindata.csv"
)
L2_TEST = (
    REPO_ROOT
    / "Level 2 Analysis"
    / "GA-optimized-EIF-SVM-for-anomaly-detection-main"
    / "data"
    / "test"
    / "eth_testdata.csv"
)

L1_FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Named evaluation regimes. L2 paths are None when order-book archives are unavailable.
REGIMES = {
    "2021_spring": {
        "description": "BTC train / ETH test, spring 2021 bull window (bundled)",
        "l1_train": L1_TRAIN,
        "l1_test": L1_TEST,
        "l2_train": L2_TRAIN,
        "l2_test": L2_TEST,
        "has_l2": True,
    },
    "2022_bear": {
        "description": (
            "BTC train / ETH test, May-Jun 2022 bear (Luna cascade). "
            "Level-1 5-min OHLCV from Binance Vision; Level-2 LOB not bundled."
        ),
        "l1_train": REPO_ROOT / "data" / "regimes" / "2022_bear" / "BTC_5min.csv",
        "l1_test": REPO_ROOT / "data" / "regimes" / "2022_bear" / "ETH_5min.csv",
        "l2_train": None,
        "l2_test": None,
        "has_l2": False,
        "native_bar": "5min",
    },
}


def aggregate_ohlcv(
    df: pd.DataFrame,
    rule: str = "5min",
) -> pd.DataFrame:
    """Resample OHLCV to a coarser bar size (e.g. 1-min -> 5-min)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("aggregate_ohlcv expects a DatetimeIndex.")
    agg = (
        df.resample(rule, label="right", closed="right")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(how="any")
    )
    return agg.astype(float)


def _read_l1_csv(path: Path) -> pd.DataFrame:
    train = pd.read_csv(path)
    # Binance Vision exports use Open time; bundled files use Date.
    if "Date" in train.columns:
        train["Date"] = pd.to_datetime(train["Date"], utc=True)
        train = train.set_index("Date").sort_index()
    elif "Open time" in train.columns:
        train["Date"] = pd.to_datetime(train["Open time"], unit="ms", utc=True)
        train = train.set_index("Date").sort_index()
    else:
        raise ValueError(f"No Date / Open time column in {path}")
    missing = [c for c in L1_FEATURE_COLS if c not in train.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return train[L1_FEATURE_COLS].astype(float)


def load_level1(
    bar: str = "1min",
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Level-1 OHLCV. bar='5min' aggregates bundled 1-min series."""
    train_path = Path(train_path) if train_path else L1_TRAIN
    test_path = Path(test_path) if test_path else L1_TEST
    train = _read_l1_csv(train_path)
    test = _read_l1_csv(test_path)

    if bar == "1min":
        return train, test
    if bar == "5min":
        # Regime CSVs already stored as 5-min do not need re-aggregation.
        # Detect by median spacing (minutes).
        def _needs_agg(df: pd.DataFrame) -> bool:
            if len(df) < 3:
                return True
            delta = df.index.to_series().diff().dropna().dt.total_seconds().median()
            return float(delta) < 200  # < ~3.3 min => treat as 1-min

        if _needs_agg(train):
            train = aggregate_ohlcv(train, "5min")
        if _needs_agg(test):
            test = aggregate_ohlcv(test, "5min")
        return train, test
    raise ValueError(f"Unsupported bar={bar!r}; use '1min' or '5min'.")


def load_level2(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(train_path) if train_path else L2_TRAIN
    test_path = Path(test_path) if test_path else L2_TEST
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feature_cols = [c for c in train.columns if c != "Unnamed: 0"]
    train = train[feature_cols].astype(float)
    test = test[feature_cols].astype(float)
    # Synthetic monotonic index so window scores can be aligned by position.
    train.index = pd.RangeIndex(start=0, stop=len(train), name="row")
    test.index = pd.RangeIndex(start=0, stop=len(test), name="row")
    return train, test


def scale_fit_transform(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    x_train = scaler.fit_transform(train.values)
    x_test = scaler.transform(test.values)
    return x_train, x_test, scaler


def sliding_windows(arr: np.ndarray, window: int = 30, step: int = 1) -> np.ndarray:
    if len(arr) < window:
        raise ValueError(f"Need at least {window} rows, received {len(arr)}.")
    out = []
    for start in range(0, len(arr) - window + 1, step):
        out.append(arr[start : start + window])
    return np.stack(out, axis=0)
