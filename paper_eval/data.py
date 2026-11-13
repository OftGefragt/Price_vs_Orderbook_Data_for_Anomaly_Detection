"""Dataset loaders for Level-1 OHLCV and Level-2 order-book inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

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


def load_level1() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(L1_TRAIN)
    test = pd.read_csv(L1_TEST)
    train["Date"] = pd.to_datetime(train["Date"], utc=True)
    test["Date"] = pd.to_datetime(test["Date"], utc=True)
    train = train.set_index("Date").sort_index()
    test = test.set_index("Date").sort_index()
    return train[L1_FEATURE_COLS].astype(float), test[L1_FEATURE_COLS].astype(float)


def load_level2() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(L2_TRAIN)
    test = pd.read_csv(L2_TEST)
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
