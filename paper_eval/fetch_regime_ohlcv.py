"""Download Binance Vision 5-min OHLCV for secondary market regimes.

No API key required. Uses public monthly kline archives from
https://data.binance.vision/

Example:
  python -m paper_eval.fetch_regime_ohlcv --regime 2022_bear
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

KLINE_COLS = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore",
]

REGIME_SPECS = {
    "2022_bear": {
        # Luna / cascading deleveraging window.
        "months": [("2022", "05"), ("2022", "06")],
        "btc_start": "2022-05-01",
        "btc_end": "2022-06-30",
        "eth_start": "2022-06-10",
        "eth_end": "2022-06-23",
        "out_dir": REPO_ROOT / "data" / "regimes" / "2022_bear",
    },
}


def _month_url(symbol: str, interval: str, year: str, month: str) -> str:
    name = f"{symbol}-{interval}-{year}-{month}.zip"
    return f"{BASE_URL}/{symbol}/{interval}/{name}"


def _download_month(symbol: str, interval: str, year: str, month: str) -> pd.DataFrame:
    url = _month_url(symbol, interval, year, month)
    print(f"Fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = [n for n in zf.namelist() if n.endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV inside {url}")
        with zf.open(members[0]) as fh:
            df = pd.read_csv(fh, header=None, names=KLINE_COLS)
    return df


def fetch_symbol_months(
    symbol: str,
    months: Iterable[tuple],
    interval: str = "5m",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for year, month in months:
        frames.append(_download_month(symbol, interval, year, month))
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Open time"]).sort_values("Open time")
    out["Date"] = pd.to_datetime(out["Open time"], unit="ms", utc=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = out[col].astype(float)
    return out[["Date", "Open", "High", "Low", "Close", "Volume", "Open time"]]


def _slice_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["Date"] >= pd.Timestamp(start, tz="UTC")) & (
        df["Date"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    )
    return df.loc[mask].copy()


def fetch_regime(regime: str) -> Path:
    if regime not in REGIME_SPECS:
        raise ValueError(f"Unknown regime {regime}. Choose from {list(REGIME_SPECS)}")
    spec = REGIME_SPECS[regime]
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    btc = fetch_symbol_months("BTCUSDT", spec["months"])
    eth = fetch_symbol_months("ETHUSDT", spec["months"])
    btc = _slice_dates(btc, spec["btc_start"], spec["btc_end"])
    eth = _slice_dates(eth, spec["eth_start"], spec["eth_end"])

    btc_path = out_dir / "BTC_5min.csv"
    eth_path = out_dir / "ETH_5min.csv"
    btc.to_csv(btc_path, index=False)
    eth.to_csv(eth_path, index=False)

    meta = {
        "regime": regime,
        "btc_rows": int(len(btc)),
        "eth_rows": int(len(eth)),
        "btc_start": str(btc["Date"].iloc[0]) if len(btc) else None,
        "btc_end": str(btc["Date"].iloc[-1]) if len(btc) else None,
        "eth_start": str(eth["Date"].iloc[0]) if len(eth) else None,
        "eth_end": str(eth["Date"].iloc[-1]) if len(eth) else None,
        "source": "https://data.binance.vision/",
        "interval": "5m",
    }
    (out_dir / "manifest.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote {btc_path} ({len(btc)} rows)")
    print(f"Wrote {eth_path} ({len(eth)} rows)")
    print(f"Wrote {out_dir / 'manifest.json'}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Fetch regime OHLCV from Binance Vision.")
    parser.add_argument(
        "--regime",
        default="2022_bear",
        choices=list(REGIME_SPECS.keys()),
    )
    args = parser.parse_args()
    fetch_regime(args.regime)


if __name__ == "__main__":
    main()
