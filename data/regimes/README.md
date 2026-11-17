# Secondary market regimes

OHLCV for regimes beyond the bundled spring 2021 window.

| Regime | Source | Bar | L2 LOB |
|---|---|---|---|
| `2022_bear` | Binance Vision monthly klines | 5-minute | not bundled |

Fetch or refresh:

```bash
python -m paper_eval.fetch_regime_ohlcv --regime 2022_bear
```
