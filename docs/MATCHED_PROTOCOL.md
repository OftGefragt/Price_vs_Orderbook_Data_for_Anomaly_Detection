# Matched evaluation protocol

This protocol removes two confounds from the original strong run:

1. **Resolution.** Level-1 1-minute bars are aggregated to 5-minute OHLCV so
   each window covers a comparable clock span to Level-2 order-book snapshots.
2. **Forest backend.** Both tiers use the same `sklearn.ensemble.IsolationForest`
   implementation (same tree count, subsample size, contamination). The earlier
   `auto` rule routed large Level-1 matrices to sklearn and Level-2 to pure-Python
   EIF, which mixed detector code paths with input type.

## Defaults

| Setting | Value |
|---|---|
| Seeds | 42, 43, 44 |
| Level-1 bar | 5-minute aggregate of bundled BTC/ETH 1-minute series |
| Level-1 train / test | full aggregated series (no subsample) |
| Forest backend | `sklearn` on both L1 and L2 |
| f-AnoGAN / GRU epochs | same as strong (10 / 15 / 8) |
| Window / proxy quantile | 30 / top 1% amplitude |

## Commands

```bash
# Resolution-matched L1 vs L2 on spring 2021 (bundled LOB)
python -m paper_eval.run_comparison --mode matched --regime 2021_spring

# Quick smoke
python -m paper_eval.run_comparison --mode matched_fast --regime 2021_spring

# Second regime: 2022 bear (Level-1 only; fetch once first)
python -m paper_eval.fetch_regime_ohlcv --regime 2022_bear
python -m paper_eval.run_comparison --mode matched --regime 2022_bear --levels L1
```

Outputs land in `results/matched_<regime>/` (or `results/matched_fast_<regime>/`).

## What 2022_bear covers

May–June 2022 BTC 5-minute train and a two-week ETH test window downloaded from
[Binance Vision](https://data.binance.vision/). Historical Level-2 order-book
features for that window are **not** bundled, so the regime run is Level-1 only
and tests whether detector ranking quality generalises beyond spring 2021 price
data. A full L1-vs-L2 claim for 2022 requires adding LOB CSVs under
`data/regimes/2022_bear/` and extending `REGIMES` in `paper_eval/data.py`.

## How to cite in the paper

Report Table 5 from the original strong protocol as the exploratory baseline,
then report the matched table as the confound-controlled comparison. State that
Precision@k / Recall@k still use weak proxy stress labels.
