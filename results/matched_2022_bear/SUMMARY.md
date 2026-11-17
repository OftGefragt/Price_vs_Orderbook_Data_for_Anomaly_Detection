# Matched 2022 bear: Level-1 only

- Source: Binance Vision BTCUSDT / ETHUSDT 5m (May–June 2022)
- Train: BTC 2022-05-01 .. 2022-06-30 (17568 bars)
- Test: ETH 2022-06-10 .. 2022-06-23 (4032 bars)
- L2: not available in this release

| Model | P@5 | R@5 |
|---|---:|---:|
| f-AnoGAN | 0.205 | 1.000 |
| GRU-DeepSVDD | 0.158 | 0.772 |
| GA-EIF+SVM | 0.015 | 0.073 |

Deep L1 detectors still rank extreme amplitude windows in a bear regime; the L1 forest remains weak.
