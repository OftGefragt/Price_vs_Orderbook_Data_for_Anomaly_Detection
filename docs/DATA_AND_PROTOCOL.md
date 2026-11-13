# Data and evaluation protocol

## Level 1 (price and volume)

| Split | Asset | File | Rows | Columns used |
|---|---|---|---:|---|
| Train | BTC | `Level 1 Analysis/data/train/BTC_1min.csv` | 87821 | Open, High, Low, Close, Volume |
| Test | ETH | `Level 1 Analysis/data/test/ETH_1min.csv` | 18721 | Open, High, Low, Close, Volume |

Timestamps cover late March to May 2021 for training and early to mid April 2021 for testing.

## Level 2 (order book)

| Split | Asset | File | Rows | Feature count |
|---|---|---|---:|---:|
| Train | BTC | `Level 2 Analysis/.../data/train/btc_traindata.csv` | 3367 | 30 |
| Test | ETH | `Level 2 Analysis/.../data/test/eth_testdata.csv` | 3365 | 30 |

The thirty features are a fixed subset of bid and ask distance, notional, limit, market, and cancel statistics. The raw Level-2 extracts in the AnoGAN folders contain a wider schema with fifteen depth levels. The paper comparison standardizes on the thirty-feature matrix so that tree-based and sequence models share one input definition.

## Preprocessing

1. Fit a `MinMaxScaler` on the training split only.
2. Transform the test split with that scaler.
3. For sequence models, build sliding windows of length 30 with step 1.

Level-1 experiments may subsample training and test rows for runtime control. The Colab defaults are 12000 training rows and 8000 test rows. Report the values used when writing the paper.

## Scoring rule

Let `s_i` be the anomaly score of window `i`. For quantile level `q` in `{0.05, 0.10}`:

```text
flag_i = 1 if s_i >= quantile_{1-q}(s)
```

This rule equalizes the flag rate across detectors. It is intended for fair comparison of ranking behaviour, not as a claim that 5 percent of the market is anomalous.

## Cross-asset design

All models train on Bitcoin and score Ethereum. The shift is intentional. It tests whether patterns learned on one asset transfer to another under the same feature family.
