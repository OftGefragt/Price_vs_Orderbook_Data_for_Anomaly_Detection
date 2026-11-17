# Strong evaluation protocol

This is the protocol intended for the paper numbers. Do not replace it with
the fast smoke settings when writing results.

## Design goals

1. Same three model families on Level-1 and Level-2 inputs.
2. Multiple random seeds for deep models and forest initialisation.
3. Shared top-quantile flagging so raw flag rates are comparable.
4. Additional Precision@k / Recall@k against weak proxy stress labels.
5. Pairwise flag agreement (Jaccard) within each data level.
6. Explicit logging of every runtime choice in `results/protocol.json`.

## Strong defaults

| Setting | Value |
|---|---|
| Seeds | 42, 43, 44 |
| f-AnoGAN GAN epochs | 10 |
| f-AnoGAN encoder epochs | 15 |
| GRU-DeepSVDD epochs | 8 |
| Level-1 train rows | 30000 (subsample of BTC) |
| Level-1 test rows | full ETH test set |
| Level-2 | full 30-feature BTC/ETH matrices |
| Window | 30 |
| Proxy stress quantile | top 1% amplitude windows |

## Why Level-1 train is capped

The full BTC minute series has 87821 rows. Training f-AnoGAN for many epochs
on the full set is too slow for a single Colab session. The strong protocol
keeps the **full ETH test set** and a large BTC train subsample, and records
the exact count in `protocol.json`.

## EIF backend rule

Pure-Python Extended Isolation Forest is used when the training matrix has at
most 8000 rows (Level-2). Larger Level-1 matrices use sklearn IsolationForest
with the same tree count, subsample size, and contamination. The backend name
is written into the results table.

## Proxy labels (important honesty note)

Precision and Recall are computed against **weak proxy stress labels**:

- Level-1: windows with the largest Close amplitude
- Level-2: windows with the largest multivariate feature-norm amplitude

These are not exchange-confirmed manipulation labels. They measure whether
high anomaly scores concentrate on extreme market moves. State this clearly
in the paper Limitations section.

## Command

```bash
python -m paper_eval.run_comparison --mode strong
```

Fast debugging:

```bash
python -m paper_eval.run_comparison --mode fast
```
