# Methods used in the unified comparison

## f-AnoGAN (one-dimensional)

An encoder maps each window to a latent code. A generator reconstructs the window. A discriminator provides feature embeddings. The anomaly score combines reconstruction error and feature-matching error:

```text
score = 1.0 * reconstruction_error + 0.1 * feature_matching_error
```

The Colab implementation uses a compact convolutional stack so that short training schedules remain practical. Architecture files in `Level 1 Analysis/` and `Level 2 Analysis/` contain the earlier course versions and pretrained weights.

## GA-EIF with one-class SVM

Extended Isolation Forest scores each point with randomly oriented hyperplanes. Hyperparameters follow the genetic search recorded in the Level-2 notebook:

| Parameter | Value |
|---|---:|
| Number of trees | 235 |
| Sample size | 1691 (capped by training size when needed) |
| Contamination | 0.062 |
| Extension level | 4 on Level 1, 26 on Level 2 when feature dimension allows |

A one-class SVM is fit on training scores for compatibility with the original pipeline. Ranking for Table 5 uses the continuous forest scores.

If the local `eif` package fails to import, the runner falls back to `sklearn.ensemble.IsolationForest` with the same tree and subsample settings. The backend name is written to the results table.

## GRU with Deep SVDD loss

A two-layer GRU encodes each window. Training minimizes the squared distance from each embedding to a hypersphere center. The center is updated toward the batch mean after every step. The anomaly score on test windows is the squared distance to the learned center.

## Relation to the original course folders

| Unified module | Original material |
|---|---|
| `paper_eval/models_anogan.py` | AnoGAN packages under Level 1 and Level 2 |
| `paper_eval/models_eif.py` | GA-EIF packages and `eif/eif_class.py` |
| `paper_eval/models_gru_svdd.py` | GRU-DeepSVDD packages |

The unified modules exist so that one notebook can produce one table without manually chaining six separate runners.
