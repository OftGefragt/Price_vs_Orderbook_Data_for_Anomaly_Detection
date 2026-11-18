# Paper-ready text (Direction 2) — do not alter numbers

## Title

Cross-Asset Transfer for Crypto Anomaly Detection: Level-1 vs Level-2 Features under Deep and Classical Models

## One-sentence claim

Under Bitcoin→Ethereum unsupervised transfer and price-range proxy labels, deep Level-1 detectors match Level-2 deep Precision@5% (~0.19), and scale-free Level-1 forests match Level-2 forests; order-book depth is not required for this ranking task once prevalence and absolute-price scale are controlled.

## Label definition (Methods)

Proxy stress labels are defined at the **window** level (length 30, step 1). For each window we compute a scalar amplitude and mark the top 1% of windows as positive, with \(k=\mathrm{round}(0.01\,n)\) and deterministic tie-breaking (larger amplitude first; smaller window index second).

**Primary amplitude (non-circular):** range of price over the window — Level-1 uses Close; Level-2 uses mid-price from the row-aligned order-book archive (`midpoint`), not the 30 model features.

**Secondary (sensitivity / legacy):** Level-1 range of min-max-scaled Close; Level-2 range of the Euclidean norm of the 30 scaled book features.

Every results table reports base rate, precision, recall, lift \(=\mathrm{precision}/\mathrm{base\ rate}\), and \(\min(1,\mathrm{base\ rate}/\mathrm{flag\ fraction})\).

## Event merging

Windows flagged (or labeled) as positive are merged into events if successive positive window starts fall within a gap of \(G\in\{15,30,60\}\) starts (default \(G=30\)). This Ethereum window yields 1–2 independent events.

## SVM stage

A one-class RBF SVM was fit on one-dimensional training forest scores, but **its output was not used to rank test points**. Continuous forest scores determine Precision/Recall. For metric claims describe a GA-configured isolation forest, not an SVM-ranked system.

## Scale-free deep headline numbers (seeds 42–44)

| Track | Model | P@5 mean | Lift@5 |
|---|---|---:|---:|
| L1 scale-free | f-AnoGAN | 0.194 | 19.4 |
| L1 scale-free | GRU-DeepSVDD | 0.115 | 11.5 |
| L2 | f-AnoGAN | 0.198 | 20.0 |
| L2 | GRU-DeepSVDD | 0.198 | 20.0 |

Source: `results_v2/scalefree_deep/`.

## Note on the exploratory 0.26 figure

The exploratory baseline that reported L2 Precision@5% ≈ 0.26 used inclusive quantile labels with a higher L2 positive share (~1.32%); that figure is not comparable to exact top-1% evaluations and should not be used as evidence of book-feature value.
