# Paper-ready text (Phase A) — do not alter numbers

## Label definition (Methods)

Proxy stress labels are defined at the **window** level (length 30, step 1). For each window we compute a scalar amplitude and mark the top 1% of windows as positive, with \(k=\mathrm{round}(0.01\,n)\) and deterministic tie-breaking (larger amplitude first; smaller window index second).  

**Primary amplitude (non-circular):** range of price over the window — Level-1 uses Close; Level-2 uses mid-price from the row-aligned order-book archive (`midpoint`), not the 30 model features.  

**Secondary (sensitivity):** Level-1 range of min-max-scaled Close; Level-2 range of the Euclidean norm of the 30 scaled book features (legacy).  

**Alternative:** sum of absolute log-returns of Close/mid over the window.

Every results table reports base rate, precision, recall, lift \(=\mathrm{precision}/\mathrm{base\ rate}\), and \(\min(1,\mathrm{base\ rate}/\mathrm{flag\ fraction})\).

## Event merging

Windows flagged (or labeled) as positive are merged into events if successive positive window starts fall within a gap of \(G\in\{15,30,60\}\) starts (default \(G=30\)). Event precision/recall use interval overlap; uncertainty uses a blocked bootstrap with block length 30.

## SVM stage

A one-class RBF SVM was fit on one-dimensional training forest scores and called at inference, but **its output was not used to rank test points**. Continuous forest scores (path length or \(-\)`score_samples`) determine Precision/Recall and Jaccard. The `contamination` hyperparameter does not change the ordering of those continuous scores. For metric claims the detector should be described as a GA-configured isolation forest (backend logged), not as an SVM-ranked system.

## Table captions (suggested)

**Table X.** Window-level ranking against primary price-range labels (exact top 1%). Metrics include base rate, precision, recall, lift, and max achievable precision at the flag budget.

**Table Y.** Event-level metrics after merging overlapping windows (gap 30), with 95% blocked-bootstrap intervals. Note the number of independent positive events per track.

**Note.** The exploratory baseline that reported L2 Precision@5% ≈ 0.26 used inclusive quantile labels with a higher L2 positive share (~1.32%); that figure is not comparable to exact top-1% evaluations.
