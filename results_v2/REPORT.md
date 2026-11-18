# Methodology v2 report (Phase A complete; B/C partial)

**Rule:** all numbers from executed code under `results_v2/`. Baseline tables frozen in `results/baseline_v1/`.

---

## Task 0 — Audit

See `docs/AUDIT.md`.

**Plain finding:** Proxy labels use window amplitudes with `quantile + >=` (tie inflation). SVM predictions are discarded for ranking. GA used label-free score variance on a missing Colab matrix. L2 eval CSVs have no timestamps/price; AnoGAN ETH file supplies aligned midpoint.

---

## Task 1 — Label prevalence

| Track | Baseline base rate | Exact top-1% base rate |
|---|---:|---:|
| L1 1-min | 1.011% (189/18692) | 1.000% (187) |
| L1 5-min | 1.050% (39/3716) | 0.996% (37) |
| L2 | **1.319% (44/3336)** | 0.989% (33) |

**Why:** L2 had 14 amplitudes tied at the cutoff; inclusive quantile overstated positives.

**Before/after on matched seed-42 P@5 (exact top-1% labels):**  
L2 f-AnoGAN/GA-EIF drop from **0.263 → 0.198** (equals max achievable precision). L1 matched AnoGAN stays ~0.194. **Lift** stays ~19–20 on both — the headline precision gap was largely a base-rate artifact.

Artifacts: `results_v2/task1_label_prevalence/`

---

## Task 2 — Common non-circular label

Primary: window **price range** (L1 Close; L2 **midpoint** from AnoGAN ETH_5min, verified identical 30-feature rows).  
Secondary: legacy labels. Alternative: abs log-return.

**Seed-42 P@5 under primary price-range (exact 1%):**

| Model | L1 | L2 |
|---|---:|---:|
| f-AnoGAN | 0.194 | 0.198 |
| GA-EIF+SVM | 0.000 | 0.198 |
| GRU-DeepSVDD | 0.188 | 0.198 |

**Plain finding:** With a non-circular price label and equalized prevalence, deep models show **no meaningful L2 precision advantage**. L2 forest still beats raw L1 forest.

Alignment: L1∩L2 unique 5-min floors = 0 in naive intersection (L2 timestamps are irregular offsets); panels remain separate tracks. BTC midpoint train file empty.

Artifacts: `results_v2/task2_common_labels/`

---

## Task 3 — Event-level evaluation

Positive **windows** → **events** (gap=30):

| Level | Positive windows | Independent events |
|---|---:|---:|
| L1 | 37 | **2** |
| L2 | 33 | **1** |

Event recall for L2 deep/forest = 1.0 means recovering that **single** episode. Window-level recall ≈ 1.0 was misleading.

Artifacts: `results_v2/task3_events/`

---

## Task 4 — SVM / GA

- Paper pipeline: `svm.predict` **discarded**; metrics = forest continuous scores.  
- L2: Spearman(forest, SVM-on-score) ≈ 0.77; top-5% P/R unchanged in this run.  
- L1: SVM ranks identically to forest here (Spearman 1.0) but forest still fails on raw L1.  
- Contamination does **not** affect continuous-score ranking.  
- GA: label-free variance; data file missing → cannot certify train-only.

**Recommendation:** rename metric claims to **GA-EIF / IF**; describe SVM as unused post-processor unless a future run uses it for ranking.

Artifacts: `results_v2/task4_svm_ga/`

---

## Task 5 — OOD / scale-free L1 / splits

Forest P@5 under primary-style price labels:

| Setup | P@5 | R@5 | Lift |
|---|---:|---:|---:|
| L1 raw OHLCV BTC→ETH | **0.000** | 0.00 | 0 |
| L1 **scale-free** BTC→ETH | **0.199** | 1.00 | 20.0 |
| L1 scale-free ETH→BTC | 0.200 | 1.00 | 20.0 |
| L1 scale-free BTC time split | 0.198 | 1.00 | 20.0 |
| L2 BTC→ETH | 0.198 | 1.00 | 20.0 |

Mean row OOD fraction for raw L1 BTC→ETH ≈ **0.81**; Spearman(score, OOD frac) ≈ 0.43.

**Plain finding:** L1 forest failure in the paper is largely an **absolute-price / cross-asset scaling** failure, not proof that order-book features are required. Scale-free L1 matches L2 forest lift.

Artifacts: `results_v2/task5_ood_splits/`

---

## Claims: survive / soften / drop

### Drop (as stated in the preprint)

1. **“L2 raises Precision@5 from ~0.19 to ~0.26”** as evidence of book-feature value — **drop**. Under equal prevalence + price labels, deep P@5 ≈ 0.19–0.20 on both; 0.26 was max-precision inflation from extra L2 positives.

### Soften

2. **“L2 makes stress identification more robust”** — soften to: *under the original circular feature-norm labels and unequal base rates, L2 looked better; under price-based labels, deep-model window ranking is similar across L1/L2, and L2 has only one independent stress event in this ETH window.*  
3. **“GA-EIF+SVM”** — soften name for metrics: SVM unused; contamination irrelevant to ranking.  
4. **“L1 forest fails while L2 forest works because of microstructure richness”** — soften heavily: raw L1 forest fails from OOD levels; **scale-free L1 forest matches L2**.

### Survive (narrow)

5. Code/protocol reproducibility of the **baseline_v1** pipeline as an exploratory artifact.  
6. Deep L1 models (AnoGAN/GRU) can rank extreme **price-range** windows on ETH after BTC training (high lift), including 2022 L1-only check (prior work).  
7. Overlapping windows require **event-level** reporting; this sample has **very few** independent events (1–2).

### Not yet supported

8. Causal claim that **order-book depth/imbalance** specifically drives robustness (needs Task 7–8).  
9. Multi-seed / agreement tables under new labels (Task 6).  
10. Economic characterization of flags (Task 10).

---

## Incomplete tasks

| Task | Status | Why incomplete |
|---|---|---|
| 6 Proper Jaccard, 10 seeds | Not started | Heavy retrain; next |
| 7 Matched controls combo | Not started | Needs scale-free + feature sets + full retrain |
| 8 L2 feature-group ablation | Not started | Depends on 7 harness |
| 9 Extra baselines LOF/IF/OCSVM | Not started | |
| 10 Economic check | Not started | Needs volume/impact series |
| Deep OOD retrain all detectors | Partial | Forest probe only in Task 5 |
| Train-only GA re-run | Not done | Optional; hyperparameters treated as fixed |

---

## Paper-ready wording (Phase A)

See also `results_v2/PAPER_READY_TEXT.md`.
