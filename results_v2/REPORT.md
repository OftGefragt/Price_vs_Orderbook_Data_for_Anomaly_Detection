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

## Task 6 — Agreement (partial)

- Deep pairwise Jaccard/Spearman: seeds **42–44** only (matched dumps).  
- Forest seed-stability: seeds **42–51**.  
- Chance Jaccard \(p/(2-p)\) included.  
- **Incomplete:** f-AnoGAN/GRU not re-trained for seeds 45–51.

## Task 7 — Matched controls (forest probe)

Under primary price labels, IsolationForest P@5 lift:

| Control | Lift |
|---|---:|
| L1 raw 5-min | 0 |
| L1 scale-free | **19.98** |
| L2 full 30 | **19.98** |
| L2 reduced 5 (pre-registered) | **19.98** |
| L1 scale-free subsampled to L2 n | **19.98** |

**Plain finding:** once L1 is scale-free, the L2−L1 lift gap **vanishes** for the forest. Reducing L2 to 5 features does not hurt.

## Task 8 — Feature groups

Distance-only groups recover full lift; notional-flow-only groups give lift 0. Leave-one-group-out still full lift if distances remain. Suggests **distance geometry**, not notionals, drives forest recovery of the single price-stress episode — still not proof of “deep book robustness.”

## Incomplete tasks

| Task | Status | Why incomplete |
|---|---|---|
| 6 full 10-seed deep | Partial | Only 42–44 deep dumps |
| 7 deep-model controls | Partial | Forest probe only |
| 8 deep-model group ablation | Partial | Forest only |
| 9 LOF / plain IF / OCSVM baselines | Done (seed 42) | See `task9_baselines/` — raw L1 IF fails but LOF/OCSVM still recover ~0.17–0.18 P@5 |
| 10 Economic check of flags | Done (seed 42) | See `task10_economic/` — flagged windows have higher realized vol / volume |
| Scale-free deep AnoGAN/GRU | **Done (Colab GPU)** | See section below + `scalefree_deep/` |
| Train-only GA re-run | Not done | Treated as fixed constants |
| Deep seeds 45–51 | Not done | Optional |

---

## Scale-free deep (Colab GPU, seeds 42–44)

Primary price-range labels, exact top-1%. L1 uses scale-free features; L2 unchanged. Config: `results_v2/scalefree_deep/config.json`.

| Track | Model | P@5 mean | P@5 std | Lift@5 |
|---|---|---:|---:|---:|
| L1 scale-free | f-AnoGAN | 0.194 | 0.005 | 19.4 |
| L1 scale-free | GRU-DeepSVDD | 0.115 | 0.025 | 11.5 |
| L2 | f-AnoGAN | 0.198 | 0.000 | 20.0 |
| L2 | GRU-DeepSVDD | 0.198 | 0.000 | 20.0 |

**Plain finding:** Under matched price labels, scale-free L1 f-AnoGAN ≈ L2 deep (~0.19 P@5). Scale-free L1 GRU remains weaker (~0.11). Supports Direction-2 soft claim: L2 is not clearly superior once prevalence/label artifacts and L1 scaling are controlled; forest recovery of L1 is mostly OOD/scale, not a deep-book free lunch.

Artifacts: `results_v2/scalefree_deep/`

---

## Remaining (optional, not blocking preprint)

1. Optional: Chris authorship contact if co-authorship is agreed  
2. Optional: Tasks 6–8 deep extensions; GA train-only re-run; deep seeds 45–51  

Manuscript (Direction-2 framing): `paper/Kaya_CrossAsset_Anomaly_Detection.docx`, `paper/RESEARCHGATE_MANUSCRIPT.md`

---

## Paper-ready wording (Phase A)

See also `results_v2/PAPER_READY_TEXT.md`.
