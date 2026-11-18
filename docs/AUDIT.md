# Task 0 audit (code + measured counts)

**Date:** 2026-11-18  
**Scope:** `paper_eval/` evaluation pipeline and Level-2 GA notebook/source used for published hyperparameters.  
**Baseline freeze:** original tables copied to `results/baseline_v1/` (not modified).  
**Rule:** numbers below were produced by running loaders/label functions; nothing was hand-edited.

---

## a) Proxy labels (where / how / when)

**Source file:** `paper_eval/proxy_labels.py`

### Level-1 (`level1_stress_flags`)

- **Granularity:** window-level boolean vector of length `n_rows - window + 1` (window=30, step=1 implied by construction).
- **Amplitude definition:** on each window of **scaled Close** (`x_test[:, 3]`),  
  `amplitude = max(Close_window) - min(Close_window)`.  
  Docstring mentions log-returns; **code does not compute log-returns**. It uses range of min-max-scaled Close.
- **Timing:** labels are built from the already-scaled test matrix **after** scaling and **as window aggregates** (not row-level labels first). Windowing is done inside the label function by looping over start indices; it does not call `sliding_windows`, but the semantics are the same.
- **Threshold:** `thr = np.quantile(amps, q)` with `q=0.99`, then `amps >= thr`.
- **Ties:** any window with amplitude **exactly equal** to `thr` is positive. `np.quantile` + `>=` can therefore yield **more than 1%** positives when ties exist. There is **no deterministic top-k / argsort tie-break**.

### Level-2 (`level2_stress_flags`)

- **Granularity:** same window-level construction.
- **Amplitude definition:** for each row, `norm = ||x_row||_2` over all **30 scaled order-book features**; then  
  `amplitude = max(norm_window) - min(norm_window)`.
- **Circular risk:** the L2 label is a function of the **same features** the detectors receive.
- **Ties:** same `quantile` + `>=` rule as L1.

### Where labels enter metrics

In `paper_eval/run_comparison.py`, labels are recomputed each seed from the (possibly subsampled) scaled test matrix, then used only for Precision/Recall. Jaccard agreement uses top-quantile **score** flags, not these labels.

---

## b) Test windows and positive counts (measured)

Window = 30. Positive share = positives / windows.

| Track | Protocol matrix | Test rows | Windows | Positives | Positive share |
|---|---|---:|---:|---:|---:|
| L1 1-min (strong / exploratory ETH full) | scaled ETH 1-min | 18,721 | 18,692 | 189 | **1.011%** |
| L1 5-min (matched) | aggregated ETH | 3,745 | 3,716 | 39 | **1.050%** |
| L2 5-min features | scaled ETH L2 | 3,365 | 3,336 | 44 | **1.319%** |

**Why L2 share > 1%:** ties at the 0.99 quantile. On L2, `n_amp == thr` was **14** and `n_amp >= thr` was **44**, while `round(0.01 * 3336) = 33`. On L1 1-min, ties also inflate slightly (189 vs target 187).

**Per-seed note:**  
- `strong` / `matched` use **full ETH test** (`max_l1_test=None`), so **labels do not change with seed**.  
- Seeds only change deep-model init and L1 **train** subsampling in `strong` (`max_l1_train=30000`).  
- `fast` would subsample test and change labels; that mode is not the paper table.

---

## c) L1 features and scaling

- **Features:** raw OHLCV levels `Open, High, Low, Close, Volume` from CSV (`paper_eval/data.py`, `L1_FEATURE_COLS`). **Not** returns/z-scores in the baseline pipeline.
- **Scaling:** `MinMaxScaler(feature_range=(0,1))` fit on **BTC train only**, then applied to ETH test (`scale_fit_transform`).
- **Consequence:** BTC Close ≈ 5e4 vs ETH Close ≈ 2e3 in spring 2021, so ETH prices sit in a different absolute band before scaling; after BTC-fit min-max, ETH can be heavily **out of [0,1] / compressed** depending on ranges (OOD concern for Task 5).

Symbols in CSV: `BTC-USD`, `ETH-USD` with UTC `Date` timestamps.

---

## d) One-class SVM stage

**Source:** `paper_eval/models_eif.py` (`EIFSVMDetector`)

| Step | What happens |
|---|---|
| Fit | Forest scores on **train** rows → `OneClassSVM(kernel='rbf', gamma='scale', nu=max(contamination,0.01))` fit on those **1-D scores** |
| Score | Forest scores on **test** rows are returned |
| SVM at inference | `self._svm.predict(scores.reshape(-1,1))` is called but the result is **discarded** (`_ = ...`) |

**Effect on reported Precision/Recall/Jaccard:** **none**. Ranking uses continuous forest path lengths / `-score_samples` only.

**Contamination:** passed into sklearn `IsolationForest(contamination=...)` and into SVM `nu`. For continuous `score_samples` ranking, sklearn contamination mainly sets the internal decision offset for hard labels, **not** the ordering of continuous scores. Pure EIF backend ignores contamination for scoring entirely.

**Paper implication:** the published name “GA-EIF+SVM” overstates the SVM’s role in the metrics. Ablation (Task 4) should confirm forest-only ≡ current pipeline for ranking metrics.

---

## e) Genetic search

**Source:** `Level 2 Analysis/GA-optimized-EIF-SVM-for-anomaly-detection-main/utils/ga_class.py` and notebook `notebooks/ga_eif_svm.ipynb`.

| Item | Code evidence |
|---|---|
| Search space | `ntrees`, `sample_size`, `contamination`, `exlevel` |
| Fitness if `labels` given | F1 of thresholded scores vs labels |
| Fitness if `labels is None` | **`np.var(scores)`** (score variance) |
| Actual notebook call | `best_params = ga_eif(X, pop_size=10, generations=10)` → **no labels** |
| Printed fitnesses | ~0.0018–0.0019 (consistent with variance, not F1∈[0,1] typical of rare-event F1) |
| Reported champion | `{ntrees: 235, sample_size: 1691, contamination: 0.062, exlevel: 26}` |
| Data `X` | From Colab `df_selection.csv` → drop columns → MinMax scale. **File not in this repo.** Cannot prove from repo alone that `X` was BTC-train-only vs mixed. |

**Leakage verdict (conservative):**  
- **No evidence** that ETH test proxy labels were used in fitness (labels omitted).  
- **Cannot clear** train-only data hygiene for the GA matrix because `df_selection.csv` is missing. Treat hyperparameters as **fixed inherited constants** for `paper_eval`, not as a re-runnable, audited search. Task 4 should re-run a train-only, label-free GA if claims depend on “GA-optimized.”

`paper_eval` does **not** re-run GA; it hard-codes the champion hyperparameters.

---

## f) Timestamps, exchange, depth, dates, source

### Level-1 (bundled)

| Field | Value |
|---|---|
| Timestamps | Yes (`Date`, UTC, `+00:00`) |
| Symbols | `BTC-USD` train, `ETH-USD` test |
| Range train | 2021-03-31 21:00 UTC → 2021-05-31 20:59 UTC |
| Range test | 2021-04-06 21:00 UTC → 2021-04-19 21:00 UTC |
| Bar | 1 minute (aggregable to 5 min in matched mode) |
| Exchange | **Not stated in CSV**; symbol form `BTC-USD` / `ETH-USD` (Coinbase-style), not `BTCUSDT` |
| Source | Bundled under `Level 1 Analysis/data/`; provenance beyond filename not documented in code |

### Level-2 (bundled GA-EIF matrices)

| Field | Value |
|---|---|
| Timestamps in used CSV | **No** — only `Unnamed: 0` + 30 feature columns |
| Synthetic index | `RangeIndex` assigned in `load_level2` |
| Rows | train 3367, test 3365, **30 features** |
| Mid / close / spread columns in these 30 | **None** |
| Depth | Feature names indicate book distances/notionals at levels (e.g. `bids_distance_*`, `asks_limit_notional_*`); not a full contiguous top-N dump |
| Related AnoGAN ETH file | `ETH_5min.csv` has `system_time` from **2021-04-07 11:37 UTC → 2021-04-19 09:50 UTC**, 3365 rows (same count as L2 test). Suggests ~5-min LOB snapshots, but **the GA-EIF CSVs used by `paper_eval` do not carry those timestamps** |
| Exchange / exact depth protocol | **Not documented** in the loaded CSVs |

**Aligned common price label (Task 2):** blocked for strict L1–L2 timestamp alignment unless timestamps are recovered from another file or external join. Closest available common series may require using AnoGAN `midpoint`/`system_time` if it matches the 30-feature row order — **not verified yet**.

---

## g) Seeds: what changes; determinism

### What each seed controls in `run_comparison.py`

| Component | Seed effect |
|---|---|
| L1 train subsample (`strong`) | `np.random.default_rng(seed)` |
| L1 test subsample (`fast` only) | `seed + 1` |
| f-AnoGAN | `torch.manual_seed(seed)`, `np.random.seed(seed)`; DataLoader `shuffle=True` without a seeded generator |
| GRU-DeepSVDD | `torch.manual_seed(seed)`, `np.random.seed(seed)` |
| Forest sklearn backend | `random_state=seed` |
| Forest EIF backend | **No seed wired** into `random` / `np.random` inside `eif_class.py` |
| Proxy labels | Independent of seed when test matrix is fixed |
| GA | Not run inside `paper_eval`; historical GA used Python `random` without fixed seed in the notebook cell |

### Determinism summary

| Detector | Deterministic given seed? |
|---|---|
| sklearn IsolationForest path | Yes (seeded) |
| Pure EIF path | **No** (unseeded `random` / `np.random`) |
| f-AnoGAN / GRU | Approximately yes on CPU with seed; still subject to shuffle order / BLAS / CUDA nondeterminism |
| Label function | Deterministic for fixed test matrix |

Matched protocol forces sklearn forest on both tiers, so forest ranking **is** seeded there. Exploratory strong L2 used pure EIF → seed does not fully control that backend.

---

## Immediate methodological risks (for later tasks)

1. **Positive prevalence mismatch** L1 ≈ 1.01–1.05% vs L2 ≈ 1.32% from quantile ties (Task 1).  
2. **L2 label circularity** (Task 2).  
3. **No timestamps on L2 matrices used by eval** → hard to align a common price label (Task 2/3).  
4. **SVM does not affect rankings** (Task 4).  
5. **GA provenance opaque**; fitness was label-free variance, but data slice unverified (Task 4).  
6. **Raw price levels + BTC→ETH min-max** (Task 5).

---

## Files inspected

- `paper_eval/proxy_labels.py`, `data.py`, `models_eif.py`, `models_anogan.py`, `models_gru_svdd.py`, `metrics.py`, `run_comparison.py`
- `Level 2 Analysis/.../utils/ga_class.py`, `notebooks/ga_eif_svm.ipynb`
- `Level 2 Analysis/.../eif/eif_class.py`
- Bundled L1/L2 CSVs; AnoGAN `ETH_5min.csv` header/time range only
