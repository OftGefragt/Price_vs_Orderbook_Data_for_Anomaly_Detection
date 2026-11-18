# Comparison of Price and Order-Book Data for Anomaly Detection

This repository compares Level-1 price/volume features with Level-2 order-book
features for unsupervised anomaly detection in cryptocurrency markets. Three
detector families are evaluated under one protocol:

1. f-AnoGAN with one-dimensional convolutions
2. Extended Isolation Forest settings from a genetic search, with a one-class SVM stage
3. A GRU encoder trained with a Deep SVDD objective

Models train on Bitcoin and score Ethereum.

## Publish this preprint

1. Open `paper/RESEARCHGATE_MANUSCRIPT.html` in a browser.
2. Print → Save as PDF.
3. Follow `paper/HOW_TO_PUBLISH.md` (ResearchGate first, arXiv optional).
4. Main numbers: `results/matched_2021_spring/`.

Manuscript source: `paper/RESEARCHGATE_MANUSCRIPT.md`

## Reproduce paper-quality numbers

Matched protocol (recommended):

```bash
git clone https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection.git
cd Price_vs_Orderbook_Data_for_Anomaly_Detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-paper.txt
python -m paper_eval.run_comparison --mode matched --regime 2021_spring
```

Colab: open `notebooks/L1_vs_L2_Comparison_Colab.ipynb` (set `MODE = "matched"` if available, else `"strong"`).

### Outputs written by a strong run

| File | Role |
|---|---|
| `results/table5_main_comparison.csv` | Per-seed metrics |
| `results/table5_main_comparison_agg.csv` | Mean and std across seeds |
| `results/table5_agreement_L1.csv` | Pairwise Jaccard of top-5% flags |
| `results/table5_agreement_L2.csv` | Same for Level-2 |
| `results/protocol.json` | Exact settings used |
| `results/scores_*.npz` | Raw anomaly scores |

## Modes

| Mode | Purpose |
|---|---|
| `strong` | Exploratory protocol: 1-min L1, 3 seeds, longer epochs |
| `matched` | Confound-controlled: L1 aggregated to 5-min, shared sklearn forest on L1 and L2 |
| `fast` / `matched_fast` | Smoke tests with shorter epochs |

```bash
python -m paper_eval.run_comparison --mode matched --regime 2021_spring
python -m paper_eval.fetch_regime_ohlcv --regime 2022_bear
python -m paper_eval.run_comparison --mode matched --regime 2022_bear --levels L1
```

Details: `docs/STRONG_PROTOCOL.md`, `docs/MATCHED_PROTOCOL.md`  
Matched tables: `results/matched_2021_spring/`, `results/matched_2022_bear/`

## Repository layout

```text
Level 1 Analysis/   Original Level-1 course packages and figures
Level 2 Analysis/   Original Level-2 course packages and figures
paper_eval/         Unified comparison code
notebooks/          Colab notebook
docs/               Protocol and method notes
results/            Generated tables after a run
```

## Paper draft (arXiv)

| File | Role |
|---|---|
| `paper/RESEARCHGATE_MANUSCRIPT.md` | Author–year manuscript (preferred for ResearchGate) |
| `paper/MANUSCRIPT.md` | BibTeX-keyed draft |
| `paper/references.bib` | Bibliography (40+ entries) |
| `paper/RELATED_WORK_OUTLINE.md` | Related Work map |
| `paper/HOW_TO_PUBLISH.md` | arXiv upload checklist |

## Documentation

- `docs/MATCHED_PROTOCOL.md` — 5-min L1, shared forest, 2022 regime
- `docs/STRONG_PROTOCOL.md` — exploratory strong settings
- `docs/DATA_AND_PROTOCOL.md` — datasets and scoring rules
- `docs/METHODS.md` — model summaries
- `docs/PAPER_CHECKLIST.md` — steps after numbers exist
- `docs/CHANGES.md` — release notes for the comparison suite

## Academic context

The model packages began as coursework at the University of Eastern Finland
(Artificial Intelligence, 2025). This repository adds a unified comparison
layer for a technical report or preprint.
