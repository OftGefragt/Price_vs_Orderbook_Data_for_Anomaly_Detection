# Comparison of Price and Order-Book Data for Anomaly Detection

This repository compares Level-1 price/volume features with Level-2 order-book
features for unsupervised anomaly detection in cryptocurrency markets. Three
detector families are evaluated under one protocol:

1. f-AnoGAN with one-dimensional convolutions
2. Extended Isolation Forest settings from a genetic search, with a one-class SVM stage
3. A GRU encoder trained with a Deep SVDD objective

Models train on Bitcoin and score Ethereum.

## Start here for paper-quality numbers

1. Open `notebooks/L1_vs_L2_Comparison_Colab.ipynb` in Google Colab.
2. Set the runtime to GPU.
3. Keep `MODE = "strong"`.
4. Run all cells.
5. Download the files under `results/`.

Local equivalent:

```bash
git clone https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection.git
cd Price_vs_Orderbook_Data_for_Anomaly_Detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-paper.txt
python -m paper_eval.run_comparison --mode strong
```

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
| `strong` | Paper protocol: 3 seeds, longer epochs, large L1 train, full ETH test |
| `fast` | Smoke test with shorter epochs and smaller L1 subsets |

Details: `docs/STRONG_PROTOCOL.md`

## Repository layout

```text
Level 1 Analysis/   Original Level-1 course packages and figures
Level 2 Analysis/   Original Level-2 course packages and figures
paper_eval/         Unified comparison code
notebooks/          Colab notebook
docs/               Protocol and method notes
results/            Generated tables after a run
```

## Documentation

- `docs/STRONG_PROTOCOL.md` — paper evaluation settings
- `docs/DATA_AND_PROTOCOL.md` — datasets and scoring rules
- `docs/METHODS.md` — model summaries
- `docs/PAPER_CHECKLIST.md` — steps after numbers exist
- `docs/CHANGES.md` — release notes for the comparison suite

## Academic context

The model packages began as coursework at the University of Eastern Finland
(Artificial Intelligence, 2025). This repository adds a unified comparison
layer for a technical report or preprint.
