# Comparison of Price and Order-Book Data for Anomaly Detection

This repository studies whether Level-2 order-book features improve unsupervised anomaly detection relative to Level-1 price and volume series in cryptocurrency markets. Three detector families are compared under a shared evaluation protocol:

1. f-AnoGAN with one-dimensional convolutions
2. Extended Isolation Forest tuned by a genetic algorithm, followed by a one-class SVM stage
3. A GRU encoder trained with a Deep SVDD objective

Models are trained on Bitcoin data and evaluated on Ethereum data.

The original course materials remain under `Level 1 Analysis/` and `Level 2 Analysis/`. The `paper_eval/` package provides a single runnable comparison that writes the main results table required for the accompanying paper draft.

## Quick start in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload or open `notebooks/L1_vs_L2_Comparison_Colab.ipynb` from this repository.
3. Set the runtime to GPU if available.
4. Run all cells.

The notebook clones this repository, installs `requirements-paper.txt`, runs the comparison, and displays `results/table5_main_comparison.csv`.

Direct notebook path after cloning:

```text
notebooks/L1_vs_L2_Comparison_Colab.ipynb
```

## Local reproduction

```bash
git clone https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection.git
cd Price_vs_Orderbook_Data_for_Anomaly_Detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-paper.txt
python -m paper_eval.run_comparison --max-l1-train 12000 --max-l1-test 8000 --seed 42
```

Outputs:

| File | Content |
|---|---|
| `results/table5_main_comparison.csv` | Main numeric comparison |
| `results/table5_main_comparison.md` | Markdown form of the same table |
| `results/scores_*.npz` | Per-model anomaly scores on the test windows |

## Data

| Level | Train | Test | Resolution | Role |
|---|---|---|---|---|
| Level 1 | BTC OHLCV | ETH OHLCV | 1 minute | Price and volume only |
| Level 2 | BTC order-book subset | ETH order-book subset | 5 minute | Selected depth and notional features |

Exact counts, date ranges, and feature lists are documented in `docs/DATA_AND_PROTOCOL.md`.

## Evaluation protocol

Each detector produces a continuous anomaly score on the test windows. Flags are assigned with a shared top-quantile rule:

- top 5 percent of scores
- top 10 percent of scores

Because the quantile rule fixes the flag rate by construction, the useful comparisons are:

- agreement between models on the same data level (Jaccard against f-AnoGAN)
- score dispersion
- training and inference time
- qualitative market-context plots retained from the earlier course outputs

## Repository layout

```text
Level 1 Analysis/     Original Level-1 model packages and figures
Level 2 Analysis/     Original Level-2 model packages and figures
paper_eval/           Unified training and scoring code for the paper table
notebooks/            Colab notebook
docs/                 Formal notes on data, methods, and protocol
results/              Generated tables and score files
requirements-paper.txt
```

## Documentation

- `docs/DATA_AND_PROTOCOL.md`: datasets, features, and scoring rules
- `docs/METHODS.md`: short method descriptions aligned with the code
- `docs/CHANGES.md`: what was added for the paper-ready release
- `docs/PAPER_CHECKLIST.md`: remaining steps before arXiv submission

## Academic context

The underlying model packages were developed for the Artificial Intelligence course at the University of Eastern Finland (2025). This release reorganizes those materials into a reproducible comparison suitable for a technical report or preprint.

## Citation

If you use this repository, cite the GitHub URL and the course year. Update the citation block after the preprint is posted on arXiv.
