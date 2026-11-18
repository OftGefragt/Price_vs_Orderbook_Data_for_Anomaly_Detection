# Cross-Asset Transfer for Crypto Anomaly Detection (L1 vs L2)

**Title:** *Cross-Asset Transfer for Crypto Anomaly Detection: Level-1 vs Level-2 Features under Deep and Classical Models*

This repository studies **Bitcoin → Ethereum** unsupervised anomaly ranking with Level-1 OHLCV versus Level-2 order-book features, using deep (f-AnoGAN, GRU–Deep SVDD) and classical (Isolation Forest–family) models. Controlled methodology (`results_v2/`) shows that earlier L2 Precision@5% gains were largely **label prevalence and L1 scale artifacts**; after controls, **cross-asset transfer works without requiring L2**.

## Publish this preprint

| File | Use |
|---|---|
| `paper/Kaya_CrossAsset_Anomaly_Detection.docx` | **Word manuscript (APA7)** — preferred upload |
| `paper/RESEARCHGATE_MANUSCRIPT.md` | Author–year source |
| `paper/RESEARCHGATE_MANUSCRIPT.html` | Browser → Print → PDF |
| `paper/HOW_TO_PUBLISH.md` | ResearchGate / SSRN / arXiv steps |
| `results_v2/REPORT.md` | Controlled methodology report |
| `results_v2/scalefree_deep/` | Main scale-free deep GPU tables |

Regenerate Word/HTML after edits:

```bash
python paper/build_docx.py
python paper/build_html.py
```

## Headline findings (controlled)

| Setup | Precision@5% (approx.) |
|---|---:|
| Exploratory L2 (legacy labels) | ~0.26 |
| Deep L1 / L2 after exact top-1% + price labels | ~0.19–0.20 |
| Raw L1 forest BTC→ETH | ~0.00 |
| Scale-free L1 forest | ~0.20 |
| Scale-free L1 f-AnoGAN vs L2 deep | 0.194 vs 0.198 |

## Reproduce

```bash
git clone https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection.git
cd Price_vs_Orderbook_Data_for_Anomaly_Detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-paper.txt

# Exploratory matched baseline
python -m paper_eval.run_comparison --mode matched --regime 2021_spring

# Controlled Direction-2 CPU suite
python -m paper_eval.v2.run_direction2_cpu

# Scale-free deep (GPU recommended)
python -m paper_eval.v2.run_scalefree_deep --mode full
```

Colab: `notebooks/Methodology_v2_Direction2_Colab.ipynb` — see `docs/DIRECTION2_COLAB.md`.

## Repository layout

```text
paper/              Manuscript (DOCX, MD, HTML, BibTeX)
paper_eval/         Unified comparison + results_v2 methodology
results/            Exploratory / matched baseline tables
results_v2/         Controlled audits, scale-free deep, Tasks 1–10
docs/               Protocols and audit notes
Level 1 Analysis/   Original L1 course packages
Level 2 Analysis/   Original L2 course packages
notebooks/          Colab notebooks
```

## Academic context

Model packages began as coursework at the University of Eastern Finland (Artificial Intelligence, 2025). This repository adds a unified comparison layer and a controlled cross-asset transfer analysis for preprint deposit. OftGefragt is the code host, not a co-author.
