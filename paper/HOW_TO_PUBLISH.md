# How to publish this preprint

## Title

**Cross-Asset Transfer for Crypto Anomaly Detection: Level-1 vs Level-2 Features under Deep and Classical Models**

## What to upload

| File | Use |
|---|---|
| `paper/Kaya_CrossAsset_Anomaly_Detection.docx` | **Primary manuscript** (readable APA7 Word) |
| `paper/RESEARCHGATE_MANUSCRIPT.pdf` | Optional PDF (print from HTML) |
| `paper/RESEARCHGATE_MANUSCRIPT.md` | Source of truth in the repo |
| `paper/references.bib` | BibTeX for LaTeX/arXiv workflows |
| GitHub URL | Code + `results_v2/` tables |

Regenerate artifacts:

```bash
python paper/build_docx.py
python paper/build_html.py
# Then open HTML → Print → Save as PDF if needed
```

## ResearchGate / SSRN (recommended first)

1. Open `paper/Kaya_CrossAsset_Anomaly_Detection.docx` (or export PDF from the HTML).
2. Add new → Research → Preprint.
3. Paste the Abstract from the manuscript.
4. Link: https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection
5. Keywords: anomaly detection; cryptocurrency; cross-asset transfer; order book; deep learning.

## arXiv (optional)

1. Primary category: `cs.LG`; secondary: `q-fin.ST`.
2. Upload PDF; put the GitHub URL in Comments.
3. Note: extends UEF AI coursework (2025); OftGefragt is the code host, not a co-author.

## Lead with (correct claims)

- **Cross-asset transfer (BTC→ETH) works** for ranking extreme price windows.
- Deep L1 is more robust than raw L1 forests under scale shift; scale-free L1 closes the gap.
- After label/prevalence controls, **deep L1 ≈ deep L2**; L2 is not required for this task.
- Main controlled table: `results_v2/scalefree_deep/`.

## Do not claim

- Confirmed pump-and-dump / fraud labels (proxy price-range stress only).
- That exploratory L2 Precision@5% ≈ 0.26 proves book-feature value.
- A full 2022 L1-versus-L2 result (2022 is L1-only).
- That OftGefragt is a co-author.
- Many independent ETH stress events (this window has 1–2).
