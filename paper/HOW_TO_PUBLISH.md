# How to publish this preprint

## What is ready in this repository

| File | Use |
|---|---|
| `paper/RESEARCHGATE_MANUSCRIPT.md` | **Upload source** (author–year citations) |
| `paper/RESEARCHGATE_MANUSCRIPT.html` | Open in browser → Print → Save as PDF |
| `paper/references.bib` | BibTeX for arXiv/LaTeX workflows |
| `paper/MANUSCRIPT.md` | Alternate draft with BibTeX-style keys |
| `results/matched_2021_spring/` | **Main** Table 1 numbers |
| `results/table5_main_comparison_agg.csv` | Exploratory 1-min L1 run |
| `results/matched_2022_bear/` | 2022 L1-only check |
| GitHub repo | Code + data + protocols |

## ResearchGate (recommended first)

1. Open `paper/RESEARCHGATE_MANUSCRIPT.html` in Chrome/Safari.
2. Optionally replace the contact line with your email in the `.md` file and regenerate HTML (`python paper/build_html.py`).
3. Print → Save as PDF (A4, backgrounds on).
4. On ResearchGate: Add new → Research → Preprint.
5. Title: *Level-1 Price versus Level-2 Order-Book Features for Unsupervised Anomaly Detection in Cryptocurrency Markets*
6. Upload the PDF.
7. Paste the Abstract from the manuscript.
8. Link the GitHub URL in the description.
9. Categories: Machine Learning / Cryptocurrency / Anomaly Detection (as available).

## arXiv (optional second step)

1. Convert the markdown to PDF (pandoc if installed), or reuse the ResearchGate PDF.
2. Create an arXiv account.
3. Primary category: `cs.LG`; secondary: `q-fin.ST`.
4. Upload PDF; put the GitHub URL in Comments.
5. Note in Comments: extends UEF AI coursework (2025); OftGefragt is the code host, not a co-author.

```bash
# optional, if pandoc is installed
pandoc paper/RESEARCHGATE_MANUSCRIPT.md -o paper/kaya_l1_l2_anomaly.pdf
```

## Do not claim

- Confirmed pump-and-dump / fraud labels (you use proxy stress labels).
- A full 2022 L1-versus-L2 result (2022 is L1-only).
- That OftGefragt is a co-author.

## Lead with

**Matched 2021 Table 1** (5-min L1 + shared forest backend). That is the confound-controlled result.
