# How to publish this manuscript on arXiv

## What is ready

1. `references.bib` — bibliography skeleton (40+ entries)
2. `RELATED_WORK_OUTLINE.md` — section map from citations to prose
3. `MANUSCRIPT.md` — full draft with Table 5 numbers filled
4. Strong-run tables already in `../results/`

## Before you click Submit on arXiv

1. Open the two arXiv papers cited as `arxiv2412pd` and `arxiv2507lob` and copy the exact author lists into `references.bib`.
2. Skim every citation you keep in the final PDF. Remove any entry you did not actually read.
3. Optional: convert `MANUSCRIPT.md` to LaTeX with pandoc:

```bash
pandoc MANUSCRIPT.md --bibliography=references.bib -o kaya_l1_l2_anomaly.pdf
```

4. Create an arXiv account, choose primary category `cs.LG`, secondary `q-fin.ST`.
5. Upload the PDF and, optionally, a ZIP of the code or link the GitHub URL in the comments.
6. State in the comments that this extends UEF AI coursework (2025).

## Writing notes (keep it non-generic)

- Prefer the concrete numbers already in Section 6.
- Do not add marketing adjectives.
- When you edit Related Work, paraphrase with the paper closed, then check the PDF.
- Keep Limitations as written; prestige venues punish overclaiming more than modest scope.

## What this is not yet

A camera-ready Oxford University Press journal article. It is an arXiv-ready technical report with a real bibliography and completed results tables.
