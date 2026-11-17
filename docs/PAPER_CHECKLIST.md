# Checklist before writing the PDF

## Before the manuscript

1. Run `python -m paper_eval.run_comparison --mode strong` (or the Colab notebook with `MODE = "strong"`).
2. Confirm these files exist:
   - `results/table5_main_comparison_agg.csv`
   - `results/table5_agreement_L1.csv`
   - `results/table5_agreement_L2.csv`
   - `results/protocol.json`
3. Open `protocol.json` and copy seed list, epoch counts, and row counts into Experimental Setup.
4. Note the EIF/IsolationForest backend string from the results table.
5. Commit the CSV/MD/JSON result files to the repository (or attach them to the preprint archive).

## Manuscript content that must match the run

1. Table 5 = aggregated metrics (`*_mean` and `*_std` columns).
2. Agreement tables = Jaccard matrices.
3. Limitations must mention:
   - proxy stress labels are not confirmed manipulation events
   - Level-1 and Level-2 sampling rates differ (1 min vs 5 min)
   - Level-1 train may be subsampled (see protocol.json)
   - large-n Level-1 forest may use sklearn IsolationForest

## Only after the numbers exist

1. Draft Abstract and Results with real values (no placeholders).
2. Add selected qualitative figures from the course folders.
3. Export PDF and submit to arXiv (`cs.LG`, optional `q-fin.ST`).
