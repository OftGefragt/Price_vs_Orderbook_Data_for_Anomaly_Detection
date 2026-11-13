# Changes in the paper-ready release

This document lists what was added to the public repository for reproducible comparison. Existing Level-1 and Level-2 course packages were left in place.

## Added

1. `paper_eval/`
   Unified data loading, three detectors, shared metrics, and `run_comparison.py`.
2. `notebooks/L1_vs_L2_Comparison_Colab.ipynb`
   End-to-end Colab workflow: clone, install, run, display Table 5, download CSV.
3. `requirements-paper.txt`
   Minimal dependency list for the comparison suite.
4. `docs/`
   Formal notes on data, methods, changes, and the paper checklist.
5. `results/`
   Destination folder for generated tables and score files.
6. Root `README.md` rewrite
   Clear instructions for Colab and local runs.

## Behavioural choices

1. One feature matrix for Level 2 (thirty selected columns) for all unified detectors.
2. Shared top-quantile flagging for fair rate control across models.
3. Optional Level-1 row subsampling to keep Colab runtime manageable.
4. Short deep-model epochs by default. Increase them for final reported numbers if compute allows.
5. Automatic fallback from Extended Isolation Forest to sklearn IsolationForest when the local EIF code cannot be imported.

## Not removed

Pretrained weights, course READMEs, PCA and t-SNE figures, and the genetic-algorithm notebook outputs remain available for qualitative discussion in the paper.
