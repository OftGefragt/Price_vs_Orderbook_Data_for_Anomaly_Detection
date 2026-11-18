# Task 4 — SVM stage and GA check

## SVM ablation (primary price label, seed 42)

level                  ranking  seed  spearman_forest_vs_svm_rank  frac_svm_outlier    n  flag_frac  k_flags  n_positives  base_rate  precision  recall      lift  max_achievable_precision  tp note
   L1              forest_only    42                     1.000000          1.000000 3716       0.05      186           37   0.009957   0.000000     0.0  0.000000                  0.198925   0  NaN
   L1              forest_only    42                     1.000000          1.000000 3716       0.10      372           37   0.009957   0.000000     0.0  0.000000                  0.099462   0  NaN
   L1 svm_rank_on_forest_score    42                     1.000000          1.000000 3716       0.05      186           37   0.009957   0.000000     0.0  0.000000                  0.198925   0  NaN
   L1 svm_rank_on_forest_score    42                     1.000000          1.000000 3716       0.10      372           37   0.009957   0.000000     0.0  0.000000                  0.099462   0  NaN
   L2              forest_only    42                     0.770723          0.106686 3336       0.05      167           33   0.009892   0.197605     1.0 19.976048                  0.197605  33  NaN
   L2              forest_only    42                     0.770723          0.106686 3336       0.10      334           33   0.009892   0.098802     1.0  9.988024                  0.098802  33  NaN
   L2 svm_rank_on_forest_score    42                     0.770723          0.106686 3336       0.05      167           33   0.009892   0.197605     1.0 19.976048                  0.197605  33  NaN
   L2 svm_rank_on_forest_score    42                     0.770723          0.106686 3336       0.10      334           33   0.009892   0.098802     1.0  9.988024                  0.098802  33  NaN

## Documentation for the paper

```json
{
  "fitness_if_labels_provided": "F1 of thresholded EIF scores vs labels",
  "fitness_in_recorded_notebook_run": "np.var(scores) (labels omitted)",
  "recorded_call": "best_params = ga_eif(X, pop_size=10, generations=10)",
  "recorded_champion": {
    "ntrees": 235,
    "sample_size": 1691,
    "contamination": 0.062,
    "exlevel": 26
  },
  "eth_test_labels_used": false,
  "train_only_matrix_verified": false,
  "reason_unverified": "X came from missing Colab file df_selection.csv",
  "paper_eval_reruns_ga": false,
  "recommendation": "Describe hyperparameters as fixed inherited constants from a label-free variance search on an unreproducible Colab matrix; do not claim a leakage-free audited GA unless re-run on BTC train only.",
  "contamination_and_continuous_scores": "sklearn IsolationForest contamination does not change the ordering of score_samples; it affects the hard decision offset. Pure EIF ignores contamination for path-length scores. Therefore the tuned contamination value does not affect the paper's Precision/Recall/Jaccard under top-quantile flagging of continuous scores.",
  "svm_paper_wording": "A one-class SVM was fit on one-dimensional training forest scores and invoked at inference, but its predictions were not used to rank test windows. Reported metrics use continuous forest anomaly scores only. Rename the detector to GA-EIF (or IF) for metric claims, or report the SVM as an unused post-processor."
}
```

## Finding

The published pipeline's ranking ignores SVM output. Spearman between forest scores and SVM-on-score ranking is reported in the CSV. Contamination does not affect continuous-score top-quantile metrics. GA was label-free variance on an unverifiable Colab matrix.
