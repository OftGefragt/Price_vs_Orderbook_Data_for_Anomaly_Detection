# Main comparison (matched_fast, 2021_spring)

Precision/Recall use weak proxy stress labels (top 1% amplitude windows). Flag rates use a shared top-quantile rule on anomaly scores.

| model | data_level | n_seeds | backend | n_windows_mean | n_windows_std | score_mean_mean | score_mean_std | score_std_mean | score_std_std | flag_rate_5pct_mean | flag_rate_5pct_std | flag_rate_10pct_mean | flag_rate_10pct_std | train_seconds_mean | train_seconds_std | infer_seconds_mean | infer_seconds_std | precision_at_5pct_mean | precision_at_5pct_std | recall_at_5pct_mean | recall_at_5pct_std | precision_at_10pct_mean | precision_at_10pct_std | recall_at_10pct_mean | recall_at_10pct_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f-AnoGAN | L1 | 1 |  | 3716.0000 | 0.0000 | 2.5772 | 0.0000 | 0.7626 | 0.0000 | 0.0501 | 0.0000 | 0.1001 | 0.0000 | 22.1753 | 0.0000 | 0.3596 | 0.0000 | 0.1935 | 0.0000 | 0.9231 | 0.0000 | 0.0995 | 0.0000 | 0.9487 | 0.0000 |
| GA-EIF+SVM | L1 | 1 | sklearn_iforest | 3745.0000 | 0.0000 | 0.7544 | 0.0000 | 0.0260 | 0.0000 | 0.0654 | 0.0000 | 0.1007 | 0.0000 | 0.9877 | 0.0000 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| GRU-DeepSVDD | L1 | 1 |  | 3716.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0013 | 0.0000 | 0.0501 | 0.0000 | 0.1001 | 0.0000 | 5.9054 | 0.0000 | 0.1763 | 0.0000 | 0.1559 | 0.0000 | 0.7436 | 0.0000 | 0.0995 | 0.0000 | 0.9487 | 0.0000 |
| f-AnoGAN | L2 | 1 |  | 3336.0000 | 0.0000 | 0.0321 | 0.0000 | 0.0407 | 0.0000 | 0.0501 | 0.0000 | 0.1001 | 0.0000 | 4.7405 | 0.0000 | 0.4470 | 0.0000 | 0.2635 | 0.0000 | 1.0000 | 0.0000 | 0.1317 | 0.0000 | 1.0000 | 0.0000 |
| GA-EIF+SVM | L2 | 1 | sklearn_iforest | 3365.0000 | 0.0000 | 0.3580 | 0.0000 | 0.0477 | 0.0000 | 0.0502 | 0.0000 | 0.1001 | 0.0000 | 0.1825 | 0.0000 | 0.0348 | 0.0000 | 0.2635 | 0.0000 | 1.0000 | 0.0000 | 0.1317 | 0.0000 | 1.0000 | 0.0000 |
| GRU-DeepSVDD | L2 | 1 |  | 3336.0000 | 0.0000 | 0.0005 | 0.0000 | 0.0023 | 0.0000 | 0.0501 | 0.0000 | 0.1001 | 0.0000 | 1.0226 | 0.0000 | 0.1331 | 0.0000 | 0.2455 | 0.0000 | 0.9318 | 0.0000 | 0.1317 | 0.0000 | 1.0000 | 0.0000 |

## Per-seed rows

| model | data_level | seed | n_windows | score_mean | score_std | flag_rate_5pct | flag_rate_10pct | train_seconds | infer_seconds | precision_at_5pct | recall_at_5pct | precision_at_10pct | recall_at_10pct | backend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f-AnoGAN | L1 | 42 | 3716 | 2.5772 | 0.7626 | 0.0501 | 0.1001 | 22.1753 | 0.3596 | 0.1935 | 0.9231 | 0.0995 | 0.9487 | nan |
| GA-EIF+SVM | L1 | 42 | 3745 | 0.7544 | 0.0260 | 0.0654 | 0.1007 | 0.9877 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | sklearn_iforest |
| GRU-DeepSVDD | L1 | 42 | 3716 | 0.0003 | 0.0013 | 0.0501 | 0.1001 | 5.9054 | 0.1763 | 0.1559 | 0.7436 | 0.0995 | 0.9487 | nan |
| f-AnoGAN | L2 | 42 | 3336 | 0.0321 | 0.0407 | 0.0501 | 0.1001 | 4.7405 | 0.4470 | 0.2635 | 1.0000 | 0.1317 | 1.0000 | nan |
| GA-EIF+SVM | L2 | 42 | 3365 | 0.3580 | 0.0477 | 0.0502 | 0.1001 | 0.1825 | 0.0348 | 0.2635 | 1.0000 | 0.1317 | 1.0000 | sklearn_iforest |
| GRU-DeepSVDD | L2 | 42 | 3336 | 0.0005 | 0.0023 | 0.0501 | 0.1001 | 1.0226 | 0.1331 | 0.2455 | 0.9318 | 0.1317 | 1.0000 | nan |
