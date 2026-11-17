# Main comparison (matched, 2022_bear)

Precision/Recall use weak proxy stress labels (top 1% amplitude windows). Flag rates use a shared top-quantile rule on anomaly scores.

| model | data_level | n_seeds | backend | n_windows_mean | n_windows_std | score_mean_mean | score_mean_std | score_std_mean | score_std_std | flag_rate_5pct_mean | flag_rate_5pct_std | flag_rate_10pct_mean | flag_rate_10pct_std | train_seconds_mean | train_seconds_std | infer_seconds_mean | infer_seconds_std | precision_at_5pct_mean | precision_at_5pct_std | recall_at_5pct_mean | recall_at_5pct_std | precision_at_10pct_mean | precision_at_10pct_std | recall_at_10pct_mean | recall_at_10pct_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f-AnoGAN | L1 | 3 |  | 4003.0000 | 0.0000 | 2.4723 | 0.0041 | 1.0394 | 0.0813 | 0.0502 | 0.0000 | 0.1002 | 0.0000 | 79.1580 | 0.4587 | 0.3220 | 0.0173 | 0.2050 | 0.0000 | 1.0000 | 0.0000 | 0.1025 | 0.0000 | 1.0000 | 0.0000 |
| GA-EIF+SVM | L1 | 3 | sklearn_iforest | 4032.0000 | 0.0000 | 0.7682 | 0.0056 | 0.0249 | 0.0031 | 0.1673 | 0.0048 | 0.1673 | 0.0048 | 0.9382 | 0.0865 | 0.1165 | 0.0043 | 0.0150 | 0.0071 | 0.0732 | 0.0345 | 0.0075 | 0.0035 | 0.0732 | 0.0345 |
| GRU-DeepSVDD | L1 | 3 |  | 4003.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0003 | 0.0000 | 0.0502 | 0.0000 | 0.1002 | 0.0000 | 14.5863 | 0.1170 | 0.1986 | 0.0569 | 0.1583 | 0.0024 | 0.7724 | 0.0115 | 0.0892 | 0.0012 | 0.8699 | 0.0115 |

## Per-seed rows

| model | data_level | seed | n_windows | score_mean | score_std | flag_rate_5pct | flag_rate_10pct | train_seconds | infer_seconds | precision_at_5pct | recall_at_5pct | precision_at_10pct | recall_at_10pct | backend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f-AnoGAN | L1 | 42 | 4003 | 2.4693 | 0.9880 | 0.0502 | 0.1002 | 79.0477 | 0.2994 | 0.2050 | 1.0000 | 0.1025 | 1.0000 | nan |
| GA-EIF+SVM | L1 | 42 | 4032 | 0.7754 | 0.0292 | 0.1627 | 0.1627 | 0.8629 | 0.1153 | 0.0200 | 0.0976 | 0.0100 | 0.0976 | sklearn_iforest |
| GRU-DeepSVDD | L1 | 42 | 4003 | 0.0002 | 0.0004 | 0.0502 | 0.1002 | 14.4278 | 0.1562 | 0.1550 | 0.7561 | 0.0875 | 0.8537 | nan |
| f-AnoGAN | L1 | 43 | 4003 | 2.4695 | 0.9760 | 0.0502 | 0.1002 | 79.7669 | 0.3249 | 0.2050 | 1.0000 | 0.1025 | 1.0000 | nan |
| GA-EIF+SVM | L1 | 43 | 4032 | 0.7618 | 0.0233 | 0.1654 | 0.1654 | 1.0594 | 0.1223 | 0.0200 | 0.0976 | 0.0100 | 0.0976 | sklearn_iforest |
| GRU-DeepSVDD | L1 | 43 | 4003 | 0.0002 | 0.0003 | 0.0502 | 0.1002 | 14.6247 | 0.1605 | 0.1600 | 0.7805 | 0.0900 | 0.8780 | nan |
| f-AnoGAN | L1 | 44 | 4003 | 2.4781 | 1.1542 | 0.0502 | 0.1002 | 78.6596 | 0.3415 | 0.2050 | 1.0000 | 0.1025 | 1.0000 | nan |
| GA-EIF+SVM | L1 | 44 | 4032 | 0.7673 | 0.0221 | 0.1739 | 0.1739 | 0.8922 | 0.1120 | 0.0050 | 0.0244 | 0.0025 | 0.0244 | sklearn_iforest |
| GRU-DeepSVDD | L1 | 44 | 4003 | 0.0002 | 0.0002 | 0.0502 | 0.1002 | 14.7066 | 0.2790 | 0.1600 | 0.7805 | 0.0900 | 0.8780 | nan |
