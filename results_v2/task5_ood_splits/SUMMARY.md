# Task 5 — OOD and split designs

## OOD shares (selected)

level
L1_1min_minmax_fit_BTC    0.804861
L1_1min_raw               0.804861
L2_minmax_fit_BTC         0.000357
L2_raw                    0.000357

## Split designs (forest, primary price-range style labels on test close/mid)

                           setup  n_train  n_test  base_rate  p5_precision  p5_recall   p5_lift  frac_rows_in_tied_score_groups
     L1_5min_rawOHLCV_BTC_to_ETH    17568    3745   0.009957      0.000000       0.00  0.000000                        0.992790
    L1_5min_scaleFree_BTC_to_ETH    17568    3745   0.009957      0.198925       1.00 19.978495                        0.000000
     L1_5min_rawOHLCV_ETH_to_BTC     3745   17568   0.009978      0.103763       0.52 10.399407                        0.998235
    L1_5min_scaleFree_ETH_to_BTC     3745   17568   0.009978      0.199544       1.00 19.998860                        0.000911
L1_5min_scaleFree_BTC_time_split    12297    5271   0.009920      0.198473       1.00 20.007634                        0.002277
L1_5min_scaleFree_ETH_time_split     2621    1124   0.010046      0.200000       1.00 19.909091                        0.001779
            L2_BTC_to_ETH_forest     3367    3365   0.009892      0.197605       1.00 19.976048                        0.000000

## Score vs OOD correlation

                      setup  spearman_score_vs_row_ood_frac  mean_row_ood_frac
L1_5min_rawOHLCV_BTC_to_ETH                        0.426486           0.812977

## Finding

Raw BTC->ETH L1 levels are severely distribution-shifted. Scale-free L1 and within-asset time splits test whether forest failure is OOD-driven. Deep-model OOD re-training is left for a later heavy run if needed.
