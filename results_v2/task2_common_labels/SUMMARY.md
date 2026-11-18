# Task 2 — Common non-circular labels

## Data fact

L2 GA-EIF test features are numerically identical to the 30 overlapping columns in AnoGAN `ETH_5min.csv`, which also provides `midpoint` and `system_time`. BTC AnoGAN train mid-price file is empty in-repo.

## Alignment

{
  "l1_5min_bars": 3745,
  "l2_bars": 3365,
  "l2_unique_5min_floors": 3365,
  "intersection_5min_timestamps": 3365,
  "l2_time_start": "2021-04-07 11:37:49.861733+00:00",
  "l2_time_end": "2021-04-19 09:50:00.345392+00:00",
  "l1_time_start": "2021-04-06 21:00:00+00:00",
  "l1_time_end": "2021-04-19 21:00:00+00:00"
}

## Prevalence (exact top 1%)

level                 label_kind  n_windows  k_positives  base_rate  target_rate  cutoff_amplitude  n_tied_at_cutoff_amp                     tie_break
   L1        primary_price_range       3716           37   0.009957         0.01        144.340000                     3 amplitude_desc_then_index_asc
   L1           secondary_legacy       3716           37   0.009957         0.01          0.004220                     3 amplitude_desc_then_index_asc
   L1 alternative_abs_log_return       3716           37   0.009957         0.01          0.165770                     1 amplitude_desc_then_index_asc
   L2        primary_price_range       3336           33   0.009892         0.01        161.755000                     1 amplitude_desc_then_index_asc
   L2           secondary_legacy       3336           33   0.009892         0.01          1.880071                    14 amplitude_desc_then_index_asc
   L2 alternative_abs_log_return       3336           33   0.009892         0.01          0.199027                     1 amplitude_desc_then_index_asc

## Seed-42 P@5% under primary price-range label

level          label_kind        model  seed                  score_root    n  flag_frac  k_flags  n_positives  base_rate  precision   recall      lift  max_achievable_precision  tp
   L1 primary_price_range     f-AnoGAN    42 results/matched_2021_spring 3716       0.05      186           37   0.009957   0.193548 0.972973 19.438535                  0.198925  36
   L1 primary_price_range   GA-EIF+SVM    42 results/matched_2021_spring 3716       0.05      186           37   0.009957   0.000000 0.000000  0.000000                  0.198925   0
   L1 primary_price_range GRU-DeepSVDD    42 results/matched_2021_spring 3716       0.05      186           37   0.009957   0.188172 0.945946 18.898576                  0.198925  35
   L2 primary_price_range     f-AnoGAN    42 results/matched_2021_spring 3336       0.05      167           33   0.009892   0.197605 1.000000 19.976048                  0.197605  33
   L2 primary_price_range   GA-EIF+SVM    42 results/matched_2021_spring 3336       0.05      167           33   0.009892   0.197605 1.000000 19.976048                  0.197605  33
   L2 primary_price_range GRU-DeepSVDD    42 results/matched_2021_spring 3336       0.05      167           33   0.009892   0.197605 1.000000 19.976048                  0.197605  33

## Finding

Primary L2 labels are no longer a function of the 30 book features. Compare primary metrics to secondary_legacy to see how much of the old L2 advantage depended on circular feature-norm stress.
