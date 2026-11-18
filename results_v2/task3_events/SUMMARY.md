# Task 3 — Event-level evaluation

## Positive events (gap=30)

level  gap  n_positive_windows  n_positive_events  mean_event_span_windows
   L1   30                  37                  2                     18.5
   L2   30                  33                  1                     33.0

## Event metrics @ top 5% flags (gap=30, seed 42)

level        model  n_label_events  n_pred_events  event_precision  event_precision_ci95_lo  event_precision_ci95_hi  event_recall  event_recall_ci95_lo  event_recall_ci95_hi  event_jaccard_windowcover
   L1     f-AnoGAN             2.0            7.0         0.142857                      0.0                 0.500000           0.5                   0.0                   1.0                   0.189474
   L1   GA-EIF+SVM             2.0            2.0         0.000000                      0.0                 0.000000           0.0                   0.0                   0.0                   0.000000
   L1 GRU-DeepSVDD             2.0           15.0         0.066667                      0.0                 0.357305           0.5                   0.0                   1.0                   0.112540
   L2     f-AnoGAN             1.0            3.0         0.333333                      0.0                 0.666667           1.0                   0.0                   1.0                   0.183333
   L2   GA-EIF+SVM             1.0            3.0         0.333333                      0.0                 0.626042           1.0                   0.0                   1.0                   0.186441
   L2 GRU-DeepSVDD             1.0            9.0         0.111111                      0.0                 0.333333           1.0                   0.0                   1.0                   0.127413

## Finding

Overlapping windows collapse many positive windows into few independent stress episodes. If L2 has very few label events, window-level recall near 1.0 is not evidence of many independent detections.
