# Task 7 — Matched controls (forest probe)

                         control  seed  n_train  n_test  n_features  base_rate  p5_precision  p5_recall   p5_lift  p10_precision  p10_recall  p10_lift  p5_lift_minus_L2full
              a_L1_5min_rawOHLCV    42    17568    3745           5   0.009957      0.000000        0.0  0.000000       0.000000         0.0  0.000000            -19.976048
             b_L1_5min_scaleFree    42    17568    3745           5   0.009957      0.198925        1.0 19.978495       0.099462         1.0  9.989247              0.002447
   b2_L1_5min_raw_plus_scaleFree    42    17568    3745          10   0.009957      0.198925        1.0 19.978495       0.099462         1.0  9.989247              0.002447
            c_L2_reduced5_topish    42     3367    3365           5   0.009892      0.197605        1.0 19.976048       0.098802         1.0  9.988024              0.000000
           d_L2_full30_sklearnIF    42     3367    3365          30   0.009892      0.197605        1.0 19.976048       0.098802         1.0  9.988024              0.000000
e_L1_scaleFree_subsample_to_L2_n    42     3367    3745           5   0.009957      0.198925        1.0 19.978495       0.099462         1.0  9.989247              0.002447
     combo_b+e_L1_scaleFree_subN    42     3367    3745           5   0.009957      0.198925        1.0 19.978495       0.099462         1.0  9.989247              0.002447

Finding: scale-free L1 closes the lift gap to full L2 under primary price labels.
