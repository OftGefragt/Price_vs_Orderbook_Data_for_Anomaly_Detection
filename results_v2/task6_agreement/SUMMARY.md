# Task 6 — Agreement

Deep pairwise Jaccard uses matched dumps seeds 42–44. Forest seed-stability uses 42–51. Chance level = p/(2-p).

level  flag_frac              mode            model_a            model_b  jaccard_mean  jaccard_std  jaccard_min  jaccard_max  chance_jaccard  n_seeds
   L1       0.05 event_cover_gap30 Forest-scaleFreeIF        GA-EIF_dump      0.053490     0.001747     0.051771     0.055263        0.025641        3
   L1       0.05 event_cover_gap30       GRU-DeepSVDD Forest-scaleFreeIF      0.414993     0.023715     0.391061     0.438486        0.025641        3
   L1       0.05 event_cover_gap30       GRU-DeepSVDD        GA-EIF_dump      0.108244     0.009928     0.098765     0.118568        0.025641        3
   L1       0.05 event_cover_gap30           f-AnoGAN Forest-scaleFreeIF      0.518519     0.032075     0.500000     0.555556        0.025641        3
   L1       0.05 event_cover_gap30           f-AnoGAN        GA-EIF_dump      0.127549     0.016087     0.109195     0.139205        0.025641        3
   L1       0.05 event_cover_gap30           f-AnoGAN       GRU-DeepSVDD      0.349436     0.064083     0.296196     0.420561        0.025641        3
   L1       0.05            window Forest-scaleFreeIF        GA-EIF_dump      0.057827     0.003467     0.053824     0.059829        0.025641        3
   L1       0.05            window       GRU-DeepSVDD Forest-scaleFreeIF      0.331096     0.043511     0.291667     0.377778        0.025641        3
   L1       0.05            window       GRU-DeepSVDD        GA-EIF_dump      0.095219     0.006694     0.087719     0.100592        0.025641        3
   L1       0.05            window           f-AnoGAN Forest-scaleFreeIF      0.506523     0.032082     0.488000     0.543568        0.025641        3
   L1       0.05            window           f-AnoGAN        GA-EIF_dump      0.139059     0.021899     0.113772     0.151703        0.025641        3
   L1       0.05            window           f-AnoGAN       GRU-DeepSVDD      0.344610     0.058145     0.296167     0.409091        0.025641        3
   L2       0.05 event_cover_gap30 Forest-scaleFreeIF        GA-EIF_dump      1.000000     0.000000     1.000000     1.000000        0.025641        3
   L2       0.05 event_cover_gap30       GRU-DeepSVDD Forest-scaleFreeIF      0.513946     0.067030     0.471810     0.591241        0.025641        3
   L2       0.05 event_cover_gap30       GRU-DeepSVDD        GA-EIF_dump      0.513946     0.067030     0.471810     0.591241        0.025641        3
   L2       0.05 event_cover_gap30           f-AnoGAN Forest-scaleFreeIF      0.560812     0.002254     0.558952     0.563319        0.025641        3
   L2       0.05 event_cover_gap30           f-AnoGAN        GA-EIF_dump      0.560812     0.002254     0.558952     0.563319        0.025641        3
   L2       0.05 event_cover_gap30           f-AnoGAN       GRU-DeepSVDD      0.550336     0.071543     0.483680     0.625926        0.025641        3
   L2       0.05            window Forest-scaleFreeIF        GA-EIF_dump      1.000000     0.000000     1.000000     1.000000        0.025641        3
   L2       0.05            window       GRU-DeepSVDD Forest-scaleFreeIF      0.410354     0.047634     0.363265     0.458515        0.025641        3
   L2       0.05            window       GRU-DeepSVDD        GA-EIF_dump      0.410354     0.047634     0.363265     0.458515        0.025641        3
   L2       0.05            window           f-AnoGAN Forest-scaleFreeIF      0.523717     0.045480     0.471366     0.553488        0.025641        3
   L2       0.05            window           f-AnoGAN        GA-EIF_dump      0.523717     0.045480     0.471366     0.553488        0.025641        3
   L2       0.05            window           f-AnoGAN       GRU-DeepSVDD      0.489690     0.043128     0.445887     0.532110        0.025641        3
