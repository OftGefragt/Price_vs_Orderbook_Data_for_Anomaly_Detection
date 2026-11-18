# Task 9 — Classical baselines

       level           model  precision   recall      lift  base_rate
L1_scaleFree IsolationForest   0.198925 1.000000 19.978495   0.009957
L1_scaleFree             LOF   0.161290 0.810811 16.198779   0.009957
L1_scaleFree     OneClassSVM   0.193548 0.972973 19.438535   0.009957
 L1_rawOHLCV IsolationForest   0.000000 0.000000  0.000000   0.009957
 L1_rawOHLCV             LOF   0.172043 0.864865 17.278698   0.009957
 L1_rawOHLCV     OneClassSVM   0.182796 0.918919 18.358617   0.009957
          L2 IsolationForest   0.197605 1.000000 19.976048   0.009892
          L2             LOF   0.197605 1.000000 19.976048   0.009892
          L2     OneClassSVM   0.197605 1.000000 19.976048   0.009892

       level         model_a     model_b  jaccard_top5  chance_jaccard
L1_scaleFree IsolationForest         LOF      0.087719        0.025641
L1_scaleFree IsolationForest OneClassSVM      0.524590        0.025641
L1_scaleFree             LOF OneClassSVM      0.087719        0.025641
 L1_rawOHLCV IsolationForest         LOF      0.000000        0.025641
 L1_rawOHLCV IsolationForest OneClassSVM      0.261017        0.025641
 L1_rawOHLCV             LOF OneClassSVM      0.094118        0.025641
          L2 IsolationForest         LOF      0.525114        0.025641
          L2 IsolationForest OneClassSVM      0.876404        0.025641
          L2             LOF OneClassSVM      0.605769        0.025641
