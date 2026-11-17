# Changes in the comparison suite

## Strong-protocol update

1. Added `--mode strong|fast` to `paper_eval.run_comparison`.
2. Strong mode uses three seeds, longer deep-model epochs, a 30000-row
   Level-1 train subsample, and the full ETH Level-1 test set.
3. Added weak proxy stress labels and Precision@k / Recall@k.
4. Added pairwise Jaccard agreement tables per data level.
5. Wrote `results/protocol.json` with the exact settings of each run.
6. Extended Isolation Forest is used for moderate sample sizes. Larger
   Level-1 matrices use sklearn IsolationForest with matched hyperparameters.
7. Updated the Colab notebook to default to `MODE = "strong"`.

## Earlier paper-ready release

1. Added `paper_eval/` unified runners.
2. Added Colab notebook and `requirements-paper.txt`.
3. Kept original Level-1 and Level-2 course packages for qualitative figures.
