# Matched 2021 spring: L1 5-min vs L2

- L1 bar: 5-minute aggregate of bundled 1-minute BTC/ETH
- Forest backend: sklearn IsolationForest on **both** tiers
- Seeds: 42, 43, 44

| Model | L1 P@5 | L2 P@5 | L2 advantage holds? |
|---|---:|---:|---|
| f-AnoGAN | 0.194 | 0.263 | yes |
| GA-EIF+SVM | 0.000 | 0.263 | yes |
| GRU-DeepSVDD | 0.190 | 0.246 | yes |

Conclusion: after removing the resolution confound and standardizing the forest backend, L2 still improves proxy Precision@5%.
