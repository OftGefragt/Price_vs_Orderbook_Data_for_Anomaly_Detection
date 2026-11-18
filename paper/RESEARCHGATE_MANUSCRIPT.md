# Cross-Asset Transfer for Crypto Anomaly Detection: Level-1 vs Level-2 Features under Deep and Classical Models

**Author**  
Melike Kaya  
University of Eastern Finland (coursework extension) / Independent researcher  
Contact: via GitHub issues at the repository below (or replace with your email before upload)

**Version:** Preprint, November 2026  
**Code and tables:** https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection  

*(OftGefragt is the GitHub account that hosts the code, not a co-author.)*

---

## Abstract

Unsupervised anomaly detectors for cryptocurrency markets are often trained on one asset and applied to another. This paper studies that **cross-asset transfer** setting (Bitcoin train → Ethereum test) and asks whether Level-2 (L2) order-book features are necessary for useful ranking of extreme price windows, relative to Level-1 (L1) OHLCV inputs, for both deep and classical detectors. We evaluate an f-AnoGAN-style convolutional model, a GRU encoder with a Deep SVDD objective, and Isolation Forest–family models (including a genetically configured forest pipeline). Ranking uses Precision@k and Recall@k against exact top-1% price-range proxy labels, with lift and event-level checks.

An exploratory matched protocol with unequal label prevalence suggested an L2 Precision@5% advantage (~0.26 vs ~0.19). After equalizing prevalence, switching to non-circular price-range labels, and correcting L1 absolute-price scale for classical models, that advantage largely disappears for deep detectors (~0.19–0.20 on both tracks). Raw L1 forests fail under BTC→ETH scale shift (Precision@5% ≈ 0), whereas deep L1 models transfer successfully; **scale-free L1 forests match L2 forests** (~0.20). Scale-free deep re-training on GPU confirms L1 f-AnoGAN ≈ L2 deep (~0.194 vs 0.198). We conclude that **cross-asset transfer is viable without L2 data** when scale is handled, and that earlier L2 gains were largely label and scale artifacts. Evidence is limited by few independent stress events (1–2) in the Ethereum window.

**Keywords:** anomaly detection; cryptocurrency; cross-asset transfer; distribution shift; limit order book; AnoGAN; isolation forest; Deep SVDD

---

## 1. Introduction

Cryptocurrency markets are liquid, continuous, and frequently exposed to abrupt stress and alleged manipulation (Gandal et al., 2018; Griffin & Shams, 2020; Kamps & Kleinberg, 2018). Surveillance systems therefore need methods that flag unusual intervals without waiting for costly event labels (Chandola et al., 2009; La Morgia et al., 2023).

Two design choices matter in practice. First, **representation**: Level-1 (L1) OHLCV bars summarize traded prices and volumes, whereas Level-2 (L2) order-book snapshots expose depth, imbalance, and notional flow (Gould et al., 2013; Zhang et al., 2019). Book features help predictive microstructure tasks (Kercheval & Zhang, 2015; Tsantekidis et al., 2017; Sirignano & Cont, 2019), but it is unclear whether they are required for unsupervised stress ranking. Second, **transfer**: models are often trained where data are abundant (e.g., Bitcoin) and applied to another asset (e.g., Ethereum). Absolute price levels, volatility regimes, and liquidity differ across assets (Cont, 2001), so classical detectors that rely on raw feature magnitudes can fail under shift even when deep models succeed.

This paper reframes the L1-versus-L2 comparison around that transfer problem. We ask: under BTC→ETH unsupervised training, do deep and classical detectors need order-book depth, or can scale-aware L1 features already support cross-asset anomaly ranking?

### Contributions

1. A Bitcoin-train / Ethereum-test comparison of L1 versus L2 inputs for deep (f-AnoGAN-style; GRU–Deep SVDD) and classical (Isolation Forest–family) detectors (Schlegl et al., 2017, 2019; Cho et al., 2014; Ruff et al., 2018; Liu et al., 2008; Hariri et al., 2021).
2. A confound-controlled evaluation: exact top-1% **price-range** labels, prevalence equalization, event merging, scale-free L1 probes, and scale-free deep re-training.
3. A corrected interpretation of an earlier exploratory L2 Precision@5% gain (~0.26): after controls, deep L1 ≈ deep L2, and classical L1 recovers once features are scale-free—supporting **cross-asset transfer without requiring L2**.
4. Public code, seeds, protocol metadata, and methodology tables (`results/` and `results_v2/`).

---

## 2. Related Work

### 2.1 Unsupervised anomaly detection

Labels are scarce and evaluation is fragile under class imbalance (Chandola et al., 2009; Aggarwal, 2017). Isolation Forest isolates rare points with random partitions (Liu et al., 2008); Extended Isolation Forest reduces axis-aligned score artifacts (Hariri et al., 2021). Density and support estimators such as LOF and one-class SVM remain common baselines (Breunig et al., 2000; Schölkopf et al., 2001; Tax & Duin, 2004).

Deep methods are reviewed by Ruff et al. (2021) and Pang et al. (2021). AnoGAN introduced GAN-based anomaly scoring (Schlegl et al., 2017); f-AnoGAN replaces slow latent search with a trained encoder (Schlegl et al., 2019). Deep SVDD learns a hypersphere of normal embeddings (Ruff et al., 2018). Sequence models extend these ideas to time series (Malhotra et al., 2015; Hundman et al., 2018; Audibert et al., 2020). Precision–recall summaries are preferred to ROC under imbalance (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015).

### 2.2 Order books, crypto surveillance, and shift

Limit order books encode supply and demand beyond the last trade (Gould et al., 2013; Cont & de Larrard, 2013). Machine learning models use book snapshots for mid-price and return prediction (Kercheval & Zhang, 2015; Tsantekidis et al., 2017; Zhang et al., 2019). Cryptocurrency research documents manipulation and pump-and-dump organization (Gandal et al., 2018; Xu & Livshits, 2019; La Morgia et al., 2020, 2023). Many detectors still start from OHLCV or social signals (Kamps & Kleinberg, 2018). Recent work mixes book pressure into early-warning scores (Bolz et al., 2025; Letteri, 2025), but few studies isolate L1 versus L2 **under explicit cross-asset transfer** with label and scale controls.

### 2.3 Gap

We focus on that gap: fixed detectors, BTC→ETH transfer, two input levels, deep versus classical models, and ranking metrics that do not require confirmed fraud labels—while diagnosing prevalence, circular labels, and absolute-price shift.

---

## 3. Data

### 3.1 Level 1 (matched protocol)

| Split | Asset | Resolution | Rows | Features |
|---|---|---|---:|---|
| Train | BTC | 5-minute aggregate of 1-minute OHLCV | 17,568 | Open, High, Low, Close, Volume |
| Test | ETH | 5-minute aggregate of 1-minute OHLCV | 3,745 | same |

The underlying 1-minute series covers spring 2021 (BTC train about 2021-03-31 to 2021-05-31; ETH test about 2021-04-06 to 2021-04-19).

**Scale-free L1 variant.** For classical and controlled deep runs, absolute OHLCV levels are replaced by returns / relative transforms so BTC and ETH share a comparable numeric regime (details in `results_v2/task5_ood_splits/` and `results_v2/scalefree_deep/`).

### 3.2 Level 2

| Split | Asset | Resolution | Rows | Features |
|---|---|---|---:|---:|
| Train | BTC | 5 minute | 3,367 | 30 selected depth/notional statistics |
| Test | ETH | 5 minute | 3,365 | same 30 features |

L1 and L2 do not share identical timestamps. After 5-minute aggregation, a window of length 30 covers a comparable clock span on both tracks. Panels are still not row-aligned; they remain two experimental tracks.

### 3.3 Secondary regime (Level-1 only)

A second Level-1 window uses Binance Vision 5-minute BTCUSDT and ETHUSDT klines for May–June 2022 (Luna-period stress). Training uses BTC from 2022-05-01 to 2022-06-30; testing uses ETH from 2022-06-10 to 2022-06-23. Historical L2 order-book matrices for that window are not bundled, so the regime check is Level-1 only.

### 3.4 Preprocessing

A MinMax scaler is fit on the training split only (track-specific). Sequence models use sliding windows of length 30 with step 1.

---

## 4. Methods

### 4.1 f-AnoGAN-style 1D detector

Following the f-AnoGAN idea of an encoder plus generator and discriminator (Schlegl et al., 2019), a compact 1D convolutional stack is trained on normal windows. The anomaly score is

\[
s = 1.0\cdot \ell_{\mathrm{rec}} + 0.1\cdot \ell_{\mathrm{feat}},
\]

where \(\ell_{\mathrm{rec}}\) is reconstruction error and \(\ell_{\mathrm{feat}}\) is a discriminator feature residual. Epochs in the main deep runs: 10 (GAN) + 15 (encoder).

### 4.2 Classical forest pipeline

Hyperparameters follow a genetic search recorded in the project notebook: 235 trees, subsample size 1691, contamination 0.062 (Holland, 1992; Eiben & Smith, 2015). Continuous forest scores are used for ranking. A one-class SVM stage exists in the pipeline but **does not determine ranking** in the released metric path; claims refer to Isolation Forest / Extended Isolation Forest scores (Liu et al., 2008; Hariri et al., 2021; Pedregosa et al., 2011). Additional classical baselines (LOF, plain IF, one-class SVM) are reported in `results_v2/task9_baselines/`.

### 4.3 GRU encoder with Deep SVDD loss

A two-layer GRU (Cho et al., 2014) maps each window to an embedding. Training minimizes distance to a hypersphere center updated toward the batch mean, in the spirit of Deep SVDD (Ruff et al., 2018). Epochs: 8 in the reported deep runs.

---

## 5. Experimental Setup

| Item | Setting |
|---|---|
| Seeds | 42, 43, 44 (deep); forest probes extend further where noted |
| Train asset | BTC |
| Test asset | ETH |
| Transfer | Models are **not** fine-tuned on ETH |
| L1 bar | 5-minute aggregate (matched) |
| Flag rule | top 5% and top 10% scores |
| Primary labels | exact top 1% **price-range** windows (L1 Close; L2 midpoint), deterministic tie-break |
| Legacy labels | inclusive quantile / L2 feature-norm amplitude (exploratory only) |
| Metrics | Precision@k, Recall@k, lift = precision / base rate; event merge gap = 30 |
| Implementation | PyTorch and scikit-learn (Paszke et al., 2019; Pedregosa et al., 2011) |

**Precision at k (P@k)** is the fraction of the top-\(k\) highest-scoring windows that fall in the proxy stress set. **Recall at k (R@k)** is the fraction of all proxy stress windows recovered inside that top-\(k\) budget. Here \(k\) is 5% or 10% of scored windows.

Methodology audits, prevalence tables, and controls live under `results_v2/` (Tasks 1–10 and `scalefree_deep/`). Exploratory baseline tables remain frozen in `results/baseline_v1/` and `results/matched_2021_spring/`.

---

## 6. Results

### 6.1 Exploratory matched protocol (legacy labels)

Mean Precision/Recall at 5% over three seeds under the original matched protocol (`results/matched_2021_spring/`), which used inclusive quantile labels and L2 feature-norm amplitudes:

| Model | Level | P@5% | R@5% |
|---|---|---:|---:|
| f-AnoGAN | L1 | 0.194 | 0.923 |
| GRU-DeepSVDD | L1 | 0.190 | 0.906 |
| GA-EIF+SVM (forest scores) | L1 | 0.000 | 0.000 |
| f-AnoGAN | L2 | 0.263 | 1.000 |
| GA-EIF+SVM (forest scores) | L2 | 0.263 | 1.000 |
| GRU-DeepSVDD | L2 | 0.246 | 0.932 |

This table suggested a clear L2 advantage. The remainder of Section 6 shows why that reading does not survive label and scale controls.

### 6.2 Prevalence and label controls

L2 inclusive-quantile labels overstated the positive share (~1.32% vs ~1.0%) because of ties at the cutoff. With **exact top-1%** labels, L2 deep/forest Precision@5% falls from ~0.263 to ~0.198 (near the maximum achievable precision at a 5% flag budget). Under **primary price-range** labels (non-circular for L2), deep models show **no meaningful L1–L2 precision gap** (seed-42 P@5 ≈ 0.19 on both tracks; see `results_v2/task1_label_prevalence/` and `task2_common_labels/`).

### 6.3 Cross-asset scale shift: deep versus classical on L1

Under primary-style price labels, forest Precision@5% on **raw** L1 BTC→ETH is **0.00**, with a large fraction of ETH rows looking out-of-distribution relative to BTC absolute levels (~0.81 mean OOD fraction). The same forest on **scale-free L1** reaches Precision@5% ≈ **0.20**, matching L2 forest lift (~20). Deep L1 models already succeed on raw L1 (~0.19), indicating greater robustness to absolute-price shift without hand-crafted normalization (`results_v2/task5_ood_splits/`, `task7_controls/`).

### 6.4 Scale-free deep comparison (main controlled result)

Colab GPU re-training (seeds 42–44; primary price-range labels; exact top 1%) compares scale-free L1 against L2:

| Track | Model | P@5 mean | P@5 std | Lift@5 |
|---|---|---:|---:|---:|
| L1 scale-free | f-AnoGAN | 0.194 | 0.005 | 19.4 |
| L1 scale-free | GRU-DeepSVDD | 0.115 | 0.025 | 11.5 |
| L2 | f-AnoGAN | 0.198 | 0.000 | 20.0 |
| L2 | GRU-DeepSVDD | 0.198 | 0.000 | 20.0 |

Source: `results_v2/scalefree_deep/`. **L1 scale-free f-AnoGAN ≈ L2 deep.** Scale-free L1 GRU remains weaker than L2 GRU in this run.

### 6.5 Event-level caution

Merging overlapping positive windows (gap = 30) yields only **two** independent L1 events and **one** L2 event in the Ethereum test window (`results_v2/task3_events/`). Window-level recall near 1.0 largely means recovering that thin event set, not many distinct market episodes.

### 6.6 Secondary checks

On May–June 2022 5-minute OHLCV (L1 only), deep detectors again recover extreme amplitude windows after BTC→ETH transfer (exploratory matched numbers in `results/matched_2022_bear/`). Flagged windows also show higher realized volatility / volume in an economic sanity check (`results_v2/task10_economic/`). Distance-related L2 feature groups dominate notional-only groups in a forest ablation (`results_v2/task8_feature_groups/`), but that does not restore a claim of deep-book necessity under price labels.

---

## 7. Discussion

The controlled results support a **cross-asset transfer** reading rather than an **L2 superiority** reading.

1. **Transfer without ETH training works** for ranking extreme price-range windows, for deep L1 models and for classical models once L1 is scale-free.
2. **Deep models are more robust to raw cross-asset scale shift** on L1 than Isolation Forest on absolute OHLCV; the gap closes after scale-free transforms.
3. **L2 is not clearly required** for this ranking task once prevalence and labels are equalized: deep L1 f-AnoGAN matches L2 deep Precision@5%.
4. The exploratory ~0.26 L2 Precision@5% figure should not be used as evidence of book-feature value; it mixed higher L2 base rate (ties) with a circular L2 amplitude and a raw-L1 forest failure.

These findings are compatible with microstructure work showing that books contain predictive information (Zhang et al., 2019; Sirignano & Cont, 2019), while clarifying that **for unsupervised BTC→ETH stress ranking under price-range proxies**, transferable L1 representations can suffice.

---

## 8. Limitations

1. Proxy stress labels are not exchange-confirmed manipulation events (Kamps & Kleinberg, 2018; Xu & Livshits, 2019).
2. L1 and L2 panels are not timestamp-aligned; comparisons are track-wise, not row-aligned.
3. Independent stress events are very few (1–2) in the reported Ethereum window.
4. Historical Level-2 archives for 2022 were not available without paid vendors; the second regime is Level-1 only.
5. Deep multi-seed coverage is concentrated on seeds 42–44; some ablations are forest-only.
6. The SVM stage in the forest pipeline is not used for metric ranking in the released path.

---

## 9. Conclusion

This paper compared Level-1 and Level-2 cryptocurrency representations for unsupervised anomaly detection under **cross-asset transfer** (Bitcoin → Ethereum), using both deep and classical models. After correcting label prevalence, adopting non-circular price-range labels, and controlling absolute-price scale, deep L1 and L2 rankings align (~0.19 Precision@5%), and scale-free L1 forests match L2 forests. Deep models transfer more gracefully on raw L1; classical models transfer once features are scale-aware. Order-book depth is therefore **optional, not necessary**, for this transfer-based ranking task. The repository releases code and controlled tables for reproduction and preprint deposit.

---

## Data and code availability

Source code and result tables:  
https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection

Primary manuscript: `paper/RESEARCHGATE_MANUSCRIPT.md`  
Word manuscript (APA7): `paper/Kaya_CrossAsset_Anomaly_Detection.docx`  
Printable HTML: `paper/RESEARCHGATE_MANUSCRIPT.html`  
Controlled methodology report: `results_v2/REPORT.md`

Reproduce key evaluations with:

```text
python -m paper_eval.run_comparison --mode matched --regime 2021_spring
python -m paper_eval.v2.run_direction2_cpu
python -m paper_eval.v2.run_scalefree_deep --mode full
```

See also `docs/DIRECTION2_COLAB.md` and `paper/HOW_TO_PUBLISH.md`.

---

## Acknowledgments

Parts of the model packages began as coursework in the Artificial Intelligence course at the University of Eastern Finland (2025).

---

## References

Aggarwal, C. C. (2017). *Outlier analysis* (2nd ed.). Springer. https://doi.org/10.1007/978-3-319-47578-3

Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A. (2020). USAD: Unsupervised anomaly detection on multivariate time series. In *Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 3395–3404). https://doi.org/10.1145/3394486.3403392

Bolz, M., Bründler, K., Kane, L., Patsias, P., Tessendorf, L., Gogol, K. M., Kim, T., & Tessone, C. J. (2025). Real-time machine learning detection of Telegram-based pump-and-dump schemes. In *Proceedings of the ACM CCS Workshop on Decentralized Finance and Security (DeFi'25)*. https://arxiv.org/abs/2412.18848

Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *ACM SIGMOD Record, 29*(2), 93–104. https://doi.org/10.1145/335191.335388

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys, 41*(3), 1–58. https://doi.org/10.1145/1541880.1541882

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). *Learning phrase representations using RNN encoder–decoder for statistical machine translation*. arXiv. https://arxiv.org/abs/1406.1078

Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance, 1*(2), 223–236. https://doi.org/10.1080/713665670

Cont, R., & de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM Journal on Financial Mathematics, 4*(1), 1–25. https://doi.org/10.1137/110856605

Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233–240). https://doi.org/10.1145/1143844.1143874

Eiben, A. E., & Smith, J. E. (2015). *Introduction to evolutionary computing* (2nd ed.). Springer. https://doi.org/10.1007/978-3-662-44874-8

Gandal, N., Hamrick, J. T., Moore, T., & Oberman, T. (2018). Price manipulation in the Bitcoin ecosystem. *Journal of Monetary Economics, 95*, 86–96. https://doi.org/10.1016/j.jmoneco.2017.12.004

Gould, M. D., Porter, M. A., Williams, S., McDonald, M., Fenn, D. J., & Howison, S. D. (2013). Limit order books. *Quantitative Finance, 13*(11), 1709–1742. https://doi.org/10.1080/14697688.2013.803148

Griffin, J. M., & Shams, A. (2020). Is Bitcoin really untethered? *The Journal of Finance, 75*(4), 1913–1964. https://doi.org/10.1111/jofi.12903

Hariri, S., Carrasco Kind, M., & Brunner, R. J. (2021). Extended Isolation Forest. *IEEE Transactions on Knowledge and Data Engineering, 33*(4), 1479–1489. https://doi.org/10.1109/TKDE.2019.2947676

Holland, J. H. (1992). *Adaptation in natural and artificial systems*. MIT Press.

Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 387–395). https://doi.org/10.1145/3219819.3219845

Kamps, J., & Kleinberg, B. (2018). To the moon: Defining and detecting cryptocurrency pump-and-dumps. *Crime Science, 7*(1), 1–18. https://doi.org/10.1186/s40163-018-0093-5

Kercheval, A. N., & Zhang, Y. (2015). Modelling high-frequency limit order book dynamics with support vector machines. *Quantitative Finance, 15*(8), 1315–1329. https://doi.org/10.1080/14697688.2015.1032546

La Morgia, M., Mei, A., Sassi, F., & Stefa, J. (2020). Pump and dumps in the Bitcoin era: Real time detection of cryptocurrency market manipulations. In *International Conference on Computer Communications and Networks (ICCCN)*. https://doi.org/10.1109/ICCCN49398.2020.9209660

La Morgia, M., Mei, A., Sassi, F., & Stefa, J. (2023). A survey on pump and dump detection in the cryptocurrency market using machine learning. *Future Internet, 15*(8), Article 267. https://doi.org/10.3390/fi15080267

Letteri, I. (2025). *A comparative analysis of statistical and machine learning models for outlier detection in Bitcoin limit order books*. arXiv. https://arxiv.org/abs/2507.14960

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. In *Proceedings of the Eighth IEEE International Conference on Data Mining* (pp. 413–422). IEEE. https://doi.org/10.1109/ICDM.2008.17

Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015). Long short term memory networks for anomaly detection in time series. In *ESANN*.

Pang, G., Shen, C., Cao, L., & Hengel, A. V. D. (2021). Deep learning for anomaly detection: A review. *ACM Computing Surveys, 54*(2), 1–38. https://doi.org/10.1145/3439950

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems* (Vol. 32).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Müller, E., & Kloft, M. (2018). Deep one-class classification. In *Proceedings of the 35th International Conference on Machine Learning* (pp. 4393–4402). PMLR.

Ruff, L., Kauffmann, J. R., Vandermeulen, R. A., Montavon, G., Samek, W., Kloft, M., Dietterich, T. G., & Müller, K.-R. (2021). A unifying review of deep and shallow anomaly detection. *Proceedings of the IEEE, 109*(5), 756–795. https://doi.org/10.1109/JPROC.2021.3052449

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432

Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. (2017). Unsupervised anomaly detection with generative adversarial networks to guide marker discovery. In *Information Processing in Medical Imaging* (pp. 146–157). Springer. https://doi.org/10.1007/978-3-319-59050-9_12

Schlegl, T., Seeböck, P., Waldstein, S. M., Langs, G., & Schmidt-Erfurth, U. (2019). f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks. *Medical Image Analysis, 54*, 30–44. https://doi.org/10.1016/j.media.2019.01.010

Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation, 13*(7), 1443–1471. https://doi.org/10.1162/089976601750264965

Sirignano, J., & Cont, R. (2019). Universal features of price formation in financial markets: Perspectives from deep learning. *Quantitative Finance, 19*(9), 1449–1459. https://doi.org/10.1080/14697688.2019.1633012

Tax, D. M. J., & Duin, R. P. W. (2004). Support vector data description. *Machine Learning, 54*(1), 45–66. https://doi.org/10.1023/B:MACH.0000008084.60811.49

Tsantekidis, A., Passalis, N., Tefas, A., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2017). Forecasting stock prices from the limit order book using convolutional neural networks. In *IEEE Conference on Business Informatics* (pp. 7–12). https://doi.org/10.1109/CBI.2017.23

Xu, J., & Livshits, B. (2019). The anatomy of a cryptocurrency pump-and-dump scheme. In *28th USENIX Security Symposium* (pp. 1609–1625).

Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing, 67*(11), 3001–3012. https://doi.org/10.1109/TSP.2019.2907260

Zhou, C., & Paffenroth, R. C. (2017). Anomaly detection with robust deep autoencoders. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 665–674). https://doi.org/10.1145/3097983.3098052

Zong, B., Song, Q., Min, M. R., Cheng, W., Lumezanu, C., Cho, D., & Chen, H. (2018). Deep autoencoding Gaussian mixture model for unsupervised anomaly detection. In *International Conference on Learning Representations*.
