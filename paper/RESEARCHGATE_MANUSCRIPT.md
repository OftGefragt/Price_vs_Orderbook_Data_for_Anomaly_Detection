# Level-1 Price versus Level-2 Order-Book Features for Unsupervised Anomaly Detection in Cryptocurrency Markets

**Author**  
Melike Kaya  
University of Eastern Finland (coursework extension) / Independent researcher  
Email: (add your preferred contact email before upload)

**Version:** Technical report, March 2026  
**Code:** https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection  
*(OftGefragt is the GitHub account name that hosts the code, not a co-author.)*

---

## Abstract

Unsupervised anomaly detectors are often trained on cryptocurrency price series, yet it is unclear whether richer Level-2 (L2) order-book features improve detection over Level-1 (L1) open-high-low-close-volume (OHLCV) inputs. This paper compares three detector families under one protocol: a one-dimensional f-AnoGAN-style model, an Extended Isolation Forest configuration selected by a genetic search and paired with a one-class SVM stage, and a GRU encoder trained with a Deep SVDD objective. Models are trained on Bitcoin and evaluated on Ethereum. Ranking quality is measured with Precision at k and Recall at k against weak proxy stress labels (extreme amplitude windows), together with cross-model agreement of top-quantile flags. Across three random seeds, L2 inputs raise Precision at 5% from about 0.18 on L1 to about 0.26 on L2 for f-AnoGAN and for the forest pipeline, with Recall at 5% reaching 1.0 on L2 for those two families. Pairwise Jaccard agreement among top-5% flags is also higher on L2 than on L1. Code, seeds, and protocol files are released. The study does not claim verified manipulation labels; proxy stress labels and sampling-rate differences remain important limitations.

**Keywords:** anomaly detection; cryptocurrency; limit order book; AnoGAN; isolation forest; Deep SVDD

---

## 1. Introduction

Cryptocurrency markets are liquid, continuous, and frequently exposed to abrupt stress and alleged manipulation (Gandal et al., 2018; Griffin and Shams, 2020; Kamps and Kleinberg, 2018). Surveillance systems therefore need methods that flag unusual intervals without waiting for costly event labels (Chandola et al., 2009; La Morgia et al., 2023).

A practical design choice is the input representation. Level-1 data summarize traded prices and volumes. Level-2 order-book data expose depth, imbalance, and cancel or limit flow (Gould et al., 2013; Zhang et al., 2019). Order-book features help in predictive microstructure tasks (Kercheval and Zhang, 2015; Tsantekidis et al., 2017; Sirignano and Cont, 2019), and recent pump-and-dump detectors sometimes include book pressure statistics (Bolz et al., 2025). It is less clear, under a single unsupervised protocol, whether L2 features improve anomaly ranking relative to L1 features across several modern detector families.

### Contributions

1. A matched Bitcoin-train / Ethereum-test comparison of L1 versus L2 inputs for three unsupervised families: f-AnoGAN-style scoring (Schlegl et al., 2017, 2019), GA-configured Extended Isolation Forest with a one-class SVM stage (Hariri et al., 2021; Liu et al., 2008), and GRU with Deep SVDD (Cho et al., 2014; Ruff et al., 2018).
2. A shared top-quantile flagging rule plus Precision/Recall against weak proxy stress labels, with multi-seed aggregation.
3. Public code, protocol metadata, and tables that reproduce the reported numbers.

---

## 2. Related Work

### 2.1 Unsupervised anomaly detection

Classical surveys emphasize that labels are scarce and that evaluation is fragile under class imbalance (Chandola et al., 2009; Aggarwal, 2017). Isolation Forest isolates rare points with random partitions (Liu et al., 2008). Extended Isolation Forest replaces axis-aligned cuts with random hyperplanes and reduces score artifacts (Hariri et al., 2021). Density and support estimators such as LOF and one-class SVM remain common baselines (Breunig et al., 2000; Scholkopf et al., 2001; Tax and Duin, 2004).

Deep methods are reviewed by Ruff et al. (2021) and Pang et al. (2021). Autoencoder and mixture hybrids score reconstruction or density in latent space (Zhou and Paffenroth, 2017; Zong et al., 2018). AnoGAN introduced GAN-based anomaly scoring through reconstruction and discriminator features (Schlegl et al., 2017); f-AnoGAN replaces slow latent search with a trained encoder (Schlegl et al., 2019). Deep SVDD learns a hypersphere of normal embeddings (Ruff et al., 2018). Sequence models extend these ideas to time series (Malhotra et al., 2015; Hundman et al., 2018; Audibert et al., 2020).

### 2.2 Order books and crypto surveillance

Limit order books encode supply and demand beyond the last trade (Gould et al., 2013; Cont and de Larrard, 2013). Machine learning models use book snapshots for mid-price and return prediction (Kercheval and Zhang, 2015; Tsantekidis et al., 2017; Zhang et al., 2019). Cryptocurrency research documents manipulation and pump-and-dump organization (Gandal et al., 2018; Xu and Livshits, 2019; La Morgia et al., 2020, 2023). Many detectors still start from OHLCV or social signals (Kamps and Kleinberg, 2018). Work that mixes book features into early-warning scores suggests microstructure cues matter (Bolz et al., 2025; Letteri, 2025), but few studies isolate the L1 versus L2 representation effect inside the same unsupervised training recipe.

### 2.3 Gap

This paper focuses on that missing comparison: fixed detectors, fixed train/test assets, two input levels, and ranking metrics that do not require confirmed fraud labels.

---

## 3. Data

### 3.1 Level 1

| Split | Asset | Resolution | Rows used in strong run | Features |
|---|---|---|---:|---|
| Train | BTC | 1 minute | 30,000 (subsample of 87,821) | Open, High, Low, Close, Volume |
| Test | ETH | 1 minute | 18,721 (full) | same |

### 3.2 Level 2

| Split | Asset | Resolution | Rows | Features |
|---|---|---|---:|---:|
| Train | BTC | 5 minute | 3,367 | 30 selected depth/notional statistics |
| Test | ETH | 5 minute | 3,365 | same 30 features |

L1 and L2 do not share identical timestamps or sampling rates. They are treated as two experimental tracks rather than as aligned panels.

### 3.3 Preprocessing

A MinMax scaler is fit on the training split only. Sequence models use sliding windows of length 30 with step 1.

---

## 4. Methods

### 4.1 f-AnoGAN-style 1D detector

Following the f-AnoGAN idea of an encoder plus generator and discriminator (Schlegl et al., 2019), a compact 1D convolutional stack is trained on normal windows. The anomaly score is

\[
s = 1.0\cdot \ell_{\mathrm{rec}} + 0.1\cdot \ell_{\mathrm{feat}},
\]

where \(\ell_{\mathrm{rec}}\) is reconstruction error and \(\ell_{\mathrm{feat}}\) is a discriminator feature residual. Strong-protocol epochs: 10 (GAN) + 15 (encoder).

### 4.2 GA-configured forest with one-class SVM

Hyperparameters follow a genetic search recorded in the project notebook: 235 trees, subsample size 1691, contamination 0.062, extension level 26 when dimensionality allows (level 4 on L1). The search follows standard evolutionary model selection practice (Holland, 1992; Eiben and Smith, 2015). Continuous forest scores are used for ranking. On Level-2 the Extended Isolation Forest implementation is used (Hariri et al., 2021). On large Level-1 matrices, scikit-learn Isolation Forest is used with matched tree and subsample settings for runtime (Liu et al., 2008; Pedregosa et al., 2011). The backend name is logged.

### 4.3 GRU encoder with Deep SVDD loss

A two-layer GRU (Cho et al., 2014) maps each window to an embedding. Training minimizes distance to a hypersphere center updated toward the batch mean, in the spirit of Deep SVDD (Ruff et al., 2018). Strong-protocol epochs: 8.

---

## 5. Experimental Setup

| Item | Strong protocol |
|---|---|
| Seeds | 42, 43, 44 |
| Train asset | BTC |
| Test asset | ETH |
| Flag rule | top 5% and top 10% scores |
| Proxy labels | top 1% amplitude windows (Close range on L1; feature-norm range on L2) |
| Implementation | PyTorch and scikit-learn (Paszke et al., 2019; Pedregosa et al., 2011) |

Precision at k and Recall at k are preferred over ROC summaries under imbalance (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015). Exact counts are stored with the released `protocol.json` file.

---

## 6. Results

### 6.1 Main comparison (Table 1)

Mean Precision/Recall at 5% over three seeds:

| Model | Level | P@5% | R@5% | P@10% | R@10% | Backend |
|---|---|---:|---:|---:|---:|---|
| f-AnoGAN | L1 | 0.179 | 0.884 | 0.101 | 1.000 | deep |
| GRU-DeepSVDD | L1 | 0.179 | 0.884 | 0.099 | 0.975 | deep |
| GA-EIF+SVM | L1 | 0.007 | 0.035 | 0.004 | 0.037 | sklearn IF |
| f-AnoGAN | L2 | 0.263 | 1.000 | 0.132 | 1.000 | deep |
| GA-EIF+SVM | L2 | 0.263 | 1.000 | 0.132 | 1.000 | EIF |
| GRU-DeepSVDD | L2 | 0.246 | 0.932 | 0.132 | 1.000 | deep |

On this proxy task, L2 improves Precision at 5% by roughly eight percentage points for f-AnoGAN (0.179 to 0.263) and moves the forest pipeline from near failure on L1 to parity with f-AnoGAN on L2.

### 6.2 Agreement of top-5% flags

Jaccard overlap is higher on L2 than on L1:

| Pair | L1 Jaccard | L2 Jaccard |
|---|---:|---:|
| f-AnoGAN vs GRU-DeepSVDD | 0.425 | 0.446 |
| f-AnoGAN vs GA-EIF+SVM | 0.130 | 0.452 |
| GRU-DeepSVDD vs GA-EIF+SVM | 0.130 | 0.352 |

The forest detector joins the deep models on L2, while on L1 it barely overlaps them. That pattern matches the Precision/Recall gap.

### 6.3 Runtime

Mean train times on the strong run (CPU) were about 135 s for L1 f-AnoGAN, 27 s for L1 GRU-DeepSVDD, 21 s for L2 f-AnoGAN, and 8 s for L2 EIF. Inference was under two seconds per deep model on the reported test sizes.

---

## 7. Discussion

The results support a cautious claim: under this protocol, order-book features help unsupervised detectors concentrate high scores on extreme multivariate or price-amplitude windows. The gain is visible both in Precision/Recall and in cross-model agreement.

Two readings are compatible with the microstructure literature. First, L2 features carry imbalance and depth information that is only weakly present in OHLCV bars (Gould et al., 2013; Zhang et al., 2019). Second, L1 forests may fail when the feature space is small and stress is mostly a temporal pattern better captured by windowed deep models; on L2 the same isolation idea becomes useful again.

Proxy recall of 1.0 should not be read as perfect detection of fraud. The proxy definition already selects extreme windows, and a 5% score budget can cover that thin set when rankings are good.

---

## 8. Limitations

1. Proxy stress labels are not exchange-confirmed manipulation events (Kamps and Kleinberg, 2018; Xu and Livshits, 2019).
2. L1 is 1-minute data; L2 is 5-minute data. Direct timestamp alignment is not claimed.
3. Level-1 training uses 30,000 BTC rows rather than the full 87,821 for deep-model wall time.
4. Large-n Level-1 forests use scikit-learn Isolation Forest rather than pure Extended Isolation Forest.
5. The calendar window is concentrated in spring 2021.
6. No external supervised baseline with social-channel pump labels is included in this version.

---

## 9. Conclusion

This paper compared Level-1 and Level-2 inputs for unsupervised cryptocurrency anomaly detection across three model families under a reproducible Bitcoin-to-Ethereum protocol. Order-book features improved proxy Precision at 5% and raised agreement among detectors. The accompanying repository provides code and tables. Future work should add labeled event studies, additional assets and periods, and classical baselines such as LOF and standard Isolation Forest on identical windows.

---

## Data and code availability

Source code and result tables:  
https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection

Reproduce the strong evaluation with:

```text
python -m paper_eval.run_comparison --mode strong
```

---

## Acknowledgments

Parts of the model packages began as coursework in the Artificial Intelligence course at the University of Eastern Finland (2025).

---

## References

Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.

Audibert, J., Michiardi, P., Guyard, F., Marti, S., and Zuluaga, M. A. (2020). USAD: Unsupervised anomaly detection on multivariate time series. In *Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 3395-3404.

Bolz, M., Brundler, K., Kane, L., Patsias, P., Tessendorf, L., Gogol, K. M., Kim, T., and Tessone, C. J. (2025). Real-time machine learning detection of Telegram-based pump-and-dump schemes. In *Proceedings of the ACM CCS Workshop on Decentralized Finance and Security (DeFi'25)*. Also arXiv:2412.18848.

Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. (2000). LOF: Identifying density-based local outliers. *ACM SIGMOD Record*, 29(2), 93-104.

Chandola, V., Banerjee, A., and Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv:1406.1078.

Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223-236.

Cont, R., and de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM Journal on Financial Mathematics*, 4(1), 1-25.

Davis, J., and Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning*, pages 233-240.

Eiben, A. E., and Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.). Springer.

Gandal, N., Hamrick, J. T., Moore, T., and Oberman, T. (2018). Price manipulation in the Bitcoin ecosystem. *Journal of Monetary Economics*, 95, 86-96.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial nets. In *Advances in Neural Information Processing Systems*, volume 27.

Gould, M. D., Porter, M. A., Williams, S., McDonald, M., Fenn, D. J., and Howison, S. D. (2013). Limit order books. *Quantitative Finance*, 13(11), 1709-1742.

Griffin, J. M., and Shams, A. (2020). Is Bitcoin really untethered? *The Journal of Finance*, 75(4), 1913-1964.

Hariri, S., Carrasco Kind, M., and Brunner, R. J. (2021). Extended Isolation Forest. *IEEE Transactions on Knowledge and Data Engineering*, 33(4), 1479-1489.

Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.

Hundman, K., Constantinou, V., Laporte, C., Colwell, I., and Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 387-395.

Kamps, J., and Kleinberg, B. (2018). To the moon: Defining and detecting cryptocurrency pump-and-dumps. *Crime Science*, 7(1), 1-18.

Kercheval, A. N., and Zhang, Y. (2015). Modelling high-frequency limit order book dynamics with support vector machines. *Quantitative Finance*, 15(8), 1315-1329.

La Morgia, M., Mei, A., Sassi, F., and Stefa, J. (2020). Pump and dumps in the Bitcoin era: Real time detection of cryptocurrency market manipulations. In *International Conference on Computer Communications and Networks (ICCCN)*.

La Morgia, M., Mei, A., Sassi, F., and Stefa, J. (2023). A survey on pump and dump detection in the cryptocurrency market using machine learning. *Future Internet*, 15(8), 267.

Letteri, I. (2025). A comparative analysis of statistical and machine learning models for outlier detection in Bitcoin limit order books. arXiv:2507.14960.

Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). Isolation Forest. In *Proceedings of the Eighth IEEE International Conference on Data Mining (ICDM)*, pages 413-422.

Malhotra, P., Vig, L., Shroff, G., and Agarwal, P. (2015). Long short term memory networks for anomaly detection in time series. In *ESANN*.

Pang, G., Shen, C., Cao, L., and Hengel, A. V. D. (2021). Deep learning for anomaly detection: A review. *ACM Computing Surveys*, 54(2), 1-38.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems*, volume 32.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Muller, E., and Kloft, M. (2018). Deep one-class classification. In *Proceedings of the 35th International Conference on Machine Learning*, pages 4393-4402.

Ruff, L., Kauffmann, J. R., Vandermeulen, R. A., Montavon, G., Samek, W., Kloft, M., Dietterich, T. G., and Muller, K.-R. (2021). A unifying review of deep and shallow anomaly detection. *Proceedings of the IEEE*, 109(5), 756-795.

Saito, T., and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.

Schlegl, T., Seebock, P., Waldstein, S. M., Schmidt-Erfurth, U., and Langs, G. (2017). Unsupervised anomaly detection with generative adversarial networks to guide marker discovery. In *Information Processing in Medical Imaging (IPMI)*, pages 146-157. Springer.

Schlegl, T., Seebock, P., Waldstein, S. M., Langs, G., and Schmidt-Erfurth, U. (2019). f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks. *Medical Image Analysis*, 54, 30-44.

Scholkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., and Williamson, R. C. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7), 1443-1471.

Sirignano, J., and Cont, R. (2019). Universal features of price formation in financial markets: Perspectives from deep learning. *Quantitative Finance*, 19(9), 1449-1459.

Tax, D. M. J., and Duin, R. P. W. (2004). Support vector data description. *Machine Learning*, 54(1), 45-66.

Tsantekidis, A., Passalis, N., Tefas, A., Kanniainen, J., Gabbouj, M., and Iosifidis, A. (2017). Forecasting stock prices from the limit order book using convolutional neural networks. In *IEEE Conference on Business Informatics*, pages 7-12.

Xu, J., and Livshits, B. (2019). The anatomy of a cryptocurrency pump-and-dump scheme. In *28th USENIX Security Symposium*, pages 1609-1625.

Zhang, Z., Zohren, S., and Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing*, 67(11), 3001-3012.

Zhou, C., and Paffenroth, R. C. (2017). Anomaly detection with robust deep autoencoders. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 665-674.

Zong, B., Song, Q., Min, M. R., Cheng, W., Lumezanu, C., Cho, D., and Chen, H. (2018). Deep autoencoding Gaussian mixture model for unsupervised anomaly detection. In *International Conference on Learning Representations*.
