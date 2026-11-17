# Level-1 Price versus Level-2 Order-Book Features for Unsupervised Anomaly Detection in Cryptocurrency Markets

**Melike Kaya**  
Independent researcher / University of Eastern Finland (coursework extension)  
Corresponding repository: https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection  

**Status.** Technical report prepared for arXiv (`cs.LG`; secondary `q-fin.ST`).  
**Code and tables.** Strong-protocol results are in `results/` (`protocol.json`, `table5_main_comparison_agg.csv`).

---

## Abstract

Unsupervised anomaly detectors are often trained on cryptocurrency price series, yet practitioners disagree on whether richer Level-2 (L2) order-book features improve detection over Level-1 (L1) OHLCV inputs. We compare three detector families under a matched protocol: a one-dimensional f-AnoGAN-style model, an Extended Isolation Forest configuration selected by a genetic search and paired with a one-class SVM stage, and a GRU encoder trained with a Deep SVDD objective. Models are trained on Bitcoin and evaluated on Ethereum. Ranking quality is measured with Precision@k and Recall@k against weak proxy stress labels (extreme amplitude windows), together with cross-model agreement of top-quantile flags. Across three seeds, L2 inputs raise Precision@5% from about 0.18 on L1 to about 0.26 on L2 for f-AnoGAN and for the forest pipeline, with Recall@5% reaching 1.0 on L2 for those two families. Pairwise Jaccard agreement among top-5% flags is also higher on L2 than on L1. We release code, seeds, and protocol files. The study does not claim verified manipulation labels; proxy stress labels and sampling-rate differences remain important limitations.

**Keywords:** anomaly detection, cryptocurrency, limit order book, AnoGAN, isolation forest, Deep SVDD

---

## 1. Introduction

Cryptocurrency markets are liquid, continuous, and frequently exposed to abrupt stress and alleged manipulation [gandal2018price, griffin2020bitcoin, kamps2018to]. Surveillance systems therefore need methods that flag unusual intervals without waiting for costly event labels [chandola2009anomaly, la2023survey].

A practical design choice is the input representation. Level-1 data summarize traded prices and volumes. Level-2 order-book data expose depth, imbalance, and cancel or limit flow [gould2013limit, zhang2019deeplob]. Order-book features help in predictive microstructure tasks [kercheval2015modelling, tsantekidis2017forecasting, sirignano2019universal], and recent pump-and-dump detectors sometimes include book pressure statistics [arxiv2412pd]. It is less clear, under a single unsupervised protocol, whether L2 features improve anomaly ranking relative to L1 features across several modern detector families.

### Contributions

1. A matched BTC-train / ETH-test comparison of L1 versus L2 inputs for three unsupervised families: f-AnoGAN-style scoring [schlegl2017anogan, schlegl2019fanogan], GA-configured Extended Isolation Forest with a one-class SVM stage [hariri2021extended, liu2008isolation], and GRU + Deep SVDD [cho2014learning, ruff2018deep].
2. A shared top-quantile flagging rule plus Precision/Recall against weak proxy stress labels, with multi-seed aggregation.
3. Public code, protocol metadata, and tables that reproduce the reported numbers.

---

## 2. Related Work

### 2.1 Unsupervised anomaly detection

Classical surveys emphasize that labels are scarce and that evaluation is fragile under class imbalance [chandola2009anomaly, aggarwal2017outlier]. Isolation Forest isolates rare points with random partitions [liu2008isolation]. Extended Isolation Forest replaces axis-aligned cuts with random hyperplanes and reduces score artifacts [hariri2021extended]. Density and support estimators such as LOF and one-class SVM remain common baselines [breunig2000lof, scholkopf2001estimating, tax2004support].

Deep methods are reviewed by Ruff et al. and Pang et al. [ruff2021deep, pang2021deep]. Autoencoder and mixture hybrids score reconstruction or density in latent space [zhou2017anomaly, zong2018deep]. AnoGAN introduced GAN-based anomaly scoring through reconstruction and discriminator features [schlegl2017anogan]; f-AnoGAN replaces slow latent search with a trained encoder [schlegl2019fanogan]. Deep SVDD learns a hypersphere of normal embeddings [ruff2018deep]. Sequence models extend these ideas to time series [malhotra2015long, hundman2018detecting, audibert2020usad].

### 2.2 Order books and crypto surveillance

Limit order books encode supply and demand beyond the last trade [gould2013limit, cont2014price]. Machine learning models use book snapshots for mid-price and return prediction [kercheval2015modelling, tsantekidis2017forecasting, zhang2019deeplob]. Cryptocurrency research documents manipulation and pump-and-dump organization [gandal2018price, xu2019anatomy, lamorgia2020pump, la2023survey]. Many detectors still start from OHLCV or social signals [kamps2018to]. Work that mixes book features into early-warning scores suggests microstructure cues matter [arxiv2412pd, arxiv2507lob], but few studies isolate the L1 versus L2 representation effect inside the same unsupervised training recipe.

### 2.3 Gap

We focus on that missing comparison: fixed detectors, fixed train/test assets, two input levels, and ranking metrics that do not require confirmed fraud labels.

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

Feature names and grouping appear in the repository documentation. L1 and L2 do not share identical timestamps or sampling rates. We treat them as two experimental tracks rather than as aligned panels.

### 3.3 Preprocessing

A MinMax scaler is fit on the training split only. Sequence models use sliding windows of length 30 with step 1.

---

## 4. Methods

### 4.1 f-AnoGAN-style 1D detector

Following the f-AnoGAN idea of an encoder plus generator and discriminator [schlegl2019fanogan], we train a compact 1D convolutional stack on normal windows. The anomaly score is

\[
s = 1.0\cdot \ell_{\mathrm{rec}} + 0.1\cdot \ell_{\mathrm{feat}},
\]

where \(\ell_{\mathrm{rec}}\) is reconstruction error and \(\ell_{\mathrm{feat}}\) is a discriminator feature residual. Strong-protocol epochs: 10 (GAN) + 15 (encoder).

### 4.2 GA-configured forest with one-class SVM

Hyperparameters follow a genetic search recorded in the project notebook: 235 trees, subsample size 1691, contamination 0.062, extension level 26 when dimensionality allows (level 4 on L1). The search lineage follows standard evolutionary model selection practice [holland1992adaptation, eiben2015introduction]. Continuous forest scores are used for ranking. On Level-2 we use the repository Extended Isolation Forest implementation [hariri2021extended]. On large Level-1 matrices we use scikit-learn Isolation Forest with matched tree and subsample settings for runtime [liu2008isolation, pedregosa2011scikit]. The backend name is logged.

### 4.3 GRU encoder with Deep SVDD loss

A two-layer GRU [cho2014learning] maps each window to an embedding. Training minimizes distance to a hypersphere center updated toward the batch mean, in the spirit of Deep SVDD [ruff2018deep]. Strong-protocol epochs: 8.

---

## 5. Experimental Setup

| Item | Strong protocol |
|---|---|
| Seeds | 42, 43, 44 |
| Train asset | BTC |
| Test asset | ETH |
| Flag rule | top 5% and top 10% scores |
| Proxy labels | top 1% amplitude windows (Close range on L1; feature-norm range on L2) |
| Implementation | PyTorch and scikit-learn [paszke2019pytorch, pedregosa2011scikit] |

Precision@k and Recall@k are preferred over ROC summaries under imbalance [davis2006relationship, saito2015precision]. Exact counts are stored in `results/protocol.json`.

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

On this proxy task, L2 improves Precision@5% by roughly eight percentage points for f-AnoGAN (0.179 to 0.263) and moves the forest pipeline from near failure on L1 to parity with f-AnoGAN on L2. Seed-wise standard deviations for the leading L2 entries are zero or near zero under the fixed quantile rule and stable scores.

### 6.2 Agreement of top-5% flags (last seed)

Jaccard overlap is higher on L2 than on L1:

| Pair | L1 Jaccard | L2 Jaccard |
|---|---:|---:|
| f-AnoGAN vs GRU-DeepSVDD | 0.425 | 0.446 |
| f-AnoGAN vs GA-EIF+SVM | 0.130 | 0.452 |
| GRU-DeepSVDD vs GA-EIF+SVM | 0.130 | 0.352 |

The forest detector joins the deep models on L2, while on L1 it barely overlaps them. That pattern matches the Precision/Recall gap.

### 6.3 Runtime

Mean train times on the strong run (CPU in our local execution) were about 135 s for L1 f-AnoGAN, 27 s for L1 GRU-DeepSVDD, 21 s for L2 f-AnoGAN, and 8 s for L2 EIF. Inference was under two seconds per deep model on the reported test sizes.

---

## 7. Discussion

The results support a cautious claim: under this protocol, order-book features help unsupervised detectors concentrate high scores on extreme multivariate or price-amplitude windows. The gain is visible both in Precision/Recall and in cross-model agreement.

Two mechanistic readings are compatible with the microstructure literature. First, L2 features carry imbalance and depth information that is only weakly present in OHLCV bars [gould2013limit, zhang2019deeplob]. Second, L1 forests may fail when the feature space is small and stress is mostly a temporal pattern better captured by windowed deep models; on L2 the same isolation idea becomes useful again.

We do not interpret proxy recall of 1.0 as perfect detection of fraud. The proxy definition already selects extreme windows, and a 5% score budget can cover that thin set when rankings are good.

---

## 8. Limitations

1. Proxy stress labels are not exchange-confirmed manipulation events [kamps2018to, xu2019anatomy].
2. L1 is 1-minute data; L2 is 5-minute data. Direct timestamp alignment is not claimed.
3. Level-1 training uses 30,000 BTC rows rather than the full 87,821 for deep-model wall time.
4. Large-n Level-1 forests use sklearn Isolation Forest rather than pure Extended Isolation Forest.
5. The calendar window is concentrated in spring 2021.
6. No external supervised baseline with social-channel pump labels is included in this version.

---

## 9. Conclusion

We compared Level-1 and Level-2 inputs for unsupervised cryptocurrency anomaly detection across three model families under a reproducible BTC-to-ETH protocol. Order-book features improved proxy Precision@5% and raised agreement among detectors. The accompanying repository provides code and tables for an arXiv technical report. Future work should add labeled event studies, additional assets and periods, and classical baselines such as LOF and standard Isolation Forest on identical windows.

---

## Data and code availability

https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection  

Run:

```bash
python -m paper_eval.run_comparison --mode strong
```

---

## Acknowledgments

Parts of the model packages began as coursework in the Artificial Intelligence course at the University of Eastern Finland (2025).

---

## References

See `paper/references.bib`. In the camera-ready PDF, compile with BibTeX/biblatex. For this Markdown draft, citation keys above match the `.bib` file. Before arXiv upload, replace the two arXiv-only entries marked “verify author list” with exact author metadata from their PDFs.
