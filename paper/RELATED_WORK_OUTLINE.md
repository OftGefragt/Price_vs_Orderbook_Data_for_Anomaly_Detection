# Related Work outline (maps to `references.bib`)

Use this section plan when editing the manuscript. Each bullet is a claim you can support with the listed keys.

## 2.1 Classical and shallow anomaly detection
- Problem setting and evaluation difficulties: `chandola2009anomaly`, `aggarwal2017outlier`
- Isolation-based detectors: `liu2008isolation`, `hariri2021extended`
- Density and one-class baselines: `breunig2000lof`, `scholkopf2001estimating`, `tax2004support`

## 2.2 Deep anomaly detection
- Surveys: `ruff2021deep`, `pang2021deep`
- Reconstruction / AE family: `zhou2017anomaly`, `zong2018deep`
- GAN anomaly scoring: `schlegl2017anogan`, `schlegl2019fanogan`, `goodfellow2014gan`
- Hypersphere objectives: `ruff2018deep`
- Sequence models: `cho2014learning`, `malhotra2015long`, `hundman2018detecting`, `audibert2020usad`

## 2.3 Limit order books and predictive microstructure
- LOB structure: `gould2013limit`, `cont2014price`, `cont2001empirical`
- Learning from LOB features: `kercheval2015modelling`, `tsantekidis2017forecasting`, `zhang2019deeplob`, `sirignano2019universal`

## 2.4 Cryptocurrency manipulation and surveillance
- Manipulation evidence: `gandal2018price`, `griffin2020bitcoin`
- Pump-and-dump detection: `kamps2018to`, `xu2019anatomy`, `lamorgia2020pump`, `la2023survey`
- Order-book aware early warning: `arxiv2412pd`, `arxiv2507lob` (verify author metadata before camera-ready)

## 2.5 Hyperparameter search
- GA background: `holland1992adaptation`, `whitley1994genetic`, `eiben2015introduction`
- Alternative search: `bergstra2012random`

## 2.6 Evaluation under imbalance
- Precision-recall vs ROC: `davis2006relationship`, `saito2015precision`, `hand2009measuring`

## Gap statement (end of Related Work)
Most crypto surveillance papers either (i) rely on Level-1 OHLCV with event labels from social channels, or (ii) use order-book features inside supervised or heuristic pipelines. Fewer studies ask, under a matched unsupervised protocol, whether Level-2 features improve ranking of extreme market stress relative to Level-1 inputs across several detector families. That comparison is the focus of this paper.
