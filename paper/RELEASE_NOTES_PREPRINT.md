# Preprint release notes (November 2026)

## Title

Cross-Asset Transfer for Crypto Anomaly Detection: Level-1 vs Level-2 Features under Deep and Classical Models

## Framing change (Direction 2)

Earlier exploratory matched tables suggested L2 Precision@5% ≈ 0.26 versus L1 ≈ 0.19. Controlled methodology (`results_v2/`) shows that gain does not survive:

1. Exact top-1% prevalence equalization
2. Non-circular price-range labels
3. Scale-free L1 for classical detectors
4. Scale-free deep GPU re-run (`results_v2/scalefree_deep/`)

**Publishable claim:** BTC→ETH unsupervised transfer works; L2 is optional once scale and labels are controlled.

## Files for deposit

- `paper/Kaya_CrossAsset_Anomaly_Detection.docx`
- `paper/RESEARCHGATE_MANUSCRIPT.md` / `.html`
- `results_v2/REPORT.md`
- `results_v2/scalefree_deep/`

## Link

https://github.com/OftGefragt/Price_vs_Orderbook_Data_for_Anomaly_Detection
