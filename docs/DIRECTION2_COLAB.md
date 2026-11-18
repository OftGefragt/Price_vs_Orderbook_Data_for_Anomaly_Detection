# Direction 2 — finish methodology, then rewrite the paper

## Label prevalence / circular labels (plain English)

**Label prevalence** = what fraction of windows we call “stress.”  
Old code used “above the 99th percentile,” which **kept ties**, so L2 had ~1.32% positives and L1 ~1.0%. That made L2 Precision@5% look better (~0.26) partly because there were more positives to hit. We now force **exact top 1%** on both tracks.

**Circular labels** = the L2 “stress” label was built from the **same 30 order-book features** the model sees (feature-norm range). Then “high scores on extreme feature moves” is partly tautological.  
**Primary label now** uses **price only** (L1 Close range; L2 mid-price range), which the L2 model does not receive as an input feature.

**Scale-free L1** = replace absolute OHLCV prices with relative features (log returns, high-low/close, volume z-score, etc.) so BTC→ETH is not dominated by dollar level differences.

---

## What needs GPU vs what can run here

| Work | Local CPU (your Mac / this machine) | Colab GPU |
|---|---|---|
| Tasks 1–10 forest / classical / economics | **Yes** (minutes–tens of minutes) | Optional |
| Scale-free **deep** AnoGAN + GRU (seeds 42–44) | Possible but **slow** (hours) | **Recommended** |
| Deep seeds 45–51 | Not practical on CPU | GPU if you want full Task 6 |

You do **not** need Colab for classical controls. You **should** use Colab GPU for the scale-free deep comparison that finishes Direction 2.

---

## Local CPU command

```bash
cd Price_vs_Orderbook_Data_for_Anomaly_Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-paper.txt
python -m paper_eval.v2.run_direction2_cpu
```

Outputs land in `results_v2/task1_*` … `task10_*`.

## Colab GPU (deep scale-free)

Open `notebooks/Methodology_v2_Direction2_Colab.ipynb`:

1. Runtime → GPU  
2. Run install/clone cells  
3. Run CPU suite (optional if already on GitHub)  
4. Run:

```bash
python -m paper_eval.v2.run_scalefree_deep --mode full
```

Then download `results_v2/scalefree_deep/`.

Fast smoke:

```bash
python -m paper_eval.v2.run_scalefree_deep --mode fast
```

---

## After runs finish

1. Update `results_v2/REPORT.md` with scale-free deep numbers  
2. Rewrite `paper/RESEARCHGATE_MANUSCRIPT.md` to the cautionary Direction-2 framing  
3. Resolve Chris authorship  
4. Upload preprint
