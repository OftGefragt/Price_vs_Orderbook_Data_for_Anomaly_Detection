# Figure captions (APA 7)

## Figure 1 — keep our original drawing

**Files:** `fig1_fanogan_1d_original.png` / `.pdf`

**Figure 1**  
*1D f-AnoGAN-style training and fast anomaly scoring used in this study.*  

*Note.* Latent noise is mapped by a 1D transposed-convolution generator; a 1D convolutional discriminator distinguishes real windows from fakes. At scoring time an encoder maps a real length-30 window to latent space, and the anomaly score combines reconstruction and discriminator feature losses. Original schematic for this manuscript; method conceptually follows Schlegl et al. (2019).

---

## Figure 4 — use Hariri’s published figure + APA reprint note

Do **not** use our redraw (`fig4_if_vs_eif_original.*`). Drop that pair from the paper.

### Which original to take

From Hariri et al. (2021) / arXiv:1811.02141, the clean IF-vs-EIF intuition panels are usually:

| What you want to show | Take this figure |
|-----------------------|------------------|
| Axis-aligned vs oblique **branch cuts** | Fig. 6 (IF) + Fig. 9 (EIF), or Fig. 7 + Fig. 10 |
| Score-map artifacts (most common in slides) | **Figure 12** — (a) Standard IF vs (c) Extended IF |

**Recommended:** Figure 12 from Hariri et al. (panels a and c). Export from the IEEE PDF or arXiv PDF you already have access to; paste into Word.

### APA 7 caption (student / coursework — citation required)

**Figure 4**  
*Isolation Forest versus Extended Isolation Forest anomaly score maps.*  

*Note.* From “Extended Isolation Forest,” by S. Hariri, M. Carrasco Kind, and R. J. Brunner, 2021, *IEEE Transactions on Knowledge and Data Engineering, 33*(4), p. XXX (https://doi.org/10.1109/TKDE.2019.2947676). Copyright 2021 by IEEE.  

Replace `p. XXX` with the page where Fig. 12 appears in the PDF you used (IEEE pagination or arXiv page).

### APA 7 caption (if you will publish / post publicly — permission required)

Same as above, then add at the end:

`Reprinted with permission.`

Get that permission in ~minutes via **IEEE RightsLink** on the Xplore page:  
https://ieeexplore.ieee.org/document/8888179 → copyright / RightsLink → request reuse in thesis / journal / preprint. Keep the approval email/PDF with the manuscript.

### Reference list (already in `REFERENCES_APA7.md`)

Hariri, S., Carrasco Kind, M., & Brunner, R. J. (2021). Extended isolation forest. *IEEE Transactions on Knowledge and Data Engineering, 33*(4), 1479–1489. https://doi.org/10.1109/TKDE.2019.2947676

### How APA image citation works (short)

1. **Number + italic title** above the figure.  
2. **Note.** under the figure with: From “Article title,” by Authors, Year, *Journal, vol*(issue), p. page (DOI). Copyright Year by Copyright holder.  
3. If permission was obtained: add **Reprinted with permission.**  
4. Full article stays in the **reference list**.  
5. Citing in text is enough for *ideas*; for a *copied image* you need the figure Note (and permission if publishing).

### Copyright reality check

| Situation | Cite APA Note? | IEEE RightsLink? |
|-----------|----------------|------------------|
| Course essay / internal draft | Yes | Usually not asked; still cite |
| Thesis uploaded / arXiv / journal | Yes | **Yes** — citation alone is not a license |

Course slides that copied Hariri without a copyright line are **not** a safer source — cite Hariri/IEEE, not the slide deck.

---

## Figure 5 — original pipeline (section 4.3)

**Files:** `fig5_btc_eth_pipeline.png` / `.pdf`

**Figure 5**  
*End-to-end BTC→ETH ranking pipeline.*  

*Note.* Level-1 OHLCV and Level-2 order-book features are loaded, MinMax-scaled on Bitcoin training windows only (optional scale-free Level-1 path), used to train f-AnoGAN, GRU–Deep SVDD, and GA-configured Isolation Forest / Extended Isolation Forest models, then scored on Ethereum test windows. Continuous scores are ranked into top-*k* flags and evaluated with Precision@*k*, Recall@*k*, and lift against exact top-1% price-range proxies. A secondary May–June 2022 Luna-period run repeats the transfer for Level-1 only (`--regime 2022_bear --levels L1`). Original schematic for this manuscript.
