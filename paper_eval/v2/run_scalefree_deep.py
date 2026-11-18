"""Train AnoGAN + GRU on scale-free L1 (and L2 baseline) under primary price labels.

Intended for Colab GPU (or long local CPU). Writes results_v2/scalefree_deep/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import load_level1, load_level2
from paper_eval.models_anogan import FanoganDetector
from paper_eval.models_gru_svdd import GRUDeepSVDDDetector
from paper_eval.v2.common_labels import load_l2_with_midpoint, price_panel_from_series
from paper_eval.v2.labels import ranking_metrics, top_fraction_labels, window_amplitudes
from paper_eval.v2.run_task5_ood_splits import scale_free_l1


OUT = REPO_ROOT / "results_v2" / "scalefree_deep"


def run_one(model_name, detector, x_tr, x_te, labels, seed, level, track):
    detector.fit(x_tr)
    scores, infer_s = detector.score(x_te)
    # scores already window-level for deep models
    n = min(len(scores), len(labels))
    rows = []
    for frac in (0.05, 0.10):
        m = ranking_metrics(scores[:n], labels[:n], frac)
        rows.append(
            {
                "track": track,
                "level": level,
                "model": model_name,
                "seed": seed,
                "train_seconds": getattr(detector, "train_seconds", float("nan")),
                "infer_seconds": infer_s,
                **m,
            }
        )
    score_path = OUT / f"scores_{track}_{model_name.replace('+','_').replace('-','_').lower()}_seed{seed}.csv"
    pd.DataFrame({"score": scores[:n]}).to_csv(score_path, index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--seeds", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "fast":
        seeds = [42]
        epochs_gan, epochs_enc, epochs_gru = 3, 3, 3
    else:
        seeds = [42, 43, 44]
        epochs_gan, epochs_enc, epochs_gru = 10, 15, 8
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # Scale-free L1
    l1tr, l1te = load_level1("5min")
    sf_tr, sf_te = scale_free_l1(l1tr), scale_free_l1(l1te)
    sc = MinMaxScaler()
    x1tr = sc.fit_transform(sf_tr.values)
    x1te = sc.transform(sf_te.values)
    lab1, meta1 = top_fraction_labels(
        window_amplitudes(price_panel_from_series(l1te["Close"].to_numpy()), "close_range"),
        0.01,
    )

    # L2 for comparison (same seeds/epochs)
    l2tr, l2te = load_level2()
    sc2 = MinMaxScaler()
    x2tr = sc2.fit_transform(l2tr.values)
    x2te = sc2.transform(l2te.values)
    mid = load_l2_with_midpoint()[1].to_numpy()
    lab2, meta2 = top_fraction_labels(
        window_amplitudes(price_panel_from_series(mid), "close_range"), 0.01
    )

    for seed in seeds:
        print(f"==== seed {seed} ====")
        for track, level, xtr, xte, lab in [
            ("L1_scaleFree", "L1", x1tr, x1te, lab1),
            ("L2", "L2", x2tr, x2te, lab2),
        ]:
            jobs = [
                (
                    "f-AnoGAN",
                    FanoganDetector(
                        epochs_gan=epochs_gan, epochs_enc=epochs_enc, seed=seed
                    ),
                ),
                (
                    "GRU-DeepSVDD",
                    GRUDeepSVDDDetector(epochs=epochs_gru, seed=seed),
                ),
            ]
            for name, det in jobs:
                print(f"Training {name} on {track} ...")
                all_rows.extend(
                    run_one(name, det, xtr, xte, lab, seed, level, track)
                )

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "scalefree_deep_metrics.csv", index=False)
    agg = (
        df.groupby(["track", "model", "flag_frac"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            lift_mean=("lift", "mean"),
            base_rate_mean=("base_rate", "mean"),
        )
    )
    agg.to_csv(OUT / "scalefree_deep_metrics_agg.csv", index=False)
    config = {
        "mode": args.mode,
        "seeds": seeds,
        "epochs_gan": epochs_gan,
        "epochs_enc": epochs_enc,
        "epochs_gru": epochs_gru,
        "label": "primary_price_range exact top 1%",
        "l1_features": "scale_free",
        "l1_label_meta": meta1,
        "l2_label_meta": meta2,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "SUMMARY.md").write_text(
        "# Scale-free deep comparison\n\n" + agg.to_string(index=False) + "\n",
        encoding="utf-8",
    )
    print(agg.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
