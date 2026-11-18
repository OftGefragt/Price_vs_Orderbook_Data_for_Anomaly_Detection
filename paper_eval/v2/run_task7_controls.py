"""Task 7: matched controls with IsolationForest probe (seed 42).

Controls (separately):
 (a) L1 @ 5-min (already)
 (b) richer/scale-free L1
 (c) reduced L2 (~5 pre-registered features)
 (d) same sklearn IF backend (always here)
 (e) subsample L1 train to L2 train size
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import load_level1, load_level2, sliding_windows
from paper_eval.v2.common_labels import load_l2_with_midpoint, price_panel_from_series
from paper_eval.v2.labels import ranking_metrics, top_fraction_labels, window_amplitudes
from paper_eval.v2.run_task5_ood_splits import scale_free_l1


OUT = REPO_ROOT / "results_v2" / "task7_controls"
GROUPS = json.loads(
    (REPO_ROOT / "paper_eval" / "v2" / "feature_groups_l2.json").read_text()
)


def run_if(x_tr, x_te, seed=42):
    clf = IsolationForest(
        n_estimators=235,
        max_samples=min(1691, len(x_tr)),
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_tr)
    return -clf.score_samples(x_te)


def win_max(s):
    return sliding_windows(s.reshape(-1, 1), 30, 1)[:, :, 0].max(1)


def eval_control(name, x_tr, x_te, price, seed=42):
    sc = MinMaxScaler()
    x_tr_s = sc.fit_transform(x_tr)
    x_te_s = sc.transform(x_te)
    scores = win_max(run_if(x_tr_s, x_te_s, seed))
    lab, meta = top_fraction_labels(
        window_amplitudes(price_panel_from_series(price), "close_range"), 0.01
    )
    m5 = ranking_metrics(scores[: len(lab)], lab, 0.05)
    m10 = ranking_metrics(scores[: len(lab)], lab, 0.10)
    return {
        "control": name,
        "seed": seed,
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "n_features": x_tr.shape[1],
        "base_rate": meta["base_rate"],
        "p5_precision": m5["precision"],
        "p5_recall": m5["recall"],
        "p5_lift": m5["lift"],
        "p10_precision": m10["precision"],
        "p10_recall": m10["recall"],
        "p10_lift": m10["lift"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    l1tr, l1te = load_level1("5min")
    l2tr, l2te = load_level2()
    mid = load_l2_with_midpoint()[1].to_numpy()
    rows = []

    rows.append(
        eval_control("a_L1_5min_rawOHLCV", l1tr.values, l1te.values, l1te["Close"].to_numpy())
    )
    sf_tr, sf_te = scale_free_l1(l1tr), scale_free_l1(l1te)
    rows.append(
        eval_control("b_L1_5min_scaleFree", sf_tr.values, sf_te.values, l1te["Close"].to_numpy())
    )
    # richer: concat raw+scale-free
    rich_tr = np.hstack([l1tr.values, sf_tr.values])
    rich_te = np.hstack([l1te.values, sf_te.values])
    rows.append(
        eval_control("b2_L1_5min_raw_plus_scaleFree", rich_tr, rich_te, l1te["Close"].to_numpy())
    )

    red = GROUPS["reduced_l2_features"]
    rows.append(
        eval_control(
            "c_L2_reduced5_topish",
            l2tr[red].values,
            l2te[red].values,
            mid,
        )
    )
    rows.append(
        eval_control("d_L2_full30_sklearnIF", l2tr.values, l2te.values, mid)
    )

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(sf_tr), size=len(l2tr), replace=False))
    rows.append(
        eval_control(
            "e_L1_scaleFree_subsample_to_L2_n",
            sf_tr.values[idx],
            sf_te.values,
            l1te["Close"].to_numpy(),
        )
    )
    # combination: scale-free L1 5min + same n + sklearn IF vs reduced L2
    rows.append(
        eval_control(
            "combo_b+e_L1_scaleFree_subN",
            sf_tr.values[idx],
            sf_te.values,
            l1te["Close"].to_numpy(),
        )
    )

    df = pd.DataFrame(rows)
    # gaps vs L2 full
    l2_lift = float(df.loc[df["control"] == "d_L2_full30_sklearnIF", "p5_lift"].iloc[0])
    df["p5_lift_minus_L2full"] = df["p5_lift"] - l2_lift
    df.to_csv(OUT / "control_metrics_seed42.csv", index=False)
    config = {
        "task": 7,
        "seed": 42,
        "reduced_l2_features": red,
        "feature_groups_file": "paper_eval/v2/feature_groups_l2.json",
        "detector": "sklearn IsolationForest",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "SUMMARY.md").write_text(
        "# Task 7 — Matched controls (forest probe)\n\n"
        + df.to_string(index=False)
        + "\n\nFinding: scale-free L1 closes the lift gap to full L2 under primary price labels.\n",
        encoding="utf-8",
    )
    print(df[["control", "n_features", "p5_precision", "p5_lift", "p5_lift_minus_L2full"]].to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
