"""Task 8: L2 feature-group ablation with IsolationForest (seed 42)."""

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

from paper_eval.data import load_level2, sliding_windows
from paper_eval.v2.common_labels import load_l2_with_midpoint, price_panel_from_series
from paper_eval.v2.labels import ranking_metrics, top_fraction_labels, window_amplitudes


OUT = REPO_ROOT / "results_v2" / "task8_feature_groups"
GROUPS = json.loads((REPO_ROOT / "paper_eval" / "v2" / "feature_groups_l2.json").read_text())


def eval_cols(cols, x_tr_df, x_te_df, price, name, seed=42):
    sc = MinMaxScaler()
    tr = sc.fit_transform(x_tr_df[cols].values)
    te = sc.transform(x_te_df[cols].values)
    clf = IsolationForest(
        n_estimators=235,
        max_samples=min(1691, len(tr)),
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(tr)
    scores = sliding_windows((-clf.score_samples(te)).reshape(-1, 1), 30, 1)[:, :, 0].max(1)
    lab, meta = top_fraction_labels(
        window_amplitudes(price_panel_from_series(price), "close_range"), 0.01
    )
    m = ranking_metrics(scores[: len(lab)], lab, 0.05)
    return {
        "ablation": name,
        "n_features": len(cols),
        "features": ",".join(cols),
        "base_rate": meta["base_rate"],
        "p5_precision": m["precision"],
        "p5_recall": m["recall"],
        "p5_lift": m["lift"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = load_level2()
    mid = load_l2_with_midpoint()[1].to_numpy()
    all_cols = list(tr.columns)
    rows = [eval_cols(all_cols, tr, te, mid, "full_30")]
    groups = GROUPS["groups"]
    for gname, cols in groups.items():
        cols = [c for c in cols if c in tr.columns]
        rows.append(eval_cols(cols, tr, te, mid, f"alone_{gname}"))
    for gname, cols in groups.items():
        leave = [c for c in all_cols if c not in cols]
        rows.append(eval_cols(leave, tr, te, mid, f"leave_out_{gname}"))
    # level-ish: distances with index <=5 vs higher if present
    low = [c for c in all_cols if any(c.endswith(f"_{i}") for i in range(0, 5))]
    high = [c for c in all_cols if c not in low]
    if low:
        rows.append(eval_cols(low, tr, te, mid, "levels_index_0_to_4"))
    if high:
        rows.append(eval_cols(high, tr, te, mid, "levels_index_ge_5_or_other"))
    rows.append(eval_cols(GROUPS["reduced_l2_features"], tr, te, mid, "reduced5_preregistered"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "group_ablation_seed42.csv", index=False)
    (OUT / "config.json").write_text(
        json.dumps({"task": 8, "seed": 42, "groups": GROUPS}, indent=2), encoding="utf-8"
    )
    (OUT / "SUMMARY.md").write_text(
        "# Task 8 — Feature groups\n\n" + df.drop(columns=["features"]).to_string(index=False) + "\n",
        encoding="utf-8",
    )
    print(df.drop(columns=["features"]).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
