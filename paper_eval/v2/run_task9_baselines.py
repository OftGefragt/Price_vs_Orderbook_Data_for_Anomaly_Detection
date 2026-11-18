"""Task 9: IF / LOF / One-Class SVM baselines on identical windows + primary labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import load_level1, load_level2, sliding_windows
from paper_eval.metrics import jaccard
from paper_eval.v2.common_labels import load_l2_with_midpoint, price_panel_from_series
from paper_eval.v2.labels import ranking_metrics, top_fraction_labels, window_amplitudes
from paper_eval.v2.run_task5_ood_splits import scale_free_l1


OUT = REPO_ROOT / "results_v2" / "task9_baselines"


def row_to_window_scores(row_scores: np.ndarray, window: int = 30) -> np.ndarray:
    return sliding_windows(row_scores.reshape(-1, 1), window, 1)[:, :, 0].max(axis=1)


def top_flags(scores: np.ndarray, frac: float = 0.05) -> np.ndarray:
    n = len(scores)
    k = max(1, int(round(n * frac)))
    order = np.lexsort((np.arange(n), -scores))[:k]
    f = np.zeros(n, dtype=bool)
    f[order] = True
    return f


def fit_score_baselines(x_tr: np.ndarray, x_te: np.ndarray, seed: int = 42) -> dict:
    n = len(x_tr)
    max_samples = min(1691, n)
    out = {}

    iforest = IsolationForest(
        n_estimators=235,
        max_samples=max_samples,
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(x_tr)
    out["IsolationForest"] = -iforest.score_samples(x_te)

    # LOF novelty mode
    n_neighbors = min(20, max(5, n // 50))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        contamination=0.062,
        n_jobs=-1,
    )
    lof.fit(x_tr)
    out["LOF"] = -lof.score_samples(x_te)

    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.062)
    oc.fit(x_tr)
    out["OneClassSVM"] = -oc.decision_function(x_te)
    return out


def eval_track(level: str, x_tr, x_te, price, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    sc = MinMaxScaler()
    x_tr_s = sc.fit_transform(x_tr)
    x_te_s = sc.transform(x_te)
    amps = window_amplitudes(price_panel_from_series(price), "close_range")
    lab, meta = top_fraction_labels(amps, 0.01)
    scores_map = fit_score_baselines(x_tr_s, x_te_s, seed=seed)
    rows = []
    flags = {}
    for name, row_scores in scores_map.items():
        win_scores = row_to_window_scores(row_scores)
        flags[name] = top_flags(win_scores[: len(lab)], 0.05)
        for frac in (0.05, 0.10):
            m = ranking_metrics(win_scores[: len(lab)], lab, frac)
            rows.append(
                {
                    "level": level,
                    "model": name,
                    "seed": seed,
                    "label_kind": "primary_price_range",
                    **m,
                }
            )
    return pd.DataFrame(rows), flags


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed = 42

    l1tr, l1te = load_level1("5min")
    sf_tr, sf_te = scale_free_l1(l1tr), scale_free_l1(l1te)
    l2tr, l2te = load_level2()
    mid = load_l2_with_midpoint()[1].to_numpy()

    m1, f1 = eval_track("L1_scaleFree", sf_tr.values, sf_te.values, l1te["Close"].to_numpy(), seed)
    m1b, f1b = eval_track("L1_rawOHLCV", l1tr.values, l1te.values, l1te["Close"].to_numpy(), seed)
    m2, f2 = eval_track("L2", l2tr.values, l2te.values, mid, seed)

    metrics = pd.concat([m1, m1b, m2], ignore_index=True)
    metrics.to_csv(OUT / "baseline_metrics_seed42.csv", index=False)

    agree_rows = []
    for level, fmap in [
        ("L1_scaleFree", f1),
        ("L1_rawOHLCV", f1b),
        ("L2", f2),
    ]:
        names = list(fmap)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                agree_rows.append(
                    {
                        "level": level,
                        "model_a": a,
                        "model_b": b,
                        "jaccard_top5": jaccard(fmap[a], fmap[b]),
                        "chance_jaccard": 0.05 / (2 - 0.05),
                    }
                )
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(OUT / "baseline_agreement_top5_seed42.csv", index=False)

    # Subperiod split on L1 scale-free calendar
    cut = l1te.index[len(l1te) // 2]
    rows_sub = []
    for tag, mask_tr, te_df, te_x in [
        (
            "L1_scaleFree_ETH_first_half",
            slice(None),
            l1te.loc[:cut],
            sf_te.loc[:cut].values,
        ),
        (
            "L1_scaleFree_ETH_second_half",
            slice(None),
            l1te.loc[cut:],
            sf_te.loc[cut:].values,
        ),
    ]:
        sc = MinMaxScaler()
        x_tr = sc.fit_transform(sf_tr.values)
        x_te = sc.transform(te_x)
        pack = fit_score_baselines(x_tr, x_te, seed)
        amps = window_amplitudes(
            price_panel_from_series(te_df["Close"].to_numpy()), "close_range"
        )
        lab, _ = top_fraction_labels(amps, 0.01)
        for name, rs in pack.items():
            ws = row_to_window_scores(rs)
            m = ranking_metrics(ws[: len(lab)], lab, 0.05)
            rows_sub.append({"subperiod": tag, "model": name, "seed": seed, **m})
    pd.DataFrame(rows_sub).to_csv(OUT / "subperiod_L1_scaleFree_seed42.csv", index=False)

    config = {
        "task": 9,
        "seed": 42,
        "baselines": ["IsolationForest", "LOF", "OneClassSVM"],
        "label": "primary_price_range exact top 1%",
        "tracks": ["L1_rawOHLCV", "L1_scaleFree", "L2"],
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = [
        "# Task 9 — Classical baselines",
        "",
        metrics[metrics["flag_frac"] == 0.05][
            ["level", "model", "precision", "recall", "lift", "base_rate"]
        ].to_string(index=False),
        "",
        agree.to_string(index=False),
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(metrics[metrics["flag_frac"] == 0.05][["level", "model", "precision", "lift"]].to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
