"""Task 5: OOD diagnostics, scale-free L1, and alternate split designs."""

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

from paper_eval.data import load_level1, load_level2, aggregate_ohlcv
from paper_eval.v2.labels import ranking_metrics, top_fraction_labels, window_amplitudes
from paper_eval.v2.common_labels import price_panel_from_series


OUT = REPO_ROOT / "results_v2" / "task5_ood_splits"


def ood_share(train: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    rows = []
    for j in range(train.shape[1]):
        lo, hi = train[:, j].min(), train[:, j].max()
        x = test[:, j]
        outside = (x < lo) | (x > hi)
        rows.append(
            {
                "feature_index": j,
                "train_min": float(lo),
                "train_max": float(hi),
                "test_outside_share": float(outside.mean()),
                "test_below_share": float((x < lo).mean()),
                "test_above_share": float((x > hi).mean()),
            }
        )
    return pd.DataFrame(rows)


def scale_free_l1(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns, high-low/close, volume z-score within series."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    out["log_ret"] = np.log(close).diff().fillna(0.0)
    out["hl_range"] = ((df["High"] - df["Low"]) / close.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    vol = df["Volume"].astype(float)
    out["vol_z"] = (vol - vol.mean()) / (vol.std(ddof=0) + 1e-12)
    out["ret_abs"] = out["log_ret"].abs()
    out["oc_ret"] = np.log(df["Close"].astype(float) / df["Open"].astype(float).replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return out


def fit_if_and_score(x_tr, x_te, seed=42):
    n = len(x_tr)
    clf = IsolationForest(
        n_estimators=235,
        max_samples=min(1691, n),
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_tr)
    scores = -clf.score_samples(x_te)
    # tied scores
    # round lightly for float ties
    rounded = np.round(scores, 10)
    _, counts = np.unique(rounded, return_counts=True)
    n_tied_values = int(np.sum(counts[counts > 1]))
    frac_in_tied_groups = float(np.sum(counts[counts > 1]) / len(scores))
    return scores, {
        "n_tied_distinct_values": n_tied_values,
        "frac_rows_in_tied_score_groups": frac_in_tied_groups,
        "n_unique_scores": int(len(counts)),
    }


def windowize_row_scores(scores, window=30):
    from paper_eval.data import sliding_windows

    return sliding_windows(scores.reshape(-1, 1), window, 1)[:, :, 0].max(axis=1)


def eval_split(name, x_tr, x_te, price_te, seed=42):
    scores, tie_meta = fit_if_and_score(x_tr, x_te, seed=seed)
    pr = windowize_row_scores(scores)
    amps = window_amplitudes(price_panel_from_series(price_te), "close_range")
    lab, meta = top_fraction_labels(amps, 0.01)
    m = ranking_metrics(pr[: len(lab)], lab, 0.05)
    return {
        "setup": name,
        "seed": seed,
        "n_train": int(len(x_tr)),
        "n_test": int(len(x_te)),
        **tie_meta,
        **meta,
        **{f"p5_{k}": v for k, v in m.items()},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- OOD on raw (unscaled) then after BTC-fit minmax ---
    l1tr, l1te = load_level1("1min")
    l2tr, l2te = load_level2()
    ood_l1_raw = ood_share(l1tr.values, l1te.values)
    ood_l1_raw["level"] = "L1_1min_raw"
    ood_l1_raw["feature_name"] = list(l1tr.columns)

    sc = MinMaxScaler()
    x1tr = sc.fit_transform(l1tr.values)
    x1te = sc.transform(l1te.values)
    ood_l1_scaled = ood_share(x1tr, x1te)
    # After scaling, train is in [0,1]; OOD = outside [0,1]
    ood_l1_scaled["level"] = "L1_1min_minmax_fit_BTC"
    ood_l1_scaled["feature_name"] = list(l1tr.columns)
    # redefine outside relative to [0,1]
    for j, name in enumerate(l1tr.columns):
        x = x1te[:, j]
        ood_l1_scaled.loc[j, "test_outside_share"] = float(((x < 0) | (x > 1)).mean())
        ood_l1_scaled.loc[j, "test_below_share"] = float((x < 0).mean())
        ood_l1_scaled.loc[j, "test_above_share"] = float((x > 1).mean())

    ood_l2_raw = ood_share(l2tr.values, l2te.values)
    ood_l2_raw["level"] = "L2_raw"
    ood_l2_raw["feature_name"] = list(l2tr.columns)
    sc2 = MinMaxScaler()
    x2tr = sc2.fit_transform(l2tr.values)
    x2te = sc2.transform(l2te.values)
    ood_l2_scaled = ood_share(x2tr, x2te)
    ood_l2_scaled["level"] = "L2_minmax_fit_BTC"
    ood_l2_scaled["feature_name"] = list(l2tr.columns)
    for j in range(x2te.shape[1]):
        x = x2te[:, j]
        ood_l2_scaled.loc[j, "test_outside_share"] = float(((x < 0) | (x > 1)).mean())

    ood = pd.concat([ood_l1_raw, ood_l1_scaled, ood_l2_raw, ood_l2_scaled], ignore_index=True)
    ood.to_csv(OUT / "ood_feature_shares.csv", index=False)

    # --- Scale-free L1 vs raw L1 (5-min), BTC->ETH ---
    l1tr5, l1te5 = load_level1("5min")
    sf_tr = scale_free_l1(l1tr5)
    sf_te = scale_free_l1(l1te5)
    sc_sf = MinMaxScaler()
    xsf_tr = sc_sf.fit_transform(sf_tr.values)
    xsf_te = sc_sf.transform(sf_te.values)
    sc_raw = MinMaxScaler()
    xraw_tr = sc_raw.fit_transform(l1tr5.values)
    xraw_te = sc_raw.transform(l1te5.values)

    split_rows = []
    split_rows.append(
        eval_split("L1_5min_rawOHLCV_BTC_to_ETH", xraw_tr, xraw_te, l1te5["Close"].to_numpy())
    )
    split_rows.append(
        eval_split("L1_5min_scaleFree_BTC_to_ETH", xsf_tr, xsf_te, l1te5["Close"].to_numpy())
    )

    # ETH -> BTC (scale-free and raw), labels on BTC test close
    sc_e2b = MinMaxScaler()
    x_e2b_tr = sc_e2b.fit_transform(l1te5.values)  # fit on ETH
    x_e2b_te = sc_e2b.transform(l1tr5.values)
    split_rows.append(
        eval_split("L1_5min_rawOHLCV_ETH_to_BTC", x_e2b_tr, x_e2b_te, l1tr5["Close"].to_numpy())
    )
    sc_e2b_sf = MinMaxScaler()
    x_e2b_sf_tr = sc_e2b_sf.fit_transform(sf_te.values)
    x_e2b_sf_te = sc_e2b_sf.transform(sf_tr.values)
    split_rows.append(
        eval_split(
            "L1_5min_scaleFree_ETH_to_BTC",
            x_e2b_sf_tr,
            x_e2b_sf_te,
            l1tr5["Close"].to_numpy(),
        )
    )

    # Within-asset time split BTC: first 70% train, last 30% test
    def time_split(df, frac=0.7):
        cut = int(len(df) * frac)
        return df.iloc[:cut], df.iloc[cut:]

    btr, bte = time_split(l1tr5, 0.7)
    sc_t = MinMaxScaler()
    xbtr = sc_t.fit_transform(scale_free_l1(btr).values)
    xbte = sc_t.transform(scale_free_l1(bte).values)
    split_rows.append(
        eval_split("L1_5min_scaleFree_BTC_time_split", xbtr, xbte, bte["Close"].to_numpy())
    )
    etr, ete = time_split(l1te5, 0.7)
    sc_t2 = MinMaxScaler()
    xetr = sc_t2.fit_transform(scale_free_l1(etr).values)
    xete = sc_t2.transform(scale_free_l1(ete).values)
    split_rows.append(
        eval_split("L1_5min_scaleFree_ETH_time_split", xetr, xete, ete["Close"].to_numpy())
    )

    # L2 BTC->ETH forest for reference
    split_rows.append(
        eval_split(
            "L2_BTC_to_ETH_forest",
            x2tr,
            x2te,
            # use midpoint if available else zeros (label still needs price)
            __import__("paper_eval.v2.common_labels", fromlist=["load_l2_with_midpoint"])
            .load_l2_with_midpoint()[1]
            .to_numpy(),
        )
    )

    splits = pd.DataFrame(split_rows)
    splits.to_csv(OUT / "split_designs_forest_seed42.csv", index=False)

    # Relate scores to OOD distance for L1 raw BTC->ETH: distance = mean clipped OOD magnitude
    # Use fraction of features outside train minmax per row as simple OOD score
    lo = l1tr5.values.min(axis=0)
    hi = l1tr5.values.max(axis=0)
    outside = (l1te5.values < lo) | (l1te5.values > hi)
    row_ood = outside.mean(axis=1)
    scores_raw, _ = fit_if_and_score(xraw_tr, xraw_te, 42)
    corr = float(pd.Series(scores_raw).corr(pd.Series(row_ood), method="spearman"))
    ood_corr = pd.DataFrame(
        [
            {
                "setup": "L1_5min_rawOHLCV_BTC_to_ETH",
                "spearman_score_vs_row_ood_frac": corr,
                "mean_row_ood_frac": float(row_ood.mean()),
            }
        ]
    )
    ood_corr.to_csv(OUT / "score_vs_ood_correlation.csv", index=False)

    config = {
        "task": 5,
        "seeds": [42],
        "scale_free_features": list(sf_tr.columns),
        "splits": [
            "BTC->ETH",
            "ETH->BTC",
            "BTC time 70/30",
            "ETH time 70/30",
        ],
        "detector_for_splits": "sklearn IsolationForest (continuous scores)",
        "note": "Deep models not fully re-trained here; forest used as fast OOD/split probe.",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = [
        "# Task 5 — OOD and split designs",
        "",
        "## OOD shares (selected)",
        "",
        ood.groupby("level")["test_outside_share"].mean().to_string(),
        "",
        "## Split designs (forest, primary price-range style labels on test close/mid)",
        "",
        splits[
            [
                "setup",
                "n_train",
                "n_test",
                "base_rate",
                "p5_precision",
                "p5_recall",
                "p5_lift",
                "frac_rows_in_tied_score_groups",
            ]
        ].to_string(index=False),
        "",
        "## Score vs OOD correlation",
        "",
        ood_corr.to_string(index=False),
        "",
        "## Finding",
        "",
        "Raw BTC->ETH L1 levels are severely distribution-shifted. Scale-free L1 "
        "and within-asset time splits test whether forest failure is OOD-driven. "
        "Deep-model OOD re-training is left for a later heavy run if needed.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(splits[["setup", "p5_precision", "p5_recall", "p5_lift"]].to_string(index=False))
    print(ood_corr.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
