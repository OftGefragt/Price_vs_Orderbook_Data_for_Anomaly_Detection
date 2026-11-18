"""Task 4: forest-only vs forest+SVM ranking; document GA fitness/leakage status."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import load_level1, load_level2, scale_fit_transform, sliding_windows
from paper_eval.v2.labels import ranking_metrics, top_fraction_labels, window_amplitudes
from paper_eval.v2.common_labels import price_panel_from_series, load_l2_with_midpoint


OUT = REPO_ROOT / "results_v2" / "task4_svm_ga"


def forest_and_svm_scores(x_train, x_test, seed=42):
    """Return continuous forest scores; also SVM hard labels on those scores."""
    n = len(x_train)
    sample_size = min(1691, n)
    forest = IsolationForest(
        n_estimators=235,
        max_samples=sample_size,
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(x_train)
    train_scores = -forest.score_samples(x_train)
    test_scores = -forest.score_samples(x_test)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=max(0.062, 0.01))
    svm.fit(train_scores.reshape(-1, 1))
    svm_pred = svm.predict(test_scores.reshape(-1, 1))  # +1 inlier, -1 outlier
    # Ranking used in paper: forest scores only (SVM discarded). Alternative:
    # use -decision_function as a continuous SVM-on-score ranking.
    svm_rank = -svm.decision_function(test_scores.reshape(-1, 1)).ravel()
    return {
        "forest": test_scores,
        "svm_hard": svm_pred,
        "svm_on_forest_score": svm_rank,
        "spearman_forest_vs_svm_rank": float(
            pd.Series(test_scores).rank().corr(pd.Series(svm_rank).rank())
        ),
        "frac_svm_flag_outlier": float(np.mean(svm_pred == -1)),
    }


def window_max(scores_row: np.ndarray) -> np.ndarray:
    return sliding_windows(scores_row.reshape(-1, 1), 30, 1)[:, :, 0].max(axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # L1 5-min
    l1tr, l1te = load_level1("5min")
    x1tr, x1te, _ = scale_fit_transform(l1tr, l1te)
    lab1, _ = top_fraction_labels(
        window_amplitudes(price_panel_from_series(l1te["Close"].to_numpy()), "close_range"),
        0.01,
    )
    # L2
    l2tr_df, l2te_df = load_level2()
    x2tr, x2te, _ = scale_fit_transform(l2tr_df, l2te_df)
    _, mid, _ = load_l2_with_midpoint()
    lab2, _ = top_fraction_labels(
        window_amplitudes(price_panel_from_series(mid.to_numpy()), "close_range"),
        0.01,
    )

    for level, xtr, xte, lab in [
        ("L1", x1tr, x1te, lab1),
        ("L2", x2tr, x2te, lab2),
    ]:
        pack = forest_and_svm_scores(xtr, xte, seed=42)
        for name, raw in [
            ("forest_only", pack["forest"]),
            ("svm_rank_on_forest_score", pack["svm_on_forest_score"]),
        ]:
            pr = window_max(raw)
            for flag_frac in (0.05, 0.10):
                m = ranking_metrics(pr, lab[: len(pr)], flag_frac)
                rows.append(
                    {
                        "level": level,
                        "ranking": name,
                        "seed": 42,
                        "spearman_forest_vs_svm_rank": pack["spearman_forest_vs_svm_rank"],
                        "frac_svm_outlier": pack["frac_svm_flag_outlier"],
                        **m,
                    }
                )
        # Paper pipeline equivalence: does discarding SVM change top-5% set vs forest?
        f_flags = ranking_metrics(window_max(pack["forest"]), lab, 0.05)
        # identical ranking object
        rows.append(
            {
                "level": level,
                "ranking": "paper_pipeline_note",
                "seed": 42,
                "note": (
                    "paper_eval.models_eif.score returns forest scores; "
                    "svm.predict output is discarded"
                ),
                "spearman_forest_vs_svm_rank": pack["spearman_forest_vs_svm_rank"],
                **f_flags,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "svm_ablation_seed42.csv", index=False)

    ga_doc = {
        "fitness_if_labels_provided": "F1 of thresholded EIF scores vs labels",
        "fitness_in_recorded_notebook_run": "np.var(scores) (labels omitted)",
        "recorded_call": "best_params = ga_eif(X, pop_size=10, generations=10)",
        "recorded_champion": {
            "ntrees": 235,
            "sample_size": 1691,
            "contamination": 0.062,
            "exlevel": 26,
        },
        "eth_test_labels_used": False,
        "train_only_matrix_verified": False,
        "reason_unverified": "X came from missing Colab file df_selection.csv",
        "paper_eval_reruns_ga": False,
        "recommendation": (
            "Describe hyperparameters as fixed inherited constants from a "
            "label-free variance search on an unreproducible Colab matrix; "
            "do not claim a leakage-free audited GA unless re-run on BTC train only."
        ),
        "contamination_and_continuous_scores": (
            "sklearn IsolationForest contamination does not change the ordering of "
            "score_samples; it affects the hard decision offset. Pure EIF ignores "
            "contamination for path-length scores. Therefore the tuned contamination "
            "value does not affect the paper's Precision/Recall/Jaccard under "
            "top-quantile flagging of continuous scores."
        ),
        "svm_paper_wording": (
            "A one-class SVM was fit on one-dimensional training forest scores and "
            "invoked at inference, but its predictions were not used to rank test "
            "windows. Reported metrics use continuous forest anomaly scores only. "
            "Rename the detector to GA-EIF (or IF) for metric claims, or report the "
            "SVM as an unused post-processor."
        ),
    }
    (OUT / "ga_and_svm_documentation.json").write_text(
        json.dumps(ga_doc, indent=2), encoding="utf-8"
    )
    config = {"task": 4, "seeds": [42], "backend": "sklearn_iforest"}
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    compare = df[df["ranking"].isin(["forest_only", "svm_rank_on_forest_score"])]
    summary = [
        "# Task 4 — SVM stage and GA check",
        "",
        "## SVM ablation (primary price label, seed 42)",
        "",
        compare.to_string(index=False),
        "",
        "## Documentation for the paper",
        "",
        "```json",
        json.dumps(ga_doc, indent=2),
        "```",
        "",
        "## Finding",
        "",
        "The published pipeline's ranking ignores SVM output. Spearman between "
        "forest scores and SVM-on-score ranking is reported in the CSV. "
        "Contamination does not affect continuous-score top-quantile metrics. "
        "GA was label-free variance on an unverifiable Colab matrix.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(compare.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
