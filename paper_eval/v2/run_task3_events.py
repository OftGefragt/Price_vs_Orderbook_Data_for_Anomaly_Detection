"""Task 3: event-level evaluation under primary price labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import sliding_windows
from paper_eval.v2.events import event_level_bootstrap, merge_windows_to_events
from paper_eval.v2.labels import top_fraction_labels


OUT = REPO_ROOT / "results_v2" / "task3_events"
LABEL_DIR = REPO_ROOT / "results_v2" / "task2_common_labels"
SCORE_ROOT = REPO_ROOT / "results" / "matched_2021_spring"


def _scores(path: Path, n_windows: int) -> np.ndarray:
    s = pd.read_csv(path)["score"].to_numpy(dtype=np.float64)
    if len(s) == n_windows:
        return s
    if len(s) == n_windows + 29:
        return sliding_windows(s.reshape(-1, 1), 30, 1)[:, :, 0].max(axis=1)
    return s[:n_windows]


def _top_flags(scores: np.ndarray, frac: float = 0.05) -> np.ndarray:
    n = len(scores)
    k = max(1, int(round(n * frac)))
    idx = np.arange(n)
    order = np.lexsort((idx, -scores))[:k]
    flags = np.zeros(n, dtype=bool)
    flags[order] = True
    return flags


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    event_count_rows = []
    models = {
        "f-AnoGAN": "scores_f_anogan_{lvl}_seed42.npz",
        "GA-EIF+SVM": "scores_ga_eif_svm_{lvl}_seed42.npz",
        "GRU-DeepSVDD": "scores_gru_deepsvdd_{lvl}_seed42.npz",
    }
    for level in ("L1", "L2"):
        lab = np.load(LABEL_DIR / f"labels_{level}_primary_price_range.npy")
        for gap in (15, 30, 60):
            truth_events = merge_windows_to_events(lab, gap=gap)
            event_count_rows.append(
                {
                    "level": level,
                    "gap": gap,
                    "n_positive_windows": int(lab.sum()),
                    "n_positive_events": len(truth_events),
                    "mean_event_span_windows": float(
                        np.mean([e - s + 1 for s, e in truth_events])
                    )
                    if truth_events
                    else 0.0,
                }
            )
            for model, pat in models.items():
                path = SCORE_ROOT / pat.format(lvl=level.lower())
                scores = _scores(path, len(lab))
                flags = _top_flags(scores, 0.05)
                boot = event_level_bootstrap(
                    flags, lab, gap=gap, block_len=30, n_boot=400, seed=42
                )
                rows.append(
                    {
                        "level": level,
                        "model": model,
                        "label_kind": "primary_price_range",
                        "flag_frac": 0.05,
                        "seed": 42,
                        **boot,
                    }
                )

    events_df = pd.DataFrame(event_count_rows)
    metrics_df = pd.DataFrame(rows)
    events_df.to_csv(OUT / "positive_event_counts.csv", index=False)
    metrics_df.to_csv(OUT / "event_metrics_seed42.csv", index=False)
    config = {
        "task": 3,
        "label": "primary_price_range from task2",
        "gaps_tested": [15, 30, 60],
        "flag_frac": 0.05,
        "bootstrap": {"n_boot": 400, "block_len": 30, "seed": 42},
        "seeds": [42],
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    g30 = events_df[events_df["gap"] == 30]
    m30 = metrics_df[metrics_df["gap"] == 30]
    summary = [
        "# Task 3 — Event-level evaluation",
        "",
        "## Positive events (gap=30)",
        "",
        g30.to_string(index=False),
        "",
        "## Event metrics @ top 5% flags (gap=30, seed 42)",
        "",
        m30[
            [
                "level",
                "model",
                "n_label_events",
                "n_pred_events",
                "event_precision",
                "event_precision_ci95_lo",
                "event_precision_ci95_hi",
                "event_recall",
                "event_recall_ci95_lo",
                "event_recall_ci95_hi",
                "event_jaccard_windowcover",
            ]
        ].to_string(index=False),
        "",
        "## Finding",
        "",
        "Overlapping windows collapse many positive windows into few independent "
        "stress episodes. If L2 has very few label events, window-level recall "
        "near 1.0 is not evidence of many independent detections.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(g30.to_string(index=False))
    print(m30[["level", "model", "n_label_events", "event_precision", "event_recall"]].to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
