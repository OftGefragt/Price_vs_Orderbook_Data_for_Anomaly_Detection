"""Task 2: build primary non-circular price labels and re-score existing dumps."""

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
from paper_eval.v2.common_labels import build_task2_label_bundle
from paper_eval.v2.labels import ranking_metrics


OUT = REPO_ROOT / "results_v2" / "task2_common_labels"


def _load_scores(path: Path) -> np.ndarray:
    return pd.read_csv(path)["score"].to_numpy(dtype=np.float64)


def _windowize_if_needed(scores: np.ndarray, n_windows: int) -> np.ndarray:
    if len(scores) == n_windows:
        return scores
    if len(scores) == n_windows + 29:
        return sliding_windows(scores.reshape(-1, 1), 30, 1)[:, :, 0].max(axis=1)
    m = min(len(scores), n_windows)
    return scores[:m]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = build_task2_label_bundle(rate=0.01, window=30)

    # Save label vectors and prevalence
    prev_rows = []
    for level, kinds in bundle["labels"].items():
        for kind, payload in kinds.items():
            lab = payload["labels"]
            meta = payload["meta"]
            np.save(OUT / f"labels_{level}_{kind}.npy", lab)
            np.save(OUT / f"amps_{level}_{kind}.npy", payload["amplitudes"])
            prev_rows.append({"level": level, "label_kind": kind, **meta})
    prevalence = pd.DataFrame(prev_rows)
    prevalence.to_csv(OUT / "label_prevalence.csv", index=False)

    # Metrics on matched score dumps (seed 42)
    score_root = REPO_ROOT / "results" / "matched_2021_spring"
    models = {
        "f-AnoGAN": "scores_f_anogan_{lvl}_seed42.npz",
        "GA-EIF+SVM": "scores_ga_eif_svm_{lvl}_seed42.npz",
        "GRU-DeepSVDD": "scores_gru_deepsvdd_{lvl}_seed42.npz",
    }
    metric_rows = []
    for level in ("L1", "L2"):
        for kind, payload in bundle["labels"][level].items():
            lab = payload["labels"]
            n_win = len(lab)
            for model, pat in models.items():
                path = score_root / pat.format(lvl=level.lower())
                if not path.exists():
                    continue
                scores = _windowize_if_needed(_load_scores(path), n_win)
                lab_use = lab[: len(scores)]
                for flag_frac in (0.05, 0.10):
                    m = ranking_metrics(scores, lab_use, flag_frac)
                    metric_rows.append(
                        {
                            "level": level,
                            "label_kind": kind,
                            "model": model,
                            "seed": 42,
                            "score_root": "results/matched_2021_spring",
                            **m,
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics_seed42.csv", index=False)

    config = {
        "task": 2,
        "primary_label": "primary_price_range (L1 Close range; L2 mid-price range)",
        "secondary_label": "secondary_legacy (baseline circular / scaled close range)",
        "alternative_label": "alternative_abs_log_return",
        "positive_rate": 0.01,
        "window": 30,
        "alignment": bundle["alignment"],
        "l2_midpoint_source": bundle["l2_midpoint_source"],
        "l2_btc_midpoint_available": bundle["l2_btc_midpoint_available"],
        "notes": bundle["notes"],
        "seeds": [42],
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Plain finding
    prim = metrics[metrics["label_kind"] == "primary_price_range"]
    prim5 = prim[prim["flag_frac"] == 0.05]
    lines = [
        "# Task 2 — Common non-circular labels",
        "",
        "## Data fact",
        "",
        "L2 GA-EIF test features are numerically identical to the 30 overlapping "
        "columns in AnoGAN `ETH_5min.csv`, which also provides `midpoint` and "
        "`system_time`. BTC AnoGAN train mid-price file is empty in-repo.",
        "",
        "## Alignment",
        "",
        json.dumps(bundle["alignment"], indent=2),
        "",
        "## Prevalence (exact top 1%)",
        "",
        prevalence.to_string(index=False),
        "",
        "## Seed-42 P@5% under primary price-range label",
        "",
        prim5.to_string(index=False),
        "",
        "## Finding",
        "",
        "Primary L2 labels are no longer a function of the 30 book features. "
        "Compare primary metrics to secondary_legacy to see how much of the "
        "old L2 advantage depended on circular feature-norm stress.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(prevalence.to_string(index=False))
    print()
    print(prim5.to_string(index=False))
    print(f"Wrote under {OUT}")


if __name__ == "__main__":
    main()
