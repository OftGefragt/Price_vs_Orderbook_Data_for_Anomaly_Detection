"""Task 1: show baseline positive-share mismatch and fix to exact top 1%.

Writes under results_v2/task1_label_prevalence/.
Does not modify baseline_v1 outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import load_level1, load_level2, scale_fit_transform
from paper_eval.v2.labels import (
    baseline_quantile_labels,
    ranking_metrics,
    top_fraction_labels,
    window_amplitudes,
)


OUT = REPO_ROOT / "results_v2" / "task1_label_prevalence"


def _track_amplitudes(level: str, bar: str = "1min"):
    if level == "L1":
        tr, te = load_level1(bar=bar)
        x_tr, x_te, _ = scale_fit_transform(tr, te)
        amps = window_amplitudes(x_te, "close_range", window=30)
        kind = "close_range_on_scaled_close"
        rows = len(te)
    elif level == "L2":
        tr, te = load_level2()
        x_tr, x_te, _ = scale_fit_transform(tr, te)
        amps = window_amplitudes(x_te, "feature_norm_range", window=30)
        kind = "feature_norm_range_on_scaled_l2"
        rows = len(te)
    else:
        raise ValueError(level)
    return amps, kind, rows, x_te


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for level, bar in [("L1", "1min"), ("L1", "5min"), ("L2", "5min")]:
        if level == "L2":
            amps, kind, nrows, x_te = _track_amplitudes("L2")
            bar_used = "native_l2"
        else:
            amps, kind, nrows, x_te = _track_amplitudes("L1", bar=bar)
            bar_used = bar

        base_lab, base_meta = baseline_quantile_labels(amps, q=0.99)
        fix_lab, fix_meta = top_fraction_labels(amps, rate=0.01)

        for rule, lab, meta in [
            ("baseline_quantile_geq", base_lab, base_meta),
            ("exact_top_1pct_tiebreak", fix_lab, fix_meta),
        ]:
            rows.append(
                {
                    "level": level,
                    "bar": bar_used,
                    "amplitude_kind": kind,
                    "label_rule": rule,
                    "n_test_rows": int(nrows),
                    **meta,
                }
            )

        # Re-score baseline_v1 detectors if score dumps exist (matched or strong).
        # Prefer matched for L1-5min / L2; strong for L1-1min.
        score_roots = []
        if level == "L1" and bar == "1min":
            score_roots.append(REPO_ROOT / "results")
        if level == "L1" and bar == "5min":
            score_roots.append(REPO_ROOT / "results" / "matched_2021_spring")
        if level == "L2":
            score_roots.append(REPO_ROOT / "results" / "matched_2021_spring")
            score_roots.append(REPO_ROOT / "results")

        models = {
            "f-AnoGAN": "scores_f_anogan_{lvl}_seed{seed}.npz",
            "GA-EIF+SVM": "scores_ga_eif_svm_{lvl}_seed{seed}.npz",
            "GRU-DeepSVDD": "scores_gru_deepsvdd_{lvl}_seed{seed}.npz",
        }
        lvl = level.lower()
        for root in score_roots:
            for model, pattern in models.items():
                path = root / pattern.format(lvl=lvl, seed=42)
                if not path.exists():
                    continue
                # run_comparison writes CSV content into *.npz filenames.
                scores = pd.read_csv(path)["score"].to_numpy(dtype=np.float64)

                # Align: EIF scores are per-row; deep models are per-window.
                if len(scores) == len(amps) + 29:
                    from paper_eval.data import sliding_windows

                    win = sliding_windows(scores.reshape(-1, 1), 30, 1)[:, :, 0]
                    pr_scores = win.max(axis=1)
                    lab_base, lab_fix = base_lab, fix_lab
                elif len(scores) == len(amps):
                    pr_scores = scores
                    lab_base, lab_fix = base_lab, fix_lab
                else:
                    m = min(len(scores), len(amps))
                    pr_scores = scores[:m]
                    lab_base, lab_fix = base_lab[:m], fix_lab[:m]

                for rule, lab in [
                    ("baseline_quantile_geq", lab_base),
                    ("exact_top_1pct_tiebreak", lab_fix),
                ]:
                    for flag_frac in (0.05, 0.10):
                        mtr = ranking_metrics(pr_scores, lab, flag_frac)
                        rows.append(
                            {
                                "level": level,
                                "bar": bar_used,
                                "amplitude_kind": kind,
                                "label_rule": rule,
                                "model": model,
                                "score_root": str(root.relative_to(REPO_ROOT)),
                                "seed": 42,
                                **mtr,
                            }
                        )

    df = pd.DataFrame(rows)
    if "model" not in df.columns:
        df["model"] = np.nan
    prevalence = df[df["model"].isna()].copy()
    metrics = df[df["model"].notna()].copy()
    # Drop empty metric columns from prevalence table for readability.
    prevalence = prevalence.drop(
        columns=[c for c in prevalence.columns if c in {
            "model", "score_root", "seed", "flag_frac", "n", "k_flags",
            "n_positives", "precision", "recall", "lift",
            "max_achievable_precision", "tp",
        }],
        errors="ignore",
    )

    prevalence_path = OUT / "label_prevalence.csv"
    metrics_path = OUT / "metrics_with_lift_seed42.csv"
    prevalence.to_csv(prevalence_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    config = {
        "task": 1,
        "window": 30,
        "baseline_rule": "amps >= np.quantile(amps, 0.99)",
        "fixed_rule": "exact top round(n*0.01) with amp desc then index asc",
        "seeds_for_metrics": [42],
        "note": (
            "Positive share mismatch under baseline comes from inclusive quantile "
            "ties, especially on L2 feature-norm amplitudes."
        ),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Human summary
    lines = [
        "# Task 1 — Label prevalence",
        "",
        "## Finding",
        "",
        "Baseline `quantile + >=` yields unequal positive shares across tracks "
        "(L2 higher because of ties). Exact top-1% labeling equalizes the "
        "intended base rate to ~1% on each track (up to rounding of k).",
        "",
        "## Prevalence table",
        "",
        prevalence.to_string(index=False),
        "",
    ]
    if not metrics.empty:
        lines += [
            "## Seed-42 metrics with lift (existing score dumps)",
            "",
            "Columns include base_rate, precision, recall, lift=precision/base_rate, "
            "max_achievable_precision=min(1, n_positives/k_flags).",
            "",
            metrics.to_string(index=False),
            "",
        ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(prevalence.to_string(index=False))
    print(f"\nWrote {prevalence_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
