"""Task 6: agreement tables (existing deep seeds + forest seeds 42-51).

Deep models: use matched dumps seeds 42-44 (10-seed deep retrain not run yet).
Forest: fit sklearn IF for seeds 42-51 on L1 scale-free 5-min and L2.
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
from paper_eval.metrics import jaccard, spearman_corr
from paper_eval.v2.common_labels import load_l2_with_midpoint, price_panel_from_series
from paper_eval.v2.events import merge_windows_to_events
from paper_eval.v2.labels import top_fraction_labels, window_amplitudes
from paper_eval.v2.run_task5_ood_splits import scale_free_l1


OUT = REPO_ROOT / "results_v2" / "task6_agreement"
MATCHED = REPO_ROOT / "results" / "matched_2021_spring"


def top_flags(scores: np.ndarray, frac: float) -> np.ndarray:
    n = len(scores)
    k = max(1, int(round(n * frac)))
    order = np.lexsort((np.arange(n), -scores))[:k]
    f = np.zeros(n, dtype=bool)
    f[order] = True
    return f


def chance_jaccard(p: float) -> float:
    # For independent Bernoulli(p) flags: E[Jaccard] = p/(2-p)
    return p / (2.0 - p) if p < 2 else float("nan")


def event_flags_from_windows(flags: np.ndarray, gap: int = 30) -> np.ndarray:
    """Binary cover vector of event-merged windows (for Jaccard on covers)."""
    cover = np.zeros(len(flags), dtype=bool)
    for s, e in merge_windows_to_events(flags, gap=gap):
        cover[s : e + 1] = True
    return cover


def load_matched_score(model_key: str, level: str, seed: int, n_windows: int) -> np.ndarray:
    names = {
        "f-AnoGAN": "f_anogan",
        "GA-EIF+SVM": "ga_eif_svm",
        "GRU-DeepSVDD": "gru_deepsvdd",
    }
    path = MATCHED / f"scores_{names[model_key]}_{level.lower()}_seed{seed}.npz"
    s = pd.read_csv(path)["score"].to_numpy(dtype=np.float64)
    if len(s) == n_windows:
        return s
    if len(s) == n_windows + 29:
        return sliding_windows(s.reshape(-1, 1), 30, 1)[:, :, 0].max(axis=1)
    return s[:n_windows]


def forest_scores(x_tr, x_te, seed: int) -> np.ndarray:
    clf = IsolationForest(
        n_estimators=235,
        max_samples=min(1691, len(x_tr)),
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_tr)
    return -clf.score_samples(x_te)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Prepare matrices
    l1tr, l1te = load_level1("5min")
    sf_tr = scale_free_l1(l1tr)
    sf_te = scale_free_l1(l1te)
    sc = MinMaxScaler()
    x1tr = sc.fit_transform(sf_tr.values)
    x1te = sc.transform(sf_te.values)
    lab1, _ = top_fraction_labels(
        window_amplitudes(price_panel_from_series(l1te["Close"].to_numpy()), "close_range"),
        0.01,
    )

    l2tr, l2te = load_level2()
    sc2 = MinMaxScaler()
    x2tr = sc2.fit_transform(l2tr.values)
    x2te = sc2.transform(l2te.values)
    mid = load_l2_with_midpoint()[1].to_numpy()
    lab2, _ = top_fraction_labels(
        window_amplitudes(price_panel_from_series(mid), "close_range"), 0.01
    )

    n1, n2 = len(lab1), len(lab2)

    # Pairwise Jaccard across models per seed (deep 42-44 + forest scale-free)
    pair_rows = []
    stab_rows = []
    spear_rows = []
    budgets = [0.01, 0.05, 0.10, 0.20]

    deep_seeds = [42, 43, 44]
    forest_seeds = list(range(42, 52))

    # Precompute forest window scores for many seeds
    forest_l1 = {
        s: sliding_windows(forest_scores(x1tr, x1te, s).reshape(-1, 1), 30, 1)[:, :, 0].max(1)
        for s in forest_seeds
    }
    forest_l2 = {
        s: sliding_windows(forest_scores(x2tr, x2te, s).reshape(-1, 1), 30, 1)[:, :, 0].max(1)
        for s in forest_seeds
    }

    for level, n_win, lab, forest_map in [
        ("L1", n1, lab1, forest_l1),
        ("L2", n2, lab2, forest_l2),
    ]:
        for seed in deep_seeds:
            scores = {
                "f-AnoGAN": load_matched_score("f-AnoGAN", level, seed, n_win),
                "GRU-DeepSVDD": load_matched_score("GRU-DeepSVDD", level, seed, n_win),
                "Forest-scaleFreeIF": forest_map[seed][:n_win],
            }
            # also include original GA-EIF dump (raw-feature IF/EIF) for reference
            scores["GA-EIF_dump"] = load_matched_score("GA-EIF+SVM", level, seed, n_win)

            for frac in budgets:
                flags = {m: top_flags(sc, frac) for m, sc in scores.items()}
                models = list(flags)
                for i, a in enumerate(models):
                    for b in models[i + 1 :]:
                        for mode, fa, fb in [
                            ("window", flags[a], flags[b]),
                            (
                                "event_cover_gap30",
                                event_flags_from_windows(flags[a]),
                                event_flags_from_windows(flags[b]),
                            ),
                        ]:
                            pair_rows.append(
                                {
                                    "level": level,
                                    "seed": seed,
                                    "flag_frac": frac,
                                    "mode": mode,
                                    "model_a": a,
                                    "model_b": b,
                                    "jaccard": jaccard(fa, fb),
                                    "chance_jaccard": chance_jaccard(frac),
                                }
                            )
                for a in models:
                    for b in models:
                        if a >= b:
                            continue
                        spear_rows.append(
                            {
                                "level": level,
                                "seed": seed,
                                "model_a": a,
                                "model_b": b,
                                "spearman": spearman_corr(scores[a], scores[b]),
                            }
                        )

        # Seed stability for forest across 42-51
        for frac in budgets:
            jvals = []
            seeds = forest_seeds
            base = top_flags(forest_map[seeds[0]][:n_win], frac)
            for s in seeds[1:]:
                jvals.append(jaccard(base, top_flags(forest_map[s][:n_win], frac)))
            # all pairs mean
            allj = []
            for i, s1 in enumerate(seeds):
                for s2 in seeds[i + 1 :]:
                    allj.append(
                        jaccard(
                            top_flags(forest_map[s1][:n_win], frac),
                            top_flags(forest_map[s2][:n_win], frac),
                        )
                    )
            stab_rows.append(
                {
                    "level": level,
                    "detector": "Forest-scaleFreeIF" if level == "L1" else "Forest-IF",
                    "flag_frac": frac,
                    "n_seeds": len(seeds),
                    "jaccard_mean": float(np.mean(allj)),
                    "jaccard_std": float(np.std(allj)),
                    "jaccard_min": float(np.min(allj)),
                    "jaccard_max": float(np.max(allj)),
                    "chance_jaccard": chance_jaccard(frac),
                    "deterministic": False,
                }
            )

        # Deep seed stability 42-44
        for model in ("f-AnoGAN", "GRU-DeepSVDD", "GA-EIF+SVM"):
            for frac in budgets:
                scs = [
                    load_matched_score(model, level, s, n_win) for s in deep_seeds
                ]
                allj = []
                for i in range(len(scs)):
                    for j in range(i + 1, len(scs)):
                        allj.append(
                            jaccard(top_flags(scs[i], frac), top_flags(scs[j], frac))
                        )
                stab_rows.append(
                    {
                        "level": level,
                        "detector": model,
                        "flag_frac": frac,
                        "n_seeds": len(deep_seeds),
                        "jaccard_mean": float(np.mean(allj)),
                        "jaccard_std": float(np.std(allj)),
                        "jaccard_min": float(np.min(allj)),
                        "jaccard_max": float(np.max(allj)),
                        "chance_jaccard": chance_jaccard(frac),
                        "deterministic": False,
                        "note": "deep seeds 42-44 only; 10-seed deep retrain pending",
                    }
                )

    pairs = pd.DataFrame(pair_rows)
    stab = pd.DataFrame(stab_rows)
    spear = pd.DataFrame(spear_rows)
    # Aggregate pairs across seeds
    agg = (
        pairs.groupby(["level", "flag_frac", "mode", "model_a", "model_b"], as_index=False)
        .agg(
            jaccard_mean=("jaccard", "mean"),
            jaccard_std=("jaccard", "std"),
            jaccard_min=("jaccard", "min"),
            jaccard_max=("jaccard", "max"),
            chance_jaccard=("chance_jaccard", "first"),
            n_seeds=("seed", "nunique"),
        )
    )
    pairs.to_csv(OUT / "pairwise_jaccard_per_seed.csv", index=False)
    agg.to_csv(OUT / "pairwise_jaccard_agg.csv", index=False)
    stab.to_csv(OUT / "seed_stability_jaccard.csv", index=False)
    spear.to_csv(OUT / "spearman_per_seed.csv", index=False)
    spear_agg = (
        spear.groupby(["level", "model_a", "model_b"], as_index=False)
        .agg(spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"))
    )
    spear_agg.to_csv(OUT / "spearman_agg.csv", index=False)

    config = {
        "task": 6,
        "deep_seeds": deep_seeds,
        "forest_seeds": forest_seeds,
        "l1_forest_features": "scale_free",
        "budgets": budgets,
        "incomplete": "f-AnoGAN/GRU not re-run for seeds 45-51",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "SUMMARY.md").write_text(
        "# Task 6 — Agreement\n\n"
        "Deep pairwise Jaccard uses matched dumps seeds 42–44. "
        "Forest seed-stability uses 42–51. Chance level = p/(2-p).\n\n"
        + agg[agg["flag_frac"] == 0.05].to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(agg[agg["flag_frac"] == 0.05].head(20).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
