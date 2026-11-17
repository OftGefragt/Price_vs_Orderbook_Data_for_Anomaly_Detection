"""Train and score all model families on Level-1 and Level-2 data.

Modes:
  fast     - subsampled Level-1 1-min, short epochs, one seed (smoke / Colab)
  strong   - large Level-1 1-min subsample, longer epochs, multiple seeds
  matched  - Level-1 aggregated to 5-min bars; shared sklearn IsolationForest
             backend on both tiers; same strong epoch/seed budget
  matched_fast - matched protocol with short epochs / one seed

Writes under results/ (or results/<tag>/ when --out-tag is set):
  table5_main_comparison.csv          per-seed rows
  table5_main_comparison_agg.csv      mean +/- std across seeds
  table5_agreement_L1.csv / _L2.csv   pairwise Jaccard at 5%
  protocol.json                       exact settings used
  scores_*.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import (
    REGIMES,
    load_level1,
    load_level2,
    scale_fit_transform,
)
from paper_eval.metrics import (
    aggregate_seed_rows,
    flag_top_quantile,
    markdown_table,
    pairwise_jaccard_table,
    rows_to_table,
    summarize_run,
)
from paper_eval.models_anogan import FanoganDetector
from paper_eval.models_eif import EIFSVMDetector
from paper_eval.models_gru_svdd import GRUDeepSVDDDetector
from paper_eval.proxy_labels import (
    level1_stress_flags,
    level2_stress_flags,
    precision_recall_at_k,
)


MODES = {
    "fast": {
        "max_l1_train": 12000,
        "max_l1_test": 8000,
        "epochs_gan": 3,
        "epochs_enc": 3,
        "epochs_gru": 3,
        "seeds": [42],
        "use_full_l1": False,
        "l1_bar": "1min",
        "forest_backend": "auto",
        "run_l2": True,
    },
    "strong": {
        # Full ETH test set; BTC train capped for deep-model wall time.
        # Cap and epochs are recorded in results/protocol.json.
        "max_l1_train": 30000,
        "max_l1_test": None,
        "epochs_gan": 10,
        "epochs_enc": 15,
        "epochs_gru": 8,
        "seeds": [42, 43, 44],
        "use_full_l1": False,
        "l1_bar": "1min",
        "forest_backend": "auto",
        "run_l2": True,
    },
    "matched": {
        # Resolution-matched protocol: L1 -> 5-min bars; shared sklearn forest.
        "max_l1_train": None,
        "max_l1_test": None,
        "epochs_gan": 10,
        "epochs_enc": 15,
        "epochs_gru": 8,
        "seeds": [42, 43, 44],
        "use_full_l1": True,
        "l1_bar": "5min",
        "forest_backend": "sklearn",
        "run_l2": True,
    },
    "matched_fast": {
        "max_l1_train": None,
        "max_l1_test": None,
        "epochs_gan": 3,
        "epochs_enc": 3,
        "epochs_gru": 3,
        "seeds": [42],
        "use_full_l1": True,
        "l1_bar": "5min",
        "forest_backend": "sklearn",
        "run_l2": True,
    },
}


def _subsample(x: np.ndarray, max_rows: int | None, seed: int = 42) -> np.ndarray:
    if max_rows is None or len(x) <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=max_rows, replace=False))
    return x[idx]


def _eif_for_level(
    level: str,
    n_features: int,
    seed: int,
    forest_backend: str,
) -> EIFSVMDetector:
    # Use GA best extension when feature dim allows; otherwise clamp.
    preferred = 4 if level == "L1" else 26
    ext = min(preferred, max(0, n_features - 1))
    return EIFSVMDetector(
        extension_level=ext,
        random_state=seed,
        backend=forest_backend,
    )


def run(
    mode: str = "strong",
    max_l1_train: int | None = None,
    max_l1_test: int | None = None,
    seeds: List[int] | None = None,
    regime: str = "2021_spring",
    out_tag: Optional[str] = None,
    levels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}. Choose from {list(MODES)}.")
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime}. Choose from {list(REGIMES)}.")
    cfg = dict(MODES[mode])
    if max_l1_train is not None:
        cfg["max_l1_train"] = max_l1_train
        cfg["use_full_l1"] = False
    if max_l1_test is not None:
        cfg["max_l1_test"] = max_l1_test
        cfg["use_full_l1"] = False
    if seeds is not None:
        cfg["seeds"] = seeds

    regime_meta = REGIMES[regime]
    run_l2 = bool(cfg["run_l2"] and regime_meta.get("has_l2", True))
    if levels is not None:
        run_l2 = run_l2 and ("L2" in levels)
        run_l1 = "L1" in levels
    else:
        run_l1 = True

    tag = out_tag or f"{mode}_{regime}"
    # Keep legacy paths for the original strong/2021 run unless tagged.
    if mode in {"strong", "fast"} and regime == "2021_spring" and out_tag is None:
        out_dir = REPO_ROOT / "results"
    else:
        out_dir = REPO_ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {mode} | regime: {regime} | out: {out_dir}")
    print(f"L1 bar={cfg['l1_bar']} | forest_backend={cfg['forest_backend']}")

    l1_train_full = l1_test_full = None
    if run_l1:
        print("Loading Level-1 data...")
        # Regime CSVs for 2022 are already 5-min; still pass bar for aggregation of 1-min.
        native = regime_meta.get("native_bar")
        bar = cfg["l1_bar"] if native is None else native
        l1_train_df, l1_test_df = load_level1(
            bar=bar,
            train_path=regime_meta["l1_train"],
            test_path=regime_meta["l1_test"],
        )
        print(
            f"  L1 train={len(l1_train_df)} test={len(l1_test_df)} "
            f"bar≈{bar} "
            f"[{l1_train_df.index[0]} .. {l1_train_df.index[-1]}]"
        )
        l1_train_full, l1_test_full, _ = scale_fit_transform(l1_train_df, l1_test_df)

    l2_train = l2_test = None
    if run_l2:
        print("Loading Level-2 data...")
        l2_train_df, l2_test_df = load_level2(
            train_path=regime_meta.get("l2_train"),
            test_path=regime_meta.get("l2_test"),
        )
        l2_train, l2_test, _ = scale_fit_transform(l2_train_df, l2_test_df)
        print(f"  L2 train={len(l2_train)} test={len(l2_test)} feats={l2_train.shape[1]}")
    elif regime_meta.get("has_l2") is False:
        print("Level-2 skipped: order-book archive not bundled for this regime.")

    all_rows: List[dict] = []
    last_flags: Dict[str, Dict[str, np.ndarray]] = {"L1": {}, "L2": {}}

    protocol = {
        "mode": mode,
        "regime": regime,
        "regime_description": regime_meta.get("description"),
        "seeds": cfg["seeds"],
        "epochs_gan": cfg["epochs_gan"],
        "epochs_enc": cfg["epochs_enc"],
        "epochs_gru": cfg["epochs_gru"],
        "use_full_l1": cfg["use_full_l1"],
        "max_l1_train": cfg["max_l1_train"],
        "max_l1_test": cfg["max_l1_test"],
        "l1_bar": cfg["l1_bar"],
        "forest_backend": cfg["forest_backend"],
        "run_l1": run_l1,
        "run_l2": run_l2,
        "l2_train_rows": int(len(l2_train)) if l2_train is not None else 0,
        "l2_test_rows": int(len(l2_test)) if l2_test is not None else 0,
        "proxy_label_quantile": 0.99,
        "window": 30,
        "out_dir": str(out_dir),
    }

    for seed in cfg["seeds"]:
        print(f"\n######## SEED {seed} ########")
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        labels: Dict[str, np.ndarray] = {}

        if run_l1:
            assert l1_train_full is not None and l1_test_full is not None
            l1_train = _subsample(l1_train_full, cfg["max_l1_train"], seed=seed)
            l1_test = _subsample(l1_test_full, cfg["max_l1_test"], seed=seed + 1)
            protocol["l1_train_rows"] = int(len(l1_train))
            protocol["l1_test_rows"] = int(len(l1_test))
            datasets["L1"] = (l1_train, l1_test)
            labels["L1"] = level1_stress_flags(l1_test, window=30, q=0.99)

        if run_l2:
            assert l2_train is not None and l2_test is not None
            datasets["L2"] = (l2_train, l2_test)
            labels["L2"] = level2_stress_flags(l2_test, window=30, q=0.99)

        for level, (x_tr, x_te) in datasets.items():
            jobs = [
                (
                    "f-AnoGAN",
                    FanoganDetector(
                        epochs_gan=cfg["epochs_gan"],
                        epochs_enc=cfg["epochs_enc"],
                        seed=seed,
                    ),
                ),
                (
                    "GA-EIF+SVM",
                    _eif_for_level(
                        level, x_tr.shape[1], seed, cfg["forest_backend"]
                    ),
                ),
                (
                    "GRU-DeepSVDD",
                    GRUDeepSVDDDetector(epochs=cfg["epochs_gru"], seed=seed),
                ),
            ]

            for model_name, detector in jobs:
                print(f"\n=== {model_name} | {level} | seed={seed} ===")
                detector.fit(x_tr)
                scores, infer_s = detector.score(x_te)

                if model_name == "GA-EIF+SVM":
                    from paper_eval.data import sliding_windows

                    row_scores = scores
                    win = sliding_windows(
                        row_scores.reshape(-1, 1), window=30, step=1
                    )[:, :, 0]
                    window_scores = win.max(axis=1)
                    pr_scores = window_scores
                    rank_scores = row_scores
                else:
                    pr_scores = scores
                    rank_scores = scores

                lab = labels[level]
                p5, r5, _ = precision_recall_at_k(pr_scores, lab, 0.05)
                p10, r10, _ = precision_recall_at_k(pr_scores, lab, 0.10)

                score_path = (
                    out_dir
                    / f"scores_{model_name.replace('+', '_').replace('-', '_').lower()}_{level.lower()}_seed{seed}.npz"
                )
                pd.DataFrame({"score": rank_scores}).to_csv(score_path, index=False)

                backend = getattr(detector, "_backend", None)
                row = summarize_run(
                    model=model_name,
                    level=level,
                    scores=rank_scores if model_name == "GA-EIF+SVM" else pr_scores,
                    train_seconds=getattr(detector, "train_seconds", 0.0),
                    infer_seconds=infer_s,
                    seed=seed,
                    precision_5=p5,
                    recall_5=r5,
                    precision_10=p10,
                    recall_10=r10,
                    backend=backend,
                )
                agree_scores = pr_scores
                last_flags[level][model_name] = flag_top_quantile(agree_scores, 0.05)
                all_rows.append(row)
                print(
                    f"P@5={p5:.3f} R@5={r5:.3f}  "
                    f"train={row['train_seconds']:.1f}s infer={row['infer_seconds']:.1f}s"
                    + (f" backend={backend}" if backend else "")
                )

    per_seed = rows_to_table(all_rows)
    agg = aggregate_seed_rows(all_rows)

    per_seed_path = out_dir / "table5_main_comparison.csv"
    agg_path = out_dir / "table5_main_comparison_agg.csv"
    per_seed.to_csv(per_seed_path, index=False)
    agg.to_csv(agg_path, index=False)

    md_path = out_dir / "table5_main_comparison.md"
    md_path.write_text(
        f"# Main comparison ({mode}, {regime})\n\n"
        "Precision/Recall use weak proxy stress labels (top 1% amplitude windows). "
        "Flag rates use a shared top-quantile rule on anomaly scores.\n\n"
        + markdown_table(agg)
        + "\n\n## Per-seed rows\n\n"
        + markdown_table(per_seed)
        + "\n",
        encoding="utf-8",
    )

    for level, fmap in last_flags.items():
        if not fmap:
            continue
        agree = pairwise_jaccard_table(fmap)
        agree_path = out_dir / f"table5_agreement_{level}.csv"
        agree.to_csv(agree_path)
        print(f"Wrote {agree_path}")

    protocol_path = out_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(f"\nWrote {per_seed_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {protocol_path}")
    print("\nAggregated table:")
    print(agg.to_string(index=False))
    return per_seed, agg


def main():
    parser = argparse.ArgumentParser(description="Run L1 vs L2 paper comparison.")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="strong",
        help="matched = 5-min L1 + shared sklearn forest; strong = original 1-min protocol",
    )
    parser.add_argument(
        "--regime",
        choices=list(REGIMES.keys()),
        default="2021_spring",
        help="Market window. 2022_bear is L1-only until LOB archives are added.",
    )
    parser.add_argument(
        "--out-tag",
        type=str,
        default=None,
        help="Optional results subdirectory name (default: <mode>_<regime>).",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated levels to run, e.g. L1 or L1,L2",
    )
    parser.add_argument("--max-l1-train", type=int, default=None)
    parser.add_argument("--max-l1-test", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds, e.g. 42,43,44",
    )
    args = parser.parse_args()
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    levels = None
    if args.levels:
        levels = [s.strip().upper() for s in args.levels.split(",") if s.strip()]
    run(
        mode=args.mode,
        max_l1_train=args.max_l1_train,
        max_l1_test=args.max_l1_test,
        seeds=seeds,
        regime=args.regime,
        out_tag=args.out_tag,
        levels=levels,
    )


if __name__ == "__main__":
    main()
