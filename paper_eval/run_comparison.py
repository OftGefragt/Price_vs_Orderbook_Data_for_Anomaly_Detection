"""Train and score all model families on Level-1 and Level-2 data.

Writes:
  results/table5_main_comparison.csv
  results/table5_main_comparison.md
  results/scores_<model>_<level>.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_eval.data import load_level1, load_level2, scale_fit_transform
from paper_eval.metrics import (
    flag_top_quantile,
    markdown_table,
    rows_to_table,
    summarize_run,
)
from paper_eval.models_anogan import FanoganDetector
from paper_eval.models_eif import EIFSVMDetector
from paper_eval.models_gru_svdd import GRUDeepSVDDDetector


def _subsample(x: np.ndarray, max_rows: int, seed: int = 42) -> np.ndarray:
    if len(x) <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=max_rows, replace=False))
    return x[idx]


def run(max_l1_train: int = 12000, max_l1_test: int = 8000, seed: int = 42) -> pd.DataFrame:
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Level-1 data...")
    l1_train_df, l1_test_df = load_level1()
    l1_train, l1_test, _ = scale_fit_transform(l1_train_df, l1_test_df)
    l1_train = _subsample(l1_train, max_l1_train, seed=seed)
    l1_test = _subsample(l1_test, max_l1_test, seed=seed + 1)

    print("Loading Level-2 data...")
    l2_train_df, l2_test_df = load_level2()
    l2_train, l2_test, _ = scale_fit_transform(l2_train_df, l2_test_df)

    jobs = [
        ("f-AnoGAN", "L1", l1_train, l1_test, FanoganDetector(epochs_gan=3, epochs_enc=3, seed=seed)),
        ("f-AnoGAN", "L2", l2_train, l2_test, FanoganDetector(epochs_gan=3, epochs_enc=3, seed=seed)),
        (
            "GA-EIF+SVM",
            "L1",
            l1_train,
            l1_test,
            EIFSVMDetector(extension_level=4, random_state=seed),
        ),
        (
            "GA-EIF+SVM",
            "L2",
            l2_train,
            l2_test,
            EIFSVMDetector(extension_level=26, random_state=seed),
        ),
        (
            "GRU-DeepSVDD",
            "L1",
            l1_train,
            l1_test,
            GRUDeepSVDDDetector(epochs=3, seed=seed),
        ),
        (
            "GRU-DeepSVDD",
            "L2",
            l2_train,
            l2_test,
            GRUDeepSVDDDetector(epochs=3, seed=seed),
        ),
    ]

    rows = []
    ref_flags = {}

    for model_name, level, x_tr, x_te, detector in jobs:
        print(f"\n=== {model_name} | {level} ===")
        detector.fit(x_tr)
        scores, infer_s = detector.score(x_te)
        score_path = out_dir / f"scores_{model_name.replace('+', '_').replace('-', '_').lower()}_{level.lower()}.npz"
        pd.DataFrame({"score": scores}).to_csv(score_path, index=False)
        print(f"Saved scores: {score_path} ({len(scores)} windows)")

        flags5 = flag_top_quantile(scores, 0.05)
        if model_name == "f-AnoGAN":
            ref_flags[level] = flags5

        row = summarize_run(
            model=model_name,
            level=level,
            scores=scores,
            train_seconds=getattr(detector, "train_seconds", 0.0),
            infer_seconds=infer_s,
            reference_flags_5=ref_flags.get(level),
        )
        if hasattr(detector, "_backend"):
            row["eif_backend"] = detector._backend
        rows.append(row)
        print(
            f"flag@5%={row['flag_rate_5pct']:.4f}  "
            f"flag@10%={row['flag_rate_10pct']:.4f}  "
            f"train={row['train_seconds']:.1f}s  infer={row['infer_seconds']:.1f}s"
        )

    table = rows_to_table(rows)
    csv_path = out_dir / "table5_main_comparison.csv"
    md_path = out_dir / "table5_main_comparison.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Table 5. Main comparison\n\n"
        "Flag rates use a shared top-quantile rule on each model's anomaly scores "
        "(top 5% and top 10%). Jaccard is computed against f-AnoGAN flags on the "
        "same data level after length alignment.\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    print(table.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description="Run L1 vs L2 paper comparison.")
    parser.add_argument("--max-l1-train", type=int, default=12000)
    parser.add_argument("--max-l1-test", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(max_l1_train=args.max_l1_train, max_l1_test=args.max_l1_test, seed=args.seed)


if __name__ == "__main__":
    main()
