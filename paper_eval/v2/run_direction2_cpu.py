"""Run all Direction-2 steps that are fine on CPU (no deep retrain)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run(mod: str) -> None:
    print("\n" + "=" * 70)
    print("RUNNING", mod)
    print("=" * 70)
    subprocess.check_call([sys.executable, "-m", mod], cwd=REPO)


def main() -> None:
    # Tasks already usually present; re-run for a clean Colab/local pass.
    mods = [
        "paper_eval.v2.run_task1_prevalence",
        "paper_eval.v2.run_task2_common_labels",
        "paper_eval.v2.run_task3_events",
        "paper_eval.v2.run_task4_svm_ga",
        "paper_eval.v2.run_task5_ood_splits",
        "paper_eval.v2.run_task6_agreement",
        "paper_eval.v2.run_task7_controls",
        "paper_eval.v2.run_task8_feature_groups",
        "paper_eval.v2.run_task9_baselines",
        "paper_eval.v2.run_task10_economic",
    ]
    for m in mods:
        run(m)
    print("\nCPU Direction-2 suite finished. See results_v2/ and docs/DIRECTION2_COLAB.md")


if __name__ == "__main__":
    main()
