"""Task 10: economic / microstructure descriptors of flagged vs non-flagged events."""

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
from paper_eval.v2.common_labels import load_l2_with_midpoint, price_panel_from_series, ANOGAN_ETH
from paper_eval.v2.events import merge_windows_to_events
from paper_eval.v2.labels import top_fraction_labels, window_amplitudes
from paper_eval.v2.run_task5_ood_splits import scale_free_l1


OUT = REPO_ROOT / "results_v2" / "task10_economic"


def top_flags(scores: np.ndarray, frac: float = 0.05) -> np.ndarray:
    n = len(scores)
    k = max(1, int(round(n * frac)))
    order = np.lexsort((np.arange(n), -scores))[:k]
    f = np.zeros(n, dtype=bool)
    f[order] = True
    return f


def window_series_stats(df: pd.DataFrame, starts: np.ndarray, window: int = 30) -> pd.DataFrame:
    rows = []
    close = df["Close"].to_numpy(dtype=float) if "Close" in df.columns else df["midpoint"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns else np.full(len(df), np.nan)
    spread = df["spread"].to_numpy(dtype=float) if "spread" in df.columns else np.full(len(df), np.nan)
    for s in starts:
        seg_c = close[s : s + window]
        if np.all(seg_c > 0):
            rets = np.diff(np.log(seg_c))
        else:
            rets = np.diff(seg_c)
        rv = float(np.sqrt(np.sum(rets**2)))
        abs_ret = float(np.abs(seg_c[-1] / seg_c[0] - 1.0)) if seg_c[0] != 0 else float("nan")
        vsum = float(np.nansum(vol[s : s + window]))
        impact = abs_ret / vsum if vsum and np.isfinite(vsum) and vsum > 0 else float("nan")
        rows.append(
            {
                "start": int(s),
                "realized_vol": rv,
                "abs_return": abs_ret,
                "volume_sum": vsum,
                "abs_ret_per_volume": impact,
                "mean_spread": float(np.nanmean(spread[s : s + window])),
            }
        )
    return pd.DataFrame(rows)


def summarize_flagged(stats: pd.DataFrame, flags: np.ndarray, label: str, level: str) -> dict:
    # event-level: mark windows covered by flagged events
    cover = np.zeros(len(flags), dtype=bool)
    for a, b in merge_windows_to_events(flags, gap=30):
        cover[a : b + 1] = True
    # compare window starts
    flagged = stats.loc[cover[: len(stats)]].copy()
    other = stats.loc[~cover[: len(stats)]].copy()

    def boot_diff(col, n_boot=400, seed=42):
        rng = np.random.default_rng(seed)
        a = flagged[col].dropna().to_numpy()
        b = other[col].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            return float("nan"), float("nan"), float("nan"), float("nan")
        diffs = []
        for _ in range(n_boot):
            aa = rng.choice(a, size=len(a), replace=True)
            bb = rng.choice(b, size=len(b), replace=True)
            diffs.append(float(np.median(aa) - np.median(bb)))
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        med_diff = float(np.median(a) - np.median(b))
        # simple effect size: median diff / pooled MAD
        mad = float(np.median(np.abs(np.concatenate([a, b]) - np.median(np.concatenate([a, b]))))) + 1e-12
        return med_diff, float(lo), float(hi), med_diff / mad

    out = {"level": level, "detector": label, "n_flag_windows": int(cover.sum()), "n_other": int((~cover).sum())}
    for col, circular in [
        ("realized_vol", False),
        ("abs_return", False),
        ("volume_sum", False),
        ("abs_ret_per_volume", False),
        ("mean_spread", True),  # L2 input-related when from AnoGAN
    ]:
        med_f = float(flagged[col].median()) if len(flagged) else float("nan")
        med_o = float(other[col].median()) if len(other) else float("nan")
        d, lo, hi, eff = boot_diff(col)
        out[f"{col}_flag_median"] = med_f
        out[f"{col}_other_median"] = med_o
        out[f"{col}_median_diff"] = d
        out[f"{col}_diff_ci95_lo"] = lo
        out[f"{col}_diff_ci95_hi"] = hi
        out[f"{col}_effect_size_approx"] = eff
        out[f"{col}_circular_descriptive_only"] = bool(circular and level.startswith("L2"))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed = 42
    window = 30

    # L1 scale-free forest flags
    l1tr, l1te = load_level1("5min")
    sf_tr, sf_te = scale_free_l1(l1tr), scale_free_l1(l1te)
    sc = MinMaxScaler()
    xtr = sc.fit_transform(sf_tr.values)
    xte = sc.transform(sf_te.values)
    clf = IsolationForest(
        n_estimators=235,
        max_samples=min(1691, len(xtr)),
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(xtr)
    s1 = sliding_windows((-clf.score_samples(xte)).reshape(-1, 1), window, 1)[:, :, 0].max(1)
    f1 = top_flags(s1, 0.05)
    stats1 = window_series_stats(l1te.reset_index(drop=True).assign(Close=l1te["Close"].to_numpy(), Volume=l1te["Volume"].to_numpy()), np.arange(len(f1)), window)

    # L2 forest flags + AnoGAN spread/mid for descriptors
    l2tr, l2te = load_level2()
    an = pd.read_csv(ANOGAN_ETH)
    sc2 = MinMaxScaler()
    x2tr = sc2.fit_transform(l2tr.values)
    x2te = sc2.transform(l2te.values)
    clf2 = IsolationForest(
        n_estimators=235,
        max_samples=min(1691, len(x2tr)),
        contamination=0.062,
        random_state=seed,
        n_jobs=-1,
    )
    clf2.fit(x2tr)
    s2 = sliding_windows((-clf2.score_samples(x2te)).reshape(-1, 1), window, 1)[:, :, 0].max(1)
    f2 = top_flags(s2, 0.05)
    l2_panel = pd.DataFrame(
        {
            "midpoint": an["midpoint"].astype(float),
            "Close": an["midpoint"].astype(float),
            "Volume": (an["buys"].astype(float) + an["sells"].astype(float)),
            "spread": an["spread"].astype(float),
        }
    )
    stats2 = window_series_stats(l2_panel, np.arange(len(f2)), window)

    # Also label-defined positives as reference
    lab1, _ = top_fraction_labels(
        window_amplitudes(price_panel_from_series(l1te["Close"].to_numpy()), "close_range"), 0.01
    )
    lab2, _ = top_fraction_labels(
        window_amplitudes(price_panel_from_series(an["midpoint"].to_numpy()), "close_range"), 0.01
    )

    rows = [
        summarize_flagged(stats1, f1, "Forest_scaleFree", "L1"),
        summarize_flagged(stats1, lab1, "PrimaryLabel", "L1"),
        summarize_flagged(stats2, f2, "Forest_L2", "L2"),
        summarize_flagged(stats2, lab2, "PrimaryLabel", "L2"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "flagged_vs_other_economics_seed42.csv", index=False)
    config = {
        "task": 10,
        "seed": 42,
        "note": (
            "spread on L2 is descriptive-only / partly circular as related to book state. "
            "realized_vol/abs_return/volume are the preferred non-label economic checks."
        ),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "SUMMARY.md").write_text(
        "# Task 10 — Economic descriptors\n\n" + df.to_string(index=False) + "\n",
        encoding="utf-8",
    )
    print(df[["level", "detector", "realized_vol_median_diff", "volume_sum_median_diff", "abs_ret_per_volume_median_diff"]].to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
