#!/usr/bin/env python3
"""Summarise correlation between ETH3D loss reduction and batch diversity.

Reads `trace.jsonl` from `train_with_eval/train.py`. Reports raw Pearson r and
**step-detrended** r: each delta_L is compared to a fitted baseline curve
delta_expected(step), so early-training steps (which shrink loss more) do not
dominate the correlation with diversity.

    python train_with_eval/analyze.py runs/my_run/trace.jsonl
    python train_with_eval/analyze.py runs/my_run/trace.jsonl --plot out.png
    python train_with_eval/analyze.py runs/my_run/trace.jsonl --detrend poly_log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_with_eval.batch_diversity import pearson_correlation  # noqa: E402


def load_trace(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def extract_delta(row: dict) -> float | None:
    value = row.get("delta_loss_eth3d", row.get("delta_loss_probe"))
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def fit_delta_baseline(
    steps: np.ndarray,
    deltas: np.ndarray,
    *,
    method: str,
    rolling_window: int,
) -> np.ndarray:
    """Expected delta_L at each step (same length as inputs, sorted by step)."""
    steps = np.asarray(steps, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    order = np.argsort(steps)
    steps = steps[order]
    deltas = deltas[order]

    n = len(deltas)
    if n == 0:
        return np.array([])
    if method == "none":
        return np.full(n, float(np.mean(deltas)))

    if method == "rolling":
        w = max(rolling_window, 1)
        out = np.empty(n)
        for i in range(n):
            lo = max(0, i - w // 2)
            hi = min(n, i + (w + 1) // 2)
            out[i] = float(np.median(deltas[lo:hi]))
        return out

    if method == "poly":
        x = (steps - steps.min()) / max(steps.max() - steps.min(), 1.0)
        deg = min(3, n - 1)
        coef = np.polyfit(x, deltas, deg)
        return np.polyval(coef, x)

    if method == "poly_log":
        x = np.log(steps + 1.0)
        deg = min(3, n - 1)
        coef = np.polyfit(x, deltas, deg)
        return np.polyval(coef, x)

    raise ValueError(f"unknown detrend method {method!r}")


def detrend_deltas(
    steps: np.ndarray,
    deltas: np.ndarray,
    *,
    method: str,
    rolling_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps_sorted, delta_expected, delta_residual) aligned arrays."""
    steps = np.asarray(steps, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    order = np.argsort(steps)
    steps = steps[order]
    deltas = deltas[order]
    expected = fit_delta_baseline(steps, deltas, method=method, rolling_window=rolling_window)
    residual = deltas - expected
    return steps, expected, residual


def regression_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(y_hat)
    y, y_hat = y[mask], y_hat[mask]
    if len(y) < 2:
        return float("nan")
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", type=Path, help="path to trace.jsonl")
    p.add_argument("--plot", type=Path, default=None, help="optional scatter plot path")
    p.add_argument("--min-step", type=int, default=0, help="ignore records before this step")
    p.add_argument(
        "--detrend",
        default="poly_log",
        choices=("poly_log", "poly", "rolling", "none"),
        help="baseline curve for delta_L vs step (default: poly_log)",
    )
    p.add_argument(
        "--rolling-window",
        type=int,
        default=7,
        help="window for --detrend rolling (traced steps, not optimizer steps)",
    )
    args = p.parse_args()

    rows = [r for r in load_trace(args.trace) if r.get("step", 0) >= args.min_step]
    records = [r for r in rows if extract_delta(r) is not None]
    if not records:
        raise SystemExit(f"no usable records in {args.trace}")

    steps = np.array([r["step"] for r in records], dtype=np.float64)
    deltas = np.array([extract_delta(r) for r in records], dtype=np.float64)
    cos = np.array([r.get("avg_cos_sim", float("nan")) for r in records], dtype=np.float64)
    div = np.array([r.get("diversity", float("nan")) for r in records], dtype=np.float64)

    n = len(deltas)
    print(f"{args.trace}: {n} traced steps")
    print(f"  delta_loss_eth3d: mean {np.nanmean(deltas):+.5f}  std {np.nanstd(deltas):.5f}")

    steps_s, expected, residual = detrend_deltas(
        steps,
        deltas,
        method=args.detrend,
        rolling_window=args.rolling_window,
    )
    r2 = regression_r2(deltas[np.argsort(steps)], expected)
    print(f"  baseline fit ({args.detrend}): R²={r2:.3f}  "
          f"expected delta mean {np.mean(expected):+.5f}")

    has_div = np.isfinite(cos).any()
    if has_div:
        r_cos_raw = pearson_correlation(deltas.tolist(), cos.tolist())
        r_div_raw = pearson_correlation(deltas.tolist(), div.tolist())
        r_cos_adj = pearson_correlation(residual.tolist(), cos[np.argsort(steps)].tolist())
        r_div_adj = pearson_correlation(residual.tolist(), div[np.argsort(steps)].tolist())

        print(f"  pearson(delta_L, avg_cos_sim):           raw {r_cos_raw:+.4f}")
        print(f"  pearson(delta_L, diversity):             raw {r_div_raw:+.4f}")
        print(f"  pearson(delta_L_residual, avg_cos_sim): detrended {r_cos_adj:+.4f}")
        print(f"  pearson(delta_L_residual, diversity):   detrended {r_div_adj:+.4f}")
        print("  avg_cos_sim = mean pairwise cos sim across scenes in the batch (not within-scene views)")
    else:
        print("  (no diversity columns -- run without --no-diversity)")

    if args.plot is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("matplotlib required for --plot") from exc

        sort_idx = np.argsort(steps)
        cos_s = cos[sort_idx]
        div_s = div[sort_idx]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

        ax = axes[0]
        ax.scatter(steps_s, deltas[sort_idx], alpha=0.45, s=14, label="observed")
        ax.plot(steps_s, expected, color="C1", lw=2, label=f"baseline ({args.detrend})")
        ax.set_xlabel("training step")
        ax.set_ylabel("delta_loss_eth3d")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.legend(fontsize=8)
        ax.set_title(f"baseline R²={r2:.3f}")

        mask = np.isfinite(cos_s)
        ax = axes[1]
        ax.scatter(cos_s[mask], deltas[sort_idx][mask], alpha=0.45, s=14)
        ax.set_xlabel("avg cross-scene cos sim (patch layer 30)")
        ax.set_ylabel("delta_loss_eth3d (raw)")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"raw r={pearson_correlation(deltas[mask].tolist(), cos_s[mask].tolist()):+.3f}")

        ax = axes[2]
        ax.scatter(cos_s[mask], residual[mask], alpha=0.45, s=14, color="C2")
        ax.set_xlabel("avg cross-scene cos sim (patch layer 30)")
        ax.set_ylabel("delta_L - baseline(step)")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"detrended r={pearson_correlation(residual[mask].tolist(), cos_s[mask].tolist()):+.3f}")

        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"  wrote {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
