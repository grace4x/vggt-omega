#!/usr/bin/env python3
"""Paired comparison of two training arms over seeds and eval scenes.

Each run directory holds `eval-<set>/windows-*.jsonl` from `eval_suite.sh`.
Runs are paired by their `args.json` seed, windows by (scene, repeat) -- the
eval seed is fixed, so both arms are scored on identical windows.  For every
set the per-scene difference (treatment - baseline, averaged over repeats) is
taken inside each seed pair; the primary number is the equal-weight mean of the
per-set mean differences over `--primary-sets`.

The 95% interval is a two-level bootstrap: seed pairs and scenes are resampled
together, so it covers both training noise and which scenes made the eval set.
With 3 seed pairs the seed level is coarse -- read it alongside the per-pair
numbers, which should agree in sign before calling a difference real.

    python grad_cluster_train/compare_arms.py \\
        --baseline runs/uniform-s0 runs/uniform-s1 runs/uniform-s2 \\
        --treatment runs/cb0-k40-s0 runs/cb0-k40-s1 runs/cb0-k40-s2
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

SETS = ("eth3d", "dl3dv", "scannet", "hypersim", "tartanair")
# TartanAir is left out of the primary number: DL3DV-only models are at the floor
# there (delta_1.25_ss ~0.19), so it measures failure, not sampling quality.
PRIMARY = ("eth3d", "dl3dv", "scannet", "hypersim")


def load_windows(run: Path, name: str, metric: str) -> dict[str, float]:
    """scene -> metric averaged over that scene's repeats."""
    files = glob.glob(str(run / f"eval-{name}" / "windows-*.jsonl"))
    if len(files) != 1:
        raise SystemExit(f"expected one windows file in {run}/eval-{name}, found {len(files)}")
    per_scene: dict[str, list[float]] = {}
    seen = set()
    for line in open(files[0]):
        w = json.loads(line)
        key = (w["scene_id"], w["repeat"])
        if key in seen:
            raise SystemExit(f"duplicate window {key} in {files[0]}")
        seen.add(key)
        if metric in w and w[metric] is not None and np.isfinite(w[metric]):
            per_scene.setdefault(w["scene_id"], []).append(float(w[metric]))
    return {s: float(np.mean(v)) for s, v in per_scene.items()}


def run_seed(run: Path) -> int:
    return int(json.load(open(run / "args.json"))["seed"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, nargs="+", required=True)
    p.add_argument("--treatment", type=Path, nargs="+", required=True)
    p.add_argument("--metric", default="delta_1.25_ss")
    p.add_argument("--sets", nargs="+", default=list(SETS))
    p.add_argument("--primary-sets", nargs="+", default=list(PRIMARY))
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    base = {run_seed(r): r for r in args.baseline}
    treat = {run_seed(r): r for r in args.treatment}
    seeds = sorted(set(base) & set(treat))
    unpaired = sorted(set(base) ^ set(treat))
    if not seeds:
        raise SystemExit("no seed appears in both arms")
    if unpaired:
        print(f"ignoring unpaired seeds {unpaired}")

    # diffs[set] = (n_pairs, n_scenes) scene-level treatment - baseline
    diffs: dict[str, np.ndarray] = {}
    means: dict[str, tuple[float, float]] = {}
    for name in args.sets:
        b = [load_windows(base[s], name, args.metric) for s in seeds]
        t = [load_windows(treat[s], name, args.metric) for s in seeds]
        scenes = sorted(set.intersection(*(set(x) for x in b + t)))
        if not scenes:
            raise SystemExit(f"{name}: no scene scored in every run")
        B = np.array([[x[sc] for sc in scenes] for x in b])
        T = np.array([[x[sc] for sc in scenes] for x in t])
        diffs[name] = T - B
        means[name] = (float(B.mean()), float(T.mean()))

    rng = np.random.default_rng(0)
    n_pairs = len(seeds)
    boot = {name: np.empty(args.bootstrap) for name in args.sets}
    for i in range(args.bootstrap):
        pick = rng.integers(0, n_pairs, n_pairs)
        for name, d in diffs.items():
            cols = rng.integers(0, d.shape[1], d.shape[1])
            boot[name][i] = d[pick][:, cols].mean()
    primary = [n for n in args.primary_sets if n in diffs]
    boot_primary = np.mean([boot[n] for n in primary], axis=0)

    print(f"metric {args.metric}   treatment - baseline   seed pairs {seeds}\n")
    print(f"{'set':10s} {'scenes':>6s} {'baseline':>9s} {'treatment':>9s} {'diff':>8s} "
          f"{'95% CI':>19s} {'scenes t>b':>10s}  per-seed-pair diffs")
    for name, d in diffs.items():
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        scene_mean = d.mean(axis=0)
        per_pair = " ".join(f"{x:+.4f}" for x in d.mean(axis=1))
        tag = "" if name in primary else "  (not in primary)"
        print(f"{name:10s} {d.shape[1]:6d} {means[name][0]:9.4f} {means[name][1]:9.4f} {d.mean():+8.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] {int((scene_mean > 0).sum()):>4d}/{d.shape[1]:<5d}  {per_pair}{tag}")

    point = float(np.mean([diffs[n].mean() for n in primary]))
    lo, hi = np.percentile(boot_primary, [2.5, 97.5])
    per_pair = [float(np.mean([diffs[n][k].mean() for n in primary])) for k in range(n_pairs)]
    print(f"\nPRIMARY (equal-weight mean over {', '.join(primary)}): {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("per seed pair: " + "  ".join(f"s{s}={x:+.4f}" for s, x in zip(seeds, per_pair)))
    same_sign = all(x > 0 for x in per_pair) or all(x < 0 for x in per_pair)
    excludes_zero = lo > 0 or hi < 0
    print(f"CI excludes 0: {excludes_zero}   all seed pairs agree in sign: {same_sign}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
