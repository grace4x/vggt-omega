#!/usr/bin/env python3
"""Re-score the DL3DV/MegaSynth/Hypersim mixture runs at a ladder of eval FoVs.

The question this answers: is `dl3dv-only-4k`'s depth advantage on ETH3D real
geometry, or is it a field-of-view prior it inherited from its training mix?

The three train sets do not overlap in camera intrinsics at all --

    set         hfov            fx/W
    dl3dv       93.9 +- 6.6     0.470 +- 0.063
    megasynth   57.4 +- 6.9     0.929 +- 0.135
    hypersim    60.0 +- 0.00    0.866 +- 0.000
    eth3d-eval  83.0 +- 0.16    0.565 +- 0.002

-- so at ETH3D's native 83 deg the all-DL3DV model is the only one being scored
inside the camera regime it was trained on. Every mixture spent 25-50%% of a
fixed 40k-step budget on a lens ~2x narrower than the test set's.

`--eval-crop` narrows the eval frustum by cropping around the principal point,
which keeps the focals and moves only (H, W) and cx, cy -- exactly what a
narrower lens on the same camera would have seen. So the ladder below walks the
eval set from DL3DV's regime down into the synthetic sets' regime without
resampling a pixel:

    crop 256x384  hfov 83.1  fx/W 0.564   <- native; closest to dl3dv (0.470)
    crop 192x320  hfov 72.9  fx/W 0.677
    crop 160x288  hfov 67.3  fx/W 0.752
    crop 144x256  hfov 61.2  fx/W 0.846   <- matches hypersim  (0.866)
    crop 128x224  hfov 54.7  fx/W 0.966   <- matches megasynth (0.929)

Read the output as a *trend*, not as absolute numbers. Cropping also throws away
context and patch tokens, so every checkpoint gets worse down the ladder; what
matters is whether the *gap* to `dl3dv-only-4k` closes. Two readings:

* Gap closes or reverses  -> the advantage was the FoV prior. Synthetic data is
  not the problem, the un-jittered intrinsics mismatch is, and the fix is to
  retrain the mixtures with `--fov-jitter` rather than to drop synthetic data.
* Gap survives at matched FoV -> synthetic genuinely is not helping this model
  at this budget, and the next suspect is the eval itself (ETH3D is outdoor
  architectural; megasynth has no semantics and hypersim is indoor-only).

Note `dl3dv-megasynth-4k-50` is confounded -- it is the only run with a frozen
`--scene-list` half of the scene pool -- so it is scored here but excluded from
the headline comparison. `dl3dv-megasynth-4k-eqmix` is the honest 50/50.

    python runs/fov_sweep.py                 # run the sweep, then report
    python runs/fov_sweep.py --report-only   # re-report from cached summaries
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
DATA_ROOT = Path("/home/grace/eth3d-eval")

BASELINE = "dl3dv-only-4k"
RUNS = [
    BASELINE,
    "dl3dv-megasynth-4k-eqmix",
    "dl3dv-2-megasynth-4-hypersim-4",
    "dl3dv-megasynth-4k-50",  # confounded by --scene-list; reported, not compared
]
CONFOUNDED = {"dl3dv-megasynth-4k-50"}

# (crop_h, crop_w, note). None = native, no crop.
LADDER = [
    (None, None, "native 83.1 deg  ~dl3dv side"),
    (192, 320, "72.9 deg"),
    (160, 288, "67.3 deg"),
    (144, 256, "61.2 deg  ~hypersim"),
    (128, 224, "54.7 deg  ~megasynth"),
]

# (key, label, lower_is_better)
METRICS = [
    ("abs_rel_aligned", "abs_rel*", True),
    ("delta_1.25_aligned", "d<1.25*", False),
    ("rot_err_deg_median_window", "rot_med", True),
    ("trans_err_deg_median_window", "trans_med", True),
]


def crop_dir(out_root: Path, crop_h, crop_w) -> Path:
    return out_root / ("native" if crop_h is None else f"crop-{crop_h}x{crop_w}")


def run_one(out_root: Path, crop_h, crop_w, args) -> None:
    """One evaluate.py pass over all checkpoints at a single crop."""
    target = crop_dir(out_root, crop_h, crop_w)
    cmd = [
        str(PYTHON), str(REPO / "training" / "evaluate.py"),
        "--data-root", str(DATA_ROOT),
        "--depth-root", str(DATA_ROOT / "depth"),
        "--split", "all", "--dense-only",
        "--sampling", "covisibility",
        "--num-frames", str(args.num_frames),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--progress", "0",
        "--out", str(target),
        "--checkpoint", *[str(REPO / "runs" / r / "final.pt") for r in RUNS],
    ]
    if crop_h is not None:
        cmd += ["--eval-crop", str(crop_h), str(crop_w)]
    print(f"\n{'=' * 100}\n[sweep] {target.name}\n{'=' * 100}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def per_scene(records: list[dict], key: str) -> dict[str, float]:
    """Mean over a scene's repeats, so the unit of comparison is the 13 scenes.

    Pooling windows instead would treat three re-sampled windows of one scene as
    three independent observations, which they are not -- they share the scene's
    geometry, its exposure and its depth GT.
    """
    grouped: dict[str, list[float]] = {}
    for row in records:
        value = row.get(key)
        if value is None or value != value:  # absent, or nan from a depth-less window
            continue
        grouped.setdefault(row["scene_id"], []).append(float(value))
    return {scene: statistics.mean(v) for scene, v in grouped.items()}


def load(out_root: Path, crop_h, crop_w) -> dict[str, list[dict]]:
    target = crop_dir(out_root, crop_h, crop_w)
    windows = {}
    for name in RUNS:
        path = target / f"windows-{name}-final.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path} -- run without --report-only first")
        windows[name] = [json.loads(line) for line in path.open()]
    return windows


def paired(base: dict[str, float], other: dict[str, float], seed: int = 0):
    """Mean paired delta over the shared scenes, plus a bootstrap sign-flip rate.

    The sign-flip rate is the fraction of scene-level bootstrap resamples whose
    mean delta lands on the opposite side of zero from the observed one -- a
    two-sided "could this be noise" number. With 13 scenes anything above ~0.1
    is not worth interpreting.
    """
    scenes = sorted(set(base) & set(other))
    if not scenes:
        return None, None, 0
    diffs = [other[s] - base[s] for s in scenes]
    observed = statistics.mean(diffs)
    rng = random.Random(seed)
    flips = 0
    for _ in range(20_000):
        resample = statistics.mean(rng.choice(diffs) for _ in scenes)
        if (resample > 0) != (observed > 0):
            flips += 1
    return observed, flips / 20_000, len(scenes)


def report(out_root: Path) -> None:
    for key, label, lower_better in METRICS:
        direction = "lower is better" if lower_better else "higher is better"
        print(f"\n\n### {label}  ({key}, {direction})")
        print(f"\n{'eval crop':>12}  {'hfov note':<26} " + "  ".join(f"{r[:26]:>26}" for r in RUNS))
        print("-" * (12 + 2 + 26 + 1 + 28 * len(RUNS)))
        rows = []
        for crop_h, crop_w, note in LADDER:
            windows = load(out_root, crop_h, crop_w)
            scored = {name: per_scene(windows[name], key) for name in RUNS}
            means = {name: statistics.mean(scored[name].values()) for name in RUNS}
            crop = "native" if crop_h is None else f"{crop_h}x{crop_w}"
            print(f"{crop:>12}  {note:<26} " + "  ".join(f"{means[r]:>26.4f}" for r in RUNS))
            rows.append((crop, note, scored))

        print(f"\n  paired delta vs {BASELINE}, per-scene, [bootstrap sign-flip rate]")
        better = "closes" if lower_better else "closes"
        print(f"  a delta that {better} towards 0 down the ladder = the gap was a field-of-view prior")
        compare = [r for r in RUNS if r != BASELINE and r not in CONFOUNDED]
        print(f"\n{'eval crop':>12}  " + "  ".join(f"{r[:30]:>30}" for r in compare))
        print("-" * (12 + 2 + 32 * len(compare)))
        for crop, note, scored in rows:
            cells = []
            for name in compare:
                delta, flip, n = paired(scored[BASELINE], scored[name])
                cells.append(f"{delta:+10.4f} [{flip:.3f}] n={n:2d}" if delta is not None else " " * 22)
            print(f"{crop:>12}  " + "  ".join(f"{c:>30}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "eval-fov-sweep")
    parser.add_argument("--num-frames", type=int, default=8, help="match the original eval-eth3d pass")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report-only", action="store_true", help="skip the evals, re-report cached summaries")
    args = parser.parse_args()

    if not args.report_only:
        for crop_h, crop_w, _ in LADDER:
            run_one(args.out, crop_h, crop_w, args)

    report(args.out)
    print(f"\nsummaries under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
