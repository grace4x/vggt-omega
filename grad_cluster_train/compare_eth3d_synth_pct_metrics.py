#!/usr/bin/env python3
"""ETH3D metrics table: 0% / 50% / 80% / 100% synthetic training mixes.

Covisibility sampling, 8 frames, 3 repeats, seed 1234 — the native protocol
from `runs/fov_sweep.py`. Depth numbers are scale-shift aligned (`s * pred + t`
least squares on valid pixels). Pose numbers are pooled over every frame pair.
"""

from __future__ import annotations

import gc
import json
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dl3dv_dataset import DL3DVDataset  # noqa: E402
from training.evaluate import evaluate_checkpoint, load_checkpoint  # noqa: E402
from training.losses import VGGTOmegaLoss  # noqa: E402

OUT = ROOT / "runs/eth3d_synth_pct_metrics.png"
OUT_JSON = ROOT / "runs/eth3d_synth_pct_metrics.json"
NUM_FRAMES = 8
SAMPLING = "covisibility"
REPEATS = 3
SEED = 1234

RUNS = [
    (
        "0% synthetic",
        ROOT / "runs/dl3dv-only-4k/final.pt",
    ),
    (
        "50% synthetic",
        ROOT / "runs/dl3dv-2-megasynth-4-tartanair-4/final.pt",
    ),
    (
        "80% synthetic",
        ROOT / "runs/dl3dv-20-megasynth-40-hypersim-20-tartanair-20/final.pt",
    ),
    (
        "100% synthetic",
        ROOT / "runs/megasynth-2-hypersim-4-tartanair-4/final.pt",
    ),
]

# Headline depth + pose accuracies. `higher` is True for scores, False for errors.
METRICS = [
    ("δ<1.25†", "delta_1.25_ss", "{:.3f}", True),
    ("AUC@30", "auc_at_30", "{:.3f}", True),
    ("abs_rel†", "abs_rel_ss", "{:.3f}", False),
    ("RRA@5", "rra_at_5", "{:.3f}", True),
]

# White → green for "better within this row"; red is reserved for the worst cell.
BETTER_CMAP = LinearSegmentedColormap.from_list(
    "better", ["#f4f4f4", "#d4edda", "#7dce8a", "#2e8b57"]
)


def load_dataset() -> DL3DVDataset:
    return DL3DVDataset(
        "/home/grace/eth3d-eval",
        split="all",
        num_frames=NUM_FRAMES,
        sampling=SAMPLING,
        augment=False,
        seed=SEED,
        depth_root="/home/grace/eth3d-eval/depth",
        dense_only=True,
        name="eth3d",
    )


@torch.inference_mode()
def score(ckpt: Path, dataset: DL3DVDataset, device: torch.device) -> tuple[int, dict]:
    model, step, train_args = load_checkpoint(ckpt, None, device)
    print(f"  {ckpt.parent.name}  step {step}", flush=True)
    criterion = VGGTOmegaLoss(
        weight_camera=train_args.get("weight_camera", 5.0),
        weight_depth=train_args.get("weight_depth", 1.0),
        weight_point=train_args.get("weight_point", 0.5),
        weight_gradient=train_args.get("weight_gradient", 1.0),
        depth_kwargs={"alpha": train_args.get("conf_alpha", 0.2)},
    )
    args = Namespace(
        repeats=REPEATS, seed=SEED, batch_size=1, workers=0, eval_crop=None, progress=0
    )
    summary, _ = evaluate_checkpoint(model, dataset, criterion, device, args)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return step, summary


def cell_color(values: np.ndarray, higher: bool, index: int) -> str:
    """Per-row colour: best → green, rest → pale, worst of a spread → light red."""
    row = values if higher else -values
    span = float(row.max() - row.min())
    if span < 1e-12:
        return "#f4f4f4"
    t = float((row[index] - row.min()) / span)
    if t < 1e-9:
        return "#f8d7da"
    return matplotlib.colors.to_hex(BETTER_CMAP(Normalize(0.0, 1.0)(t)))


def draw_line(ax, values: np.ndarray, *, title: str, ylabel: str, fmt: str, higher: bool) -> None:
    xs = np.array([0, 50, 80, 100], dtype=np.float64)
    best = int(np.argmax(values) if higher else np.argmin(values))
    ax.plot(xs, values, color="#2c3e50", lw=1.8, zorder=2)
    for i, (x, v) in enumerate(zip(xs, values)):
        ax.scatter(
            [x],
            [v],
            s=46 if i == best else 36,
            zorder=3,
            color=cell_color(values, higher, i),
            edgecolors="#2e8b57" if i == best else "#2c3e50",
            linewidths=1.2,
        )
    ax.set_xticks(xs, [f"{int(x)}%" for x in xs])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("synthetic training mix", fontsize=9)
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.22, 0.02)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_xlim(-8, 108)
    ax.yaxis.grid(True, color="#e4e8ec", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c5cdd4")
    ax.spines["bottom"].set_color("#c5cdd4")
    ax.tick_params(labelsize=9, colors="#333333")
    for i, (x, v) in enumerate(zip(xs, values)):
        ax.annotate(
            fmt.format(v),
            (x, v),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold" if i == best else "normal",
            color="#145a32" if i == best else "#222222",
        )


def draw(summaries: list[dict]) -> None:
    n_cols = 1 + len(RUNS)
    n_rows = 1 + len(METRICS)
    fig = plt.figure(figsize=(11.2, 8.4), dpi=160)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1.0],
        left=0.07,
        right=0.97,
        top=0.90,
        bottom=0.12,
        hspace=0.38,
        wspace=0.28,
    )

    fig.suptitle(
        "ETH3D eval metrics  ·  synthetic vs real training mix",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    matrix = np.array(
        [[summaries[c][key] for c in range(len(RUNS))] for _, key, _, _ in METRICS],
        dtype=np.float64,
    )
    by_key = {key: matrix[i] for i, (_, key, _, _) in enumerate(METRICS)}

    draw_line(
        fig.add_subplot(gs[0, 0]),
        by_key["delta_1.25_ss"],
        title="δ<1.25†  (higher better)",
        ylabel="fraction of pixels within 25%",
        fmt="{:.3f}",
        higher=True,
    )
    draw_line(
        fig.add_subplot(gs[0, 1]),
        by_key["abs_rel_ss"],
        title="abs_rel†  (lower better)",
        ylabel="mean |pred − gt| / gt",
        fmt="{:.3f}",
        higher=False,
    )

    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_axis_off()
    ax.invert_yaxis()

    headers = ["metric"] + [name for name, _ in RUNS]
    for c, text in enumerate(headers):
        ax.fill_between([c, c + 1], 0, 1, color="#2c3e50", linewidth=0, zorder=0)
        ax.text(
            c + 0.5,
            0.5,
            text,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    for r, (label, _, fmt, higher) in enumerate(METRICS):
        y0, y1 = r + 1, r + 2
        ax.fill_between([0, 1], y0, y1, color="#eef2f5", linewidth=0, zorder=0)
        arrow = "↑" if higher else "↓"
        ax.text(0.08, y0 + 0.5, f"{label}  {arrow}", ha="left", va="center", fontsize=10)

        row = matrix[r]
        best = int(np.argmax(row) if higher else np.argmin(row))
        for c, value in enumerate(row):
            x0, x1 = c + 1, c + 2
            ax.fill_between([x0, x1], y0, y1, color=cell_color(row, higher, c), linewidth=0, zorder=0)
            ax.text(
                x0 + 0.5,
                y0 + 0.5,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold" if c == best else "normal",
                color="#145a32" if c == best else "#222222",
            )

    for x in range(n_cols + 1):
        ax.plot([x, x], [0, n_rows], color="#c5cdd4", lw=0.6, zorder=1)
    for y in range(n_rows + 1):
        ax.plot([0, n_cols], [y, y], color="#c5cdd4", lw=0.6, zorder=1)
    ax.plot([0, n_cols], [1, 1], color="#2c3e50", lw=1.1, zorder=2)

    fig.text(
        0.5,
        0.035,
        "ETH3D, 8 frames, covisibility sampling, 3 repeats, seed 1234, native FoV.  "
        "† = scale-shift aligned.  ↑ higher better, ↓ lower better; best cell bold.\n"
        "0%: dl3dv-only-4k    ·    "
        "50%: dl3dv-2-megasynth-4-tartanair-4    ·    "
        "80%: dl3dv-20-megasynth-40-hypersim-20-tartanair-20    ·    "
        "100%: megasynth-2-hypersim-4-tartanair-4",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#444444",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {OUT}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading ETH3D windows", flush=True)
    dataset = load_dataset()
    print(f"  {len(dataset)} scenes: {', '.join(e['scene'] for e in dataset.scenes)}", flush=True)

    summaries: list[dict] = []
    steps: list[int] = []
    for name, ckpt in RUNS:
        print(f"running {name}", flush=True)
        step, summary = score(ckpt, dataset, device)
        steps.append(step)
        summaries.append(summary)

    draw(summaries)

    header = f"{'metric':<12}" + "".join(f"{n:>16}" for n, _ in RUNS)
    print(header, flush=True)
    for label, key, fmt, _ in METRICS:
        row = f"{label:<12}" + "".join(fmt.format(summaries[c][key]).rjust(16) for c in range(len(RUNS)))
        print(row, flush=True)

    payload = {
        "num_frames": NUM_FRAMES,
        "sampling": SAMPLING,
        "seed": SEED,
        "repeats": REPEATS,
        "num_scenes": len(dataset),
        "results": {
            name: {
                "checkpoint": str(ckpt),
                "step": step,
                **{key: summary[key] for _, key, _, _ in METRICS},
            }
            for (name, ckpt), step, summary in zip(RUNS, steps, summaries)
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
