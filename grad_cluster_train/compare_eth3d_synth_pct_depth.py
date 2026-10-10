#!/usr/bin/env python3
"""ETH3D depth-map table: 0% / 50% / 80% / 100% synthetic training mixes.

Predictions are scale-shift aligned to GT (`s * pred + t` least squares on
valid pixels), matching `training.losses.depth_metrics(..., align_scale_shift=True)`.
Windows match the ETH3D eval protocol (8 frames, window sampling, seed 1234).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dl3dv_dataset import DL3DVDataset, collate_scenes  # noqa: E402
from training.evaluate import load_checkpoint  # noqa: E402
from training.losses import depth_metrics  # noqa: E402

OUT = ROOT / "runs/eth3d_synth_pct_depth.png"
NUM_FRAMES = 8
FRAME = 0
CMAP = "turbo"

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


def colorize(depth: np.ndarray, mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    norm = np.zeros_like(depth, dtype=np.float32)
    span = max(vmax - vmin, 1e-8)
    norm[mask] = np.clip((depth[mask] - vmin) / span, 0.0, 1.0)
    rgb = plt.get_cmap(CMAP)(norm)[..., :3]
    rgb[~mask] = 0.18
    return rgb


def to_rgb(chw: np.ndarray) -> np.ndarray:
    return np.clip(chw.transpose(1, 2, 0), 0, 1)


def load_windows() -> list[dict]:
    dataset = DL3DVDataset(
        "/home/grace/eth3d-eval",
        split="all",
        num_frames=NUM_FRAMES,
        sampling="window",
        window_size=100,
        augment=False,
        seed=1234,
        depth_root="/home/grace/eth3d-eval/depth",
        dense_only=True,
        name="eth3d",
    )
    windows = []
    for i, entry in enumerate(dataset.scenes):
        sample = dataset[i]
        batch = collate_scenes([sample])
        windows.append(
            {
                "label": entry["scene"],
                "images": batch["images"],
                "gt": batch["depth"][0].numpy(),
                "mask": batch["depth_mask"][0].numpy().astype(bool),
                "rgb": batch["images"][0].numpy(),
            }
        )
    return windows


@torch.inference_mode()
def predict(ckpt: Path, windows: list[dict], device: torch.device) -> tuple[list[np.ndarray], list[float]]:
    model, step, _ = load_checkpoint(ckpt, None, device)
    print(f"  {ckpt.parent.name}  step {step}", flush=True)
    preds, abs_rels = [], []
    for w in windows:
        images = w["images"].to(device)
        out = model(images)["depth"].float().cpu()
        if out.dim() == 5:
            out = out.squeeze(-1)
        pred = out[0].numpy()
        gt = torch.from_numpy(w["gt"])[None]
        mask = torch.from_numpy(w["mask"])[None]
        ss = depth_metrics(out, gt, mask, align_scale_shift=True)
        aligned = np.clip(ss["ss_scale"] * pred + ss["ss_shift"], 1e-6, None)
        preds.append(aligned)
        abs_rels.append(float(ss["abs_rel"]))
        del out
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds, abs_rels


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading ETH3D windows", flush=True)
    windows = load_windows()
    print(f"  {len(windows)} scenes: {', '.join(w['label'] for w in windows)}", flush=True)

    run_preds: list[list[np.ndarray]] = []
    run_absrel: list[list[float]] = []
    for name, ckpt in RUNS:
        print(f"running {name}", flush=True)
        preds, abs_rels = predict(ckpt, windows, device)
        run_preds.append(preds)
        run_absrel.append(abs_rels)

    means = [sum(vals) / len(vals) for vals in run_absrel]
    n_rows, n_cols = len(windows), 2 + len(RUNS)
    # Extra left column for scene names, extra right column for the colorbar.
    fig = plt.figure(figsize=(2.15 * n_cols + 1.6, 1.42 * n_rows + 1.05), dpi=150)
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 2,
        width_ratios=[0.55] + [1.0] * n_cols + [0.06],
        left=0.02,
        right=0.98,
        top=0.93,
        bottom=0.045,
        wspace=0.04,
        hspace=0.08,
    )
    fig.suptitle(
        "ETH3D eval depth maps  ·  synthetic vs real training mix",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    col_titles = ["RGB", "GT"] + [
        f"{name}\nmean abs_rel† {mean:.3f}" for (name, _), mean in zip(RUNS, means)
    ]

    for r, w in enumerate(windows):
        gt, mask = w["gt"][FRAME], w["mask"][FRAME]
        valid = mask & np.isfinite(gt) & (gt > 1e-6)
        vmin, vmax = np.percentile(gt[valid], [2, 98]) if valid.any() else (0.0, 1.0)
        cells = [to_rgb(w["rgb"][FRAME]), colorize(gt, valid, vmin, vmax)]
        for c in range(len(RUNS)):
            pred = run_preds[c][r][FRAME]
            cells.append(colorize(pred, np.ones_like(valid), vmin, vmax))

        lab = fig.add_subplot(gs[r, 0])
        lab.axis("off")
        lab.text(1.0, 0.5, w["label"].replace("_", " "), ha="right", va="center", fontsize=9)

        for c, img in enumerate(cells):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(img, interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, pad=6)
            if c >= 2:
                ax.text(
                    0.04,
                    0.07,
                    f"abs_rel† {run_absrel[c - 2][r]:.3f}",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="white",
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.6, "linewidth": 0},
                )

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
        cax=cax,
    )
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["near", "far"])
    cb.ax.tick_params(labelsize=8)
    cb.ax.set_ylabel("depth (per-row GT range)", fontsize=8, labelpad=6)

    fig.text(
        0.5,
        0.012,
        "Same 8-frame ETH3D windows (seed 1234, sampling=window). Predictions scale-shift aligned to GT. "
        "abs_rel† is the full-window score.  "
        "0%: dl3dv-only-4k   ·   "
        "50%: dl3dv-2-megasynth-4-tartanair-4   ·   "
        "80%: dl3dv-20-megasynth-40-hypersim-20-tartanair-20   ·   "
        "100%: megasynth-2-hypersim-4-tartanair-4",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {OUT}", flush=True)
    print("window abs_rel† (scale-shift):", flush=True)
    header = f"{'scene':<16}" + "".join(f"{n:>16}" for n, _ in RUNS)
    print(header)
    for r, w in enumerate(windows):
        row = f"{w['label']:<16}" + "".join(f"{run_absrel[c][r]:>16.3f}" for c in range(len(RUNS)))
        print(row)
    means = [sum(vals) / len(vals) for vals in run_absrel]
    print(f"{'mean':<16}" + "".join(f"{m:>16.3f}" for m in means))


if __name__ == "__main__":
    main()
