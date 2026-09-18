#!/usr/bin/env python3
"""One comparison PNG: default / random_half / layer10 / layer30.

Predictions are scale-shift aligned to GT (`s * pred + t` least squares on
valid pixels), matching `training.losses.depth_metrics(..., align_scale_shift=True)`.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dl3dv_dataset import DL3DVDataset, collate_scenes  # noqa: E402
from training.evaluate import load_checkpoint  # noqa: E402
from training.losses import depth_metrics  # noqa: E402

OUT = ROOT / "runs/depth_compare.png"
NUM_FRAMES = 8  # pipes only has 14 frames
FRAME = 0

RUNS = [
    ("default", ROOT / "runs/default/final.pt"),
    ("random half", ROOT / "runs/random_half/final.pt"),
    ("layer 10", ROOT / "runs/patch_avg_layer10/final.pt"),
    ("layer 30", ROOT / "runs/patch_avg_layer30/final.pt"),
]

SCENES = [
    ("eth3d", "courtyard"),
    ("eth3d", "playground"),
    ("eth3d", "office"),
    ("eth3d", "pipes"),
    ("scannet", "scene0081_00"),
]


def colorize(depth: np.ndarray, mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    norm = np.zeros_like(depth, dtype=np.float32)
    span = max(vmax - vmin, 1e-8)
    norm[mask] = np.clip((depth[mask] - vmin) / span, 0.0, 1.0)
    rgb = plt.get_cmap("turbo")(norm)[..., :3]
    rgb[~mask] = 0.18
    return rgb


def to_rgb(chw: np.ndarray) -> np.ndarray:
    return np.clip(chw.transpose(1, 2, 0), 0, 1)


def scene_index(dataset: DL3DVDataset, name: str) -> int:
    for i, entry in enumerate(dataset.scenes):
        if entry["scene"] == name:
            return i
    raise SystemExit(f"scene {name!r} not in {dataset.name}")


def load_windows():
    eth3d = DL3DVDataset(
        "/home/grace/eth3d-eval",
        split="all",
        num_frames=NUM_FRAMES,
        sampling="covisibility",
        augment=False,
        seed=1234,
        depth_root="/home/grace/eth3d-eval/depth",
        dense_only=True,
        name="eth3d",
    )
    scannet = DL3DVDataset(
        "/home/grace/scannet-train",
        split="val",
        num_frames=NUM_FRAMES,
        sampling="covisibility",
        augment=False,
        seed=1234,
        depth_root="/home/grace/scannet-train/depth",
        dense_only=True,
        name="scannet",
    )
    datasets = {"eth3d": eth3d, "scannet": scannet}
    windows = []
    for ds_name, scene in SCENES:
        sample = datasets[ds_name][scene_index(datasets[ds_name], scene)]
        batch = collate_scenes([sample])
        windows.append(
            {
                "label": f"{ds_name}/{scene}",
                "images": batch["images"],
                "gt": batch["depth"][0].numpy(),
                "mask": batch["depth_mask"][0].numpy().astype(bool),
                "rgb": batch["images"][0].numpy(),
            }
        )
    return windows


@torch.inference_mode()
def predict(ckpt: Path, windows: list[dict], device: torch.device) -> list[np.ndarray]:
    model, step, _ = load_checkpoint(ckpt, None, device)
    print(f"  {ckpt.parent.name}  step {step}", flush=True)
    preds = []
    for w in windows:
        images = w["images"].to(device)
        out = model(images)["depth"].float().cpu()
        if out.dim() == 5:
            out = out.squeeze(-1)
        pred = out[0].numpy()
        gt = torch.from_numpy(w["gt"])[None]
        mask = torch.from_numpy(w["mask"])[None]
        ss = depth_metrics(out, gt, mask, align_scale_shift=True)
        preds.append(np.clip(ss["ss_scale"] * pred + ss["ss_shift"], 1e-6, None))
        del out
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading windows", flush=True)
    windows = load_windows()

    run_preds: list[list[np.ndarray]] = []
    for name, ckpt in RUNS:
        print(f"running {name}", flush=True)
        run_preds.append(predict(ckpt, windows, device))

    n_rows, n_cols = len(windows), 2 + len(RUNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.9 * n_rows), dpi=140)
    col_titles = ["RGB", "GT"] + [n for n, _ in RUNS]

    for r, w in enumerate(windows):
        gt, mask = w["gt"][FRAME], w["mask"][FRAME]
        valid = mask & np.isfinite(gt) & (gt > 1e-6)
        vmin, vmax = np.percentile(gt[valid], [2, 98]) if valid.any() else (0.0, 1.0)
        cells = [to_rgb(w["rgb"][FRAME]), colorize(gt, valid, vmin, vmax)]
        for c in range(len(RUNS)):
            pred = run_preds[c][r][FRAME]
            cells.append(colorize(pred, np.ones_like(valid), vmin, vmax))
        for c, img in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(w["label"], fontsize=9)

    fig.tight_layout(w_pad=0.15, h_pad=0.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
