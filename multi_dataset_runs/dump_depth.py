#!/usr/bin/env python3
"""Dump predicted vs GT depth maps from the diversity_by_batch checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dl3dv_dataset import DL3DVDataset, collate_scenes  # noqa: E402
from training.evaluate import load_checkpoint  # noqa: E402
from training.losses import depth_metrics  # noqa: E402

OUT = ROOT / "runs/diversity_by_batch/depth_vis"
CKPT = ROOT / "runs/diversity_by_batch/final.pt"

# Held-out ETH3D: outdoor courtyard, indoor office, indoor pipes.
ETH3D_SCENES = ("courtyard", "office", "pipes")
# One ScanNet indoor RGB-D window for a different capture domain.
SCANNET_SCENE = "scene0081_00"  # held-out ScanNet val split
FRAMES_TO_SAVE = (0, 3, 7)  # first / mid / last of the 8-frame window


def colorize(depth: np.ndarray, mask: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    norm = np.zeros_like(depth, dtype=np.float32)
    span = max(vmax - vmin, 1e-8)
    norm[mask] = np.clip((depth[mask] - vmin) / span, 0.0, 1.0)
    rgb = plt.get_cmap(cmap)(norm)[..., :3]
    rgb[~mask] = 0.18  # dark grey for missing GT
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def error_colorize(rel: np.ndarray, mask: np.ndarray, vmax: float = 0.5) -> np.ndarray:
    rgb = plt.get_cmap("hot")(np.clip(rel / vmax, 0.0, 1.0))[..., :3]
    rgb[~mask] = 0.18
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def save_png(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def to_uint8_image(chw: np.ndarray) -> np.ndarray:
    return (np.clip(chw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def side_by_side(panels: list[tuple[np.ndarray, str]], title: str, out_path: Path) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2), dpi=140)
    if n == 1:
        axes = [axes]
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def scene_index(dataset: DL3DVDataset, name: str) -> int:
    for i, entry in enumerate(dataset.scenes):
        if entry["scene"] == name:
            return i
    raise SystemExit(f"scene {name!r} not in {dataset.name} ({len(dataset.scenes)} scenes)")


@torch.inference_mode()
def run_scene(model, dataset: DL3DVDataset, name: str, device: torch.device, out_dir: Path) -> dict:
    idx = scene_index(dataset, name)
    sample = dataset[idx]
    batch = collate_scenes([sample])
    images = batch["images"].to(device)
    pred = model(images)

    pred_depth = pred["depth"].float().cpu()
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)
    pred_conf = pred["depth_conf"].float().cpu()
    if pred_conf.dim() == 5:
        pred_conf = pred_conf.squeeze(-1)

    gt = batch["depth"].cpu()  # (1, S, H, W)
    mask = batch["depth_mask"].cpu().bool()
    rgb = batch["images"][0].cpu().numpy()  # (S, 3, H, W)
    pred_np = pred_depth[0].numpy()
    gt_np = gt[0].numpy()
    mask_np = mask[0].numpy()
    conf_np = pred_conf[0].numpy()

    metrics = depth_metrics(pred_depth, gt, mask, align_median=False)
    aligned = depth_metrics(pred_depth, gt, mask, align_median=True)
    scale = float(metrics["scale_ratio"])
    pred_aligned = pred_np * scale

    valid = mask_np & np.isfinite(gt_np) & (gt_np > 1e-6)
    vmin, vmax = np.percentile(gt_np[valid], [2, 98]) if valid.any() else (0.0, 1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for f in FRAMES_TO_SAVE:
        m = valid[f]
        rgb_u8 = to_uint8_image(rgb[f])
        gt_u8 = colorize(gt_np[f], m, vmin, vmax)
        pred_u8 = colorize(pred_aligned[f], np.ones_like(m), vmin, vmax)
        pred_raw_u8 = colorize(pred_np[f], np.ones_like(m), float(np.percentile(pred_np[f], 2)), float(np.percentile(pred_np[f], 98)))
        rel = np.zeros_like(gt_np[f], dtype=np.float32)
        rel[m] = np.abs(pred_aligned[f][m] - gt_np[f][m]) / np.maximum(gt_np[f][m], 1e-6)
        err_u8 = error_colorize(rel, m, vmax=0.5)
        conf_u8 = colorize(conf_np[f], np.ones_like(m), float(np.percentile(conf_np[f], 2)), float(np.percentile(conf_np[f], 98)), cmap="viridis")

        stem = f"frame{f:02d}"
        save_png(out_dir / f"{stem}_rgb.png", rgb_u8)
        save_png(out_dir / f"{stem}_gt.png", gt_u8)
        save_png(out_dir / f"{stem}_pred_aligned.png", pred_u8)
        save_png(out_dir / f"{stem}_pred_raw.png", pred_raw_u8)
        save_png(out_dir / f"{stem}_error.png", err_u8)
        save_png(out_dir / f"{stem}_conf.png", conf_u8)
        side_by_side(
            [
                (rgb_u8, "RGB"),
                (gt_u8, "GT depth"),
                (pred_u8, "Pred (median-aligned)"),
                (err_u8, "|pred−gt| / gt"),
            ],
            f"{dataset.name}/{name}  frame {f}  abs_rel_aligned={aligned['abs_rel']:.3f}",
            out_dir / f"{stem}_compare.png",
        )
        frame_abs = float(rel[m].mean()) if m.any() else float("nan")
        records.append({"frame": f, "abs_rel_aligned_pixels": frame_abs, "valid_frac": float(m.mean())})

    info = {
        "dataset": dataset.name,
        "scene": name,
        "scene_scale": float(batch["scene_scale"][0]),
        "image_hw": list(rgb.shape[-2:]),
        "num_frames": int(rgb.shape[0]),
        "scale_ratio_median": scale,
        "abs_rel_raw": metrics["abs_rel"],
        "abs_rel_aligned": aligned["abs_rel"],
        "delta_1.25_aligned": aligned["delta_1.25"],
        "frames": records,
    }
    (out_dir / "metrics.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {CKPT} on {device}", flush=True)
    model, step, train_args = load_checkpoint(CKPT, None, device)
    num_frames = int(train_args.get("num_frames") or 8)
    print(f"step {step}  preset {train_args.get('preset')}  num_frames {num_frames}", flush=True)

    eth3d = DL3DVDataset(
        "/home/grace/eth3d-eval",
        split="all",
        num_frames=num_frames,
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
        num_frames=num_frames,
        sampling="covisibility",
        augment=False,
        seed=1234,
        depth_root="/home/grace/scannet-train/depth",
        dense_only=True,
        name="scannet",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in ETH3D_SCENES:
        print(f"ETH3D {name}", flush=True)
        summaries.append(run_scene(model, eth3d, name, device, OUT / f"eth3d_{name}"))

    scannet_name = SCANNET_SCENE
    if not any(e["scene"] == scannet_name for e in scannet.scenes):
        scannet_name = scannet.scenes[0]["scene"]
        print(f"{SCANNET_SCENE} not in val; using {scannet_name}", flush=True)
    print(f"ScanNet {scannet_name}", flush=True)
    summaries.append(run_scene(model, scannet, scannet_name, device, OUT / f"scannet_{scannet_name}"))

    (OUT / "summary.json").write_text(
        json.dumps({"checkpoint": str(CKPT), "step": step, "scenes": summaries}, indent=2) + "\n"
    )
    print(f"wrote {OUT}", flush=True)
    for s in summaries:
        print(
            f"  {s['dataset']}/{s['scene']}: abs_rel_raw={s['abs_rel_raw']:.3f}  "
            f"abs_rel_aligned={s['abs_rel_aligned']:.3f}  scale={s['scale_ratio_median']:.3f}"
        )


if __name__ == "__main__":
    main()
