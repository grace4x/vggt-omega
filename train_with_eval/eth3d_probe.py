"""Fixed ETH3D probe set for TracIn-style loss tracing during training.

Builds the same dataset and window sampling as `training/evaluate.py` so
delta_L during training is comparable to the benchmark eval command:

    python training/evaluate.py \\
        --checkpoint runs/foo/final.pt \\
        --data-root ~/eth3d-eval --depth-root ~/eth3d-eval/depth \\
        --split all --num-frames 8 --repeats 3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.dl3dv_dataset import DL3DVDataset, collate_scenes


@dataclass
class Eth3dProbeConfig:
    data_root: Path | str
    depth_root: Path | str | None
    split: str = "all"
    num_frames: int | None = None
    resolution: int | None = None
    sampling: str = "covisibility"
    dense_only: bool = False
    repeats: int = 3
    seed: int = 1234
    max_windows: int = 0  # 0 = all scenes x repeats
    batch_size: int = 1


def build_eth3d_dataset(cfg: Eth3dProbeConfig, *, train_num_frames: int | None, train_resolution: int | None):
    num_frames = cfg.num_frames if cfg.num_frames is not None else train_num_frames
    if num_frames is None:
        raise ValueError("num_frames must be set on Eth3dProbeConfig or passed as train_num_frames")
    resolution = cfg.resolution if cfg.resolution is not None else train_resolution
    return DL3DVDataset(
        cfg.data_root,
        split=cfg.split,
        num_frames=num_frames,
        resolution=resolution,
        sampling=cfg.sampling,
        augment=False,
        seed=cfg.seed,
        depth_root=cfg.depth_root,
        dense_only=cfg.dense_only,
    )


def collect_eth3d_probe_batches(
    cfg: Eth3dProbeConfig,
    *,
    train_num_frames: int | None = None,
    train_resolution: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Materialise fixed ETH3D windows for before/after probe loss.

    Re-seeds the dataset between repeats exactly like `evaluate_checkpoint`.
    """
    dataset = build_eth3d_dataset(cfg, train_num_frames=train_num_frames, train_resolution=train_resolution)
    n_scenes = len(dataset.scenes)

    batches: list[dict] = []
    for repeat in range(cfg.repeats):
        dataset.seed = cfg.seed + 100_003 * repeat
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_scenes,
        )
        for batch in loader:
            batches.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()})
            if cfg.max_windows and len(batches) >= cfg.max_windows:
                break
        if cfg.max_windows and len(batches) >= cfg.max_windows:
            break

    meta = {
        "n_scenes": n_scenes,
        "n_windows": len(batches),
        "repeats": cfg.repeats,
        "num_frames": dataset.num_frames,
        "image_hw": dataset.image_hw,
        "split": cfg.split,
        "sampling": cfg.sampling,
        "seed": cfg.seed,
    }
    return batches, meta
