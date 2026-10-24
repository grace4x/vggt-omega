#!/usr/bin/env python3
"""Train VGGT-Omega on DL3DV (and optionally other datasets) with periodic
gradient clustering.

Every `--grad-cluster-every` epochs (default 10) the script:

  1. **Gradient pass** -- runs one forward + backward per training scene
     (batch_size=1, no optimizer step) and JL-projects the gradient into
     `--proj-dim` dimensions.
  2. **K-means** -- into `--grad-cluster-k` clusters (default: 512).
     `--cluster-metric euclidean` (default) is plain Euclidean k-means on the
     unnormalised projected vectors.  `--cluster-metric cosine` L2-normalises
     each projected gradient and runs spherical k-means (cosine similarity).
     Scene-token embeddings are clustered with the same metric.

Training keeps a random shuffle of the full dataset unless `--equal-mix` /
`--mix-proportions` (fixed per-dataset shares) or `--complexity-weighting` is
set; the latter re-weights the token clusters after each clustering for the
epochs that follow.  Cluster assignments (and weights)
are saved to `<out>/grad_clusters/epoch_NNNN.npz` for post-hoc inspection.

Single GPU example::

    python grad_cluster_train/train.py \\
        --data-root ~/dl3dv-train \\
        --depth-root ~/dl3dv-depth \\
        --preset small --dinov3 checkpoints/dinov3_vits16.pt \\
        --out runs/dl3dv-grad-cluster \\
        --num-frames 16 --max-steps 200000 \\
        --grad-cluster-every 10 --grad-cluster-k 64

Multi-GPU::

    torchrun --nproc_per_node=4 grad_cluster_train/train.py \\
        --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth \\
        --preset small --out runs/dl3dv-grad-cluster-4gpu \\
        --grad-cluster-every 10 --grad-cluster-k 64

All flags from the base training/train.py are supported unchanged.
Additional flags::

    --grad-cluster-every N   Recompute gradient clusters every N epochs (default 10).
    --grad-cluster-k K       Number of k-means clusters (default 512).
    --cluster-metric M       `euclidean` (default) or `cosine`.  Cosine
                             L2-normalises projected gradients and scene-token
                             embeddings, then spherical k-means.
    --proj-dim D             JL projection width before k-means (default 400).
    --proj-workers W         DataLoader workers for the gradient pass (default 4).
    --no-grad-cluster        Disable gradient clustering entirely (plain training).
    --cosine-center C        `tokens` (default), `both` or `none`: subtract the
                             mean over scenes before cosine k-means/complexity.
    --complexity-weighting   Draw token clusters by softmax(C / tau) over the
                             pruning paper's complexity C = d_inter * d_intra
                             (arXiv:2401.04578), scenes uniform within a cluster.
    --complexity-tau T       Softmax temperature (default 0.1).
    --complexity-neighbors L Centroids averaged into d_inter (default 20).
    --cluster-balance A      After each clustering, flatten token-cluster sizes
                             inside each dataset: a cluster's draw mass goes as
                             size**A (A=1 pool, A=0 every cluster equal).  Each
                             dataset keeps its pool share of the draws.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Ensure the repo root is on the path regardless of how the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from training.dl3dv_dataset import (  # noqa: E402
    DEFAULT_WINDOW_SIZE,
    DL3DVDataset,
    FovCropCollate,
    FovJitterCollate,
    SAMPLING_CHOICES,
)
from training.mixed_dataset import (  # noqa: E402
    TaggedDataset,
    assert_stackable,
    build_concat_trainset,
    collate_mixed,
    dataset_mix_weights,
)
from training.losses import VGGTOmegaLoss, depth_metrics, pose_metrics  # noqa: E402
from training.model_config import PRESETS, build_model, parameter_summary  # noqa: E402

from grad_cluster_train.grad_sampler import (  # noqa: E402
    GradClusterSampler,
    KeyedDataset,
    recompute_clusters,
)


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# optimiser / schedule
# --------------------------------------------------------------------------- #


def build_param_groups(model, weight_decay: float, lr: float, backbone_lr_mult: float) -> list[dict]:
    groups = {
        ("backbone", True):  {"params": [], "weight_decay": weight_decay, "lr": lr * backbone_lr_mult},
        ("backbone", False): {"params": [], "weight_decay": 0.0,          "lr": lr * backbone_lr_mult},
        ("head",     True):  {"params": [], "weight_decay": weight_decay, "lr": lr},
        ("head",     False): {"params": [], "weight_decay": 0.0,          "lr": lr},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "aggregator.patch_embed" in name
        decay = param.ndim >= 2
        groups[("backbone" if is_backbone else "head", decay)]["params"].append(param)

    out = []
    for (scope, decay), group in groups.items():
        if group["params"]:
            group["name"] = f"{scope}_{'decay' if decay else 'nodecay'}"
            out.append(group)
    return out


def lr_lambda_factory(warmup_steps: int, max_steps: int, min_lr_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return fn


# --------------------------------------------------------------------------- #
# train / eval steps
# --------------------------------------------------------------------------- #


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_to_device(batch, device)
        predictions = model(batch["images"])
        _, logs = criterion(predictions, batch)

        metrics = {k: v.item() for k, v in logs.items()}
        metrics.update(pose_metrics(predictions["pose_enc"], batch["pose_enc"]))
        metrics.update(depth_metrics(predictions["depth"], batch["depth"], batch["depth_mask"]))
        for k, v in metrics.items():
            if math.isfinite(v):
                totals[k] = totals.get(k, 0.0) + v
        count += 1

    model.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def evaluate_all(model, loaders: dict, criterion, device, max_batches: int) -> dict[str, float]:
    out: dict[str, float] = {}
    per_dataset = {}
    for name, loader in loaders.items():
        metrics = evaluate(model, loader, criterion, device, max_batches)
        per_dataset[name] = metrics
        for k, v in metrics.items():
            out[f"{name}/{k}"] = v
    if len(per_dataset) > 1:
        keys = set().union(*(m.keys() for m in per_dataset.values()))
        for k in keys:
            values = [m[k] for m in per_dataset.values() if k in m and math.isfinite(m[k])]
            if values:
                out[k] = sum(values) / len(values)
    elif per_dataset:
        out.update(next(iter(per_dataset.values())))
    return out


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, args) -> None:
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def save_weights(path: Path, model, step: int, args) -> None:
    """Model-only checkpoint (~1/3 the size of `save_checkpoint`); not resumable."""
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": raw.state_dict(), "args": vars(args)}, path)


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def parse_mix_proportions(spec: str) -> dict[str, float]:
    proportions: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--mix-proportions expects NAME=FRAC pairs, got {part!r}")
        name, _, value = part.partition("=")
        name = name.strip()
        try:
            proportions[name] = float(value)
        except ValueError:
            raise SystemExit(f"--mix-proportions: {value!r} for {name!r} is not a number")
    if not proportions:
        raise SystemExit("--mix-proportions needs at least one NAME=FRAC pair")
    return proportions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- data roots ----
    p.add_argument("--data-root", type=Path, default=None,
                   help="output of preprocess_dl3dv.py")
    p.add_argument("--depth-root", type=Path, default=None,
                   help="output of fetch_dl3dv_depth.py (dense GT)")
    p.add_argument("--scannet-root", type=Path, default=None,
                   help="output of preprocess_scannet.py")
    p.add_argument("--scannet-depth-root", type=Path, default=None)
    p.add_argument("--megasynth-root", type=Path, default=None)
    p.add_argument("--megasynth-depth-root", type=Path, default=None)
    p.add_argument("--hypersim-root", type=Path, default=None)
    p.add_argument("--hypersim-depth-root", type=Path, default=None)
    p.add_argument("--tartanair-root", type=Path, default=None)
    p.add_argument("--tartanair-depth-root", type=Path, default=None)

    # ---- per-dataset weights ----
    p.add_argument("--dl3dv-weight",    type=float, default=1.0)
    p.add_argument("--scannet-weight",  type=float, default=1.0)
    p.add_argument("--megasynth-weight", type=float, default=1.0)
    p.add_argument("--hypersim-weight", type=float, default=1.0)
    p.add_argument("--tartanair-weight", type=float, default=1.0)
    p.add_argument("--equal-mix", action="store_true")
    p.add_argument("--mix-proportions", type=str, default=None, metavar="NAME=FRAC,...")
    p.add_argument("--dense-only", action="store_true")

    # ---- output ----
    p.add_argument("--out", type=Path, default=Path("runs/dl3dv-grad-cluster"))

    # ---- model ----
    p.add_argument("--preset", default="small", choices=sorted(PRESETS))
    p.add_argument("--dinov3", type=Path, default=None)
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--freeze-backbone-steps", type=int, default=0)

    # ---- sampling / augmentation ----
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--sampling", default="window", choices=SAMPLING_CHOICES)
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--fov-crop", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--fov-jitter", type=int, nargs=2, default=None, metavar=("MIN_H", "MIN_W"))

    # ---- batching ----
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)

    # ---- optimiser ----
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=50.0)

    # ---- loss weights ----
    p.add_argument("--weight-camera",   type=float, default=5.0)
    p.add_argument("--weight-depth",    type=float, default=1.0)
    p.add_argument("--weight-point",    type=float, default=0.5)
    p.add_argument("--weight-gradient", type=float, default=1.0)
    p.add_argument("--conf-alpha",      type=float, default=0.2)

    # ---- logging / checkpointing ----
    p.add_argument("--log-every",  type=int, default=50)
    p.add_argument("--val-every",  type=int, default=2000)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument(
        "--keep-steps", type=int, nargs="+", default=[], metavar="STEP",
        help="also write model-only weights to <out>/stepNNNNNN.pt at these steps,\n"
             "for checkpoint averaging (see average_checkpoints.py).",
    )
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", type=int, default=0)
    p.add_argument("--scene-list", type=Path, default=None)
    p.add_argument("--dl3dv-scenes",    type=int, default=0)
    p.add_argument("--scannet-scenes",  type=int, default=0)
    p.add_argument("--megasynth-scenes", type=int, default=0)
    p.add_argument("--hypersim-scenes", type=int, default=0)
    p.add_argument("--tartanair-scenes", type=int, default=0)

    # ---- gradient clustering (G-DIG / LESS style) ----
    p.add_argument(
        "--grad-cluster-every", type=int, default=10,
        help="recompute gradient clusters every N epochs (default: 10).",
    )
    p.add_argument(
        "--grad-cluster-k", type=int, default=512,
        help="number of k-means clusters (default: 512, matching G-DIG).\n"
             "0 = auto: ceil(sqrt(N_scenes)).",
    )
    p.add_argument(
        "--cluster-metric", choices=("euclidean", "cosine"), default="euclidean",
        help="distance used to cluster projected gradients and scene-token\n"
             "embeddings (default: euclidean).\n"
             "euclidean: plain k-means on unnormalised vectors (G-DIG).\n"
             "cosine: L2-normalise each vector, then spherical k-means\n"
             "(assignment by cosine similarity).  For gradients this is\n"
             "normalise + the existing JL projection + cosine k-means.",
    )
    p.add_argument(
        "--proj-dim", type=int, default=400,
        help="Gaussian JL projection output dimension (default: 400, matching\n"
             "G-DIG).  Gradients are projected from P (total params) to this\n"
             "dimension before k-means.",
    )
    p.add_argument(
        "--proj-seed", type=int, default=42,
        help="fixed RNG seed for the JL projection matrix (default: 42).\n"
             "All scenes are projected into the same subspace.",
    )
    p.add_argument(
        "--proj-workers", type=int, default=4,
        help="DataLoader workers for the gradient projection pass (default: 4).",
    )
    p.add_argument(
        "--no-grad-cluster", action="store_true",
        help="disable gradient clustering entirely (plain training).",
    )
    p.add_argument(
        "--cosine-center", choices=("tokens", "both", "none"), default="tokens",
        help="subtract the mean over scenes before cosine geometry (spherical\n"
             "k-means and complexity): tokens only (default), tokens and\n"
             "gradients, or neither.  No effect on Euclidean k-means.",
    )
    p.add_argument(
        "--complexity-weighting", action="store_true",
        help="after each clustering, sample token clusters by\n"
             "softmax(C / tau) over C = d_inter * d_intra (arXiv:2401.04578,\n"
             "Eq. 1-2) and scenes uniformly within a cluster.  Uniform until\n"
             "the first clustering.",
    )
    p.add_argument(
        "--complexity-tau", type=float, default=0.1,
        help="softmax temperature over token-cluster complexity (default: 0.1).",
    )
    p.add_argument(
        "--complexity-neighbors", type=int, default=20,
        help="nearest centroids averaged into d_inter (default: 20).",
    )
    p.add_argument(
        "--cluster-balance", type=float, default=None, metavar="ALPHA",
        help="after each clustering, draw scenes with weight (cluster size)**(ALPHA-1)\n"
             "so a token cluster's draw mass goes as size**ALPHA, then rescale each\n"
             "dataset to its pool share.  ALPHA=0 gives every token cluster in a\n"
             "dataset the same mass; ALPHA=1 leaves the pool unchanged but still\n"
             "draws with replacement (a control).  Uniform until the first clustering.\n"
             "Not combinable with --complexity-weighting/--mix-proportions/--equal-mix.",
    )
    p.add_argument(
        "--cluster-balance-shuffle", action="store_true",
        help="control for --cluster-balance: shuffle the token-cluster labels within each\n"
             "dataset before weighting, so cluster sizes are kept but membership is random.",
    )

    return p


def source_shares(
    weights: np.ndarray, train_set, sizes: list[int], names: list[str],
) -> dict[str, float]:
    """Each source dataset's share of the draws under per-index `weights`."""
    base = np.asarray(train_set.indices) if isinstance(train_set, Subset) else np.arange(sum(sizes))
    source = np.searchsorted(np.cumsum(sizes), base, side="right")
    return {n: float(weights[source == i].sum()) for i, n in enumerate(names)}


# --------------------------------------------------------------------------- #
# dataset construction helpers
# --------------------------------------------------------------------------- #


def _make_dl3dv_split(
    args,
    split: str,
    *,
    augment: bool,
    seed: int | None,
    rank: int,
    is_main_rank: bool,
) -> dict[str, DL3DVDataset]:
    """Build one split across all configured dataset roots."""
    sources = [
        ("dl3dv",     args.data_root,      args.depth_root,       args.dense_only),
        ("scannet",   args.scannet_root,   args.scannet_depth_root or
         (args.scannet_root / "depth" if args.scannet_root else None),   False),
        ("megasynth", args.megasynth_root, args.megasynth_depth_root or
         (args.megasynth_root / "depth" if args.megasynth_root else None), False),
        ("hypersim",  args.hypersim_root,  args.hypersim_depth_root or
         (args.hypersim_root / "depth" if args.hypersim_root else None),  False),
        ("tartanair", args.tartanair_root, args.tartanair_depth_root or
         (args.tartanair_root / "depth" if args.tartanair_root else None), False),
    ]
    built = {}
    for name, root, depth_root, dense_only in sources:
        if root is None:
            continue
        try:
            built[name] = DL3DVDataset(
                root,
                name=name,
                split=split,
                num_frames=args.num_frames,
                resolution=args.resolution,
                sampling=args.sampling,
                window_size=args.window_size,
                augment=augment,
                seed=seed,
                depth_root=depth_root,
                dense_only=dense_only,
            )
        except ValueError as exc:
            if split == "train":
                raise
            if is_main_rank:
                print(f"[{name}] no {split} split ({exc}); skipping")
    return built


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = build_parser().parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    # ---- model ----
    model = build_model(
        args.preset,
        use_checkpoint=args.checkpointing,
        dinov3_checkpoint=args.dinov3,
    ).to(device)

    if is_main(rank):
        summary = parameter_summary(model)
        print(f"preset={args.preset}  " + "  ".join(f"{k}={v:.1f}M" for k, v in summary.items()))

    ddp_model = model
    if world_size > 1:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.freeze_backbone_steps > 0,
        )

    criterion = VGGTOmegaLoss(
        weight_camera=args.weight_camera,
        weight_depth=args.weight_depth,
        weight_point=args.weight_point,
        weight_gradient=args.weight_gradient,
        depth_kwargs={"alpha": args.conf_alpha},
    )

    # ---- data root check ----
    if all(
        getattr(args, r) is None
        for r in ("data_root", "scannet_root", "megasynth_root", "hypersim_root", "tartanair_root")
    ):
        raise SystemExit(
            "Pass at least one of --data-root (DL3DV), --scannet-root, "
            "--megasynth-root, --hypersim-root, --tartanair-root"
        )

    # ---- build training split ----
    train_parts = _make_dl3dv_split(
        args, "train", augment=True, seed=None, rank=rank, is_main_rank=is_main(rank)
    )

    wanted = set(args.scene_list.read_text().split()) if args.scene_list is not None else set()
    listed_subsets = {line.split("/", 1)[0] for line in wanted}

    for name, dataset in train_parts.items():
        if wanted and not any(e["subset"] in listed_subsets for e in dataset.scenes):
            if is_main(rank):
                print(f"[scene-list] {name}: not named in the list; keeping all {len(dataset.scenes)} scenes")
        elif wanted:
            dataset.scenes = [e for e in dataset.scenes if f"{e['subset']}/{e['scene']}" in wanted]
            if is_main(rank):
                print(f"[scene-list] {name}: {len(dataset.scenes)} listed scenes kept")

        cap = {
            "dl3dv": args.dl3dv_scenes, "scannet": args.scannet_scenes,
            "megasynth": args.megasynth_scenes, "hypersim": args.hypersim_scenes,
            "tartanair": args.tartanair_scenes,
        }.get(name, 0)
        if cap and cap < len(dataset.scenes):
            dataset.scenes = dataset.scenes[:cap]
            if is_main(rank):
                print(f"[{name}] capped to the first {len(dataset.scenes)} scenes")

        if args.overfit:
            dataset.scenes = dataset.scenes[: args.overfit]
            dataset.seed = 0
            dataset.augment = False

    train_parts = {n: d for n, d in train_parts.items() if len(d.scenes) > 0}
    if not train_parts:
        raise SystemExit("no training scenes survived filters")

    image_hw = assert_stackable(train_parts, args.batch_size)

    listed_weights = {
        "dl3dv": args.dl3dv_weight, "scannet": args.scannet_weight,
        "megasynth": args.megasynth_weight, "hypersim": args.hypersim_weight,
        "tartanair": args.tartanair_weight,
    }
    custom_proportions = parse_mix_proportions(args.mix_proportions) if args.mix_proportions else None
    if args.equal_mix and custom_proportions is not None:
        raise SystemExit("--equal-mix and --mix-proportions are mutually exclusive")
    weighted_mix = args.equal_mix or custom_proportions is not None
    if weighted_mix:
        flag = "--equal-mix" if args.equal_mix else "--mix-proportions"
        if args.scene_list is not None:
            raise SystemExit(f"{flag} samples the full pools; omit --scene-list")
        if len(train_parts) < 2:
            raise SystemExit(f"{flag} needs at least two datasets")
        if any(abs(listed_weights[n] - 1.0) > 1e-9 for n in train_parts):
            raise SystemExit(f"{flag} already sets the mix; omit --*-weight")
        if custom_proportions is not None and set(custom_proportions) != set(train_parts):
            raise SystemExit(
                f"--mix-proportions must name exactly the configured datasets "
                f"{sorted(train_parts)}, got {sorted(custom_proportions)}"
            )
        weights = None
    else:
        weights = listed_weights

    train_set, source_names, sizes, epoch_counts = build_concat_trainset(
        train_parts, weights, seed=args.seed
    )

    # ---- collate ----
    if args.fov_crop is not None and args.fov_jitter is not None:
        raise SystemExit("use only one of --fov-crop and --fov-jitter")
    train_collate = collate_mixed
    val_collate = collate_mixed
    if args.fov_crop is not None:
        train_collate = val_collate = FovCropCollate(collate_mixed, args.fov_crop)
    elif args.fov_jitter is not None:
        train_collate = FovJitterCollate(
            collate_mixed, args.fov_jitter,
            patch_size=PRESETS[args.preset].patch_size, seed=args.seed,
        )

    # ---- epoch sampler ----
    # Random shuffle of the full training set (replaces DistributedSampler).
    # --equal-mix / --mix-proportions draw with replacement from per-scene
    # dataset-share weights instead; --complexity-weighting swaps in per-scene
    # weights after each clustering.
    use_grad_cluster = not args.no_grad_cluster and not args.overfit
    if args.complexity_weighting:
        if not use_grad_cluster:
            raise SystemExit("--complexity-weighting needs clustering; drop --no-grad-cluster/--overfit")
        if weighted_mix:
            raise SystemExit("--complexity-weighting sets the draw mix; omit --equal-mix/--mix-proportions")
        if args.complexity_tau <= 0:
            raise SystemExit("--complexity-tau must be > 0")
        if args.complexity_neighbors < 1:
            raise SystemExit("--complexity-neighbors must be >= 1")
    if args.cluster_balance is not None:
        if not use_grad_cluster:
            raise SystemExit("--cluster-balance needs clustering; drop --no-grad-cluster/--overfit")
        if weighted_mix or args.complexity_weighting:
            raise SystemExit(
                "--cluster-balance sets the draw mix; omit --complexity-weighting/--equal-mix/--mix-proportions"
            )
        if not 0.0 <= args.cluster_balance <= 1.0:
            raise SystemExit("--cluster-balance must be in [0, 1]")
        if args.scene_list is not None:
            raise SystemExit("--cluster-balance samples the full pools; omit --scene-list")
        if any(abs(listed_weights[n] - 1.0) > 1e-9 for n in train_parts):
            raise SystemExit("--cluster-balance keeps the pool shares; omit --*-weight")
    if args.cluster_balance_shuffle and args.cluster_balance is None:
        raise SystemExit("--cluster-balance-shuffle needs --cluster-balance")
    resample = args.complexity_weighting or args.cluster_balance is not None
    epoch_size = len(train_set)
    grad_sampler = GradClusterSampler(
        n_total=len(train_set),
        epoch_size=epoch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        keyed=True,
    )
    if weighted_mix:
        grad_sampler.set_scene_weights(dataset_mix_weights(source_names, sizes, custom_proportions))
    # Keyed draws make each clip a function of (seed, epoch, scene), not OS
    # entropy, so a run is reproducible up to GPU nondeterminism and two arms
    # with the same seed share clips wherever they share draws.
    train_loader = DataLoader(
        KeyedDataset(train_set, train_parts.values()),
        batch_size=args.batch_size,
        sampler=grad_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        collate_fn=train_collate,
    )

    # ---- gradient dataset (augment=False for deterministic sketch pass) ----
    # Built separately from the training set so augmentation is disabled.
    # Only needed on rank 0 (gradient computation runs there only).
    grad_dataset = None
    grad_groups = None
    if use_grad_cluster and is_main(rank):
        grad_parts = _make_dl3dv_split(
            args, "train", augment=False, seed=args.seed,
            rank=rank, is_main_rank=True,
        )
        # Apply the same scene filters as the training set.
        for name, ds in grad_parts.items():
            if wanted and any(e["subset"] in listed_subsets for e in ds.scenes):
                ds.scenes = [e for e in ds.scenes if f"{e['subset']}/{e['scene']}" in wanted]
            cap = {
                "dl3dv": args.dl3dv_scenes, "scannet": args.scannet_scenes,
                "megasynth": args.megasynth_scenes, "hypersim": args.hypersim_scenes,
                "tartanair": args.tartanair_scenes,
            }.get(name, 0)
            if cap and cap < len(ds.scenes):
                ds.scenes = ds.scenes[:cap]
        grad_parts = {n: d for n, d in grad_parts.items() if len(d.scenes) > 0}
        if grad_parts:
            grad_dataset, _, grad_sizes, _ = build_concat_trainset(grad_parts, weights, seed=args.seed)
            if args.cluster_balance is not None:
                # With the default unit weights the concat is one pass over every
                # scene, in source order; the balance runs inside each source.
                if len(grad_dataset) != sum(grad_sizes):
                    raise SystemExit("--cluster-balance needs each scene once; omit --*-weight")
                grad_groups = np.searchsorted(
                    np.cumsum(grad_sizes), np.arange(len(grad_dataset)), side="right"
                )

    # ---- validation loaders ----
    val_loaders: dict[str, DataLoader] = {}
    if args.val_every and not args.overfit:
        for name, dataset in _make_dl3dv_split(
            args, "val", augment=False, seed=1234, rank=rank, is_main_rank=is_main(rank)
        ).items():
            val_loaders[name] = DataLoader(
                TaggedDataset(dataset, name),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(args.workers // 2, 1),
                pin_memory=True,
                collate_fn=val_collate,
            )

    if is_main(rank):
        if weighted_mix:
            mix_weights_arr = dataset_mix_weights(source_names, sizes, custom_proportions)
            shares = {}
            offset = 0
            for n, s in zip(source_names, sizes):
                shares[n] = float(mix_weights_arr[offset: offset + s].sum())
                offset += s
            label = "equal-mix" if custom_proportions is None else "mix-proportions"
            mix = "  ".join(f"{n}={s}({shares[n]:.0%} draws)" for n, s in zip(source_names, sizes))
            mix = f"{label} {mix}"
        else:
            mix = "  ".join(
                f"{n}={s}" + (f"(x{c / s:.2g})" if c != s else "")
                for n, s, c in zip(source_names, sizes, epoch_counts)
            )
        val_desc = (
            "  val " + " ".join(f"{n}={len(l.dataset)}" for n, l in val_loaders.items())
            if val_loaders else "  (no val)"
        )
        cluster_desc = (
            f"  grad-cluster-every={args.grad_cluster_every}"
            f"  k={'auto(sqrt(N))' if args.grad_cluster_k == 0 else args.grad_cluster_k}"
            f"  proj-dim={args.proj_dim}"
            f"  metric={args.cluster_metric}"
            + (f"  cosine-center={args.cosine_center}" if args.cluster_metric == "cosine" or args.complexity_weighting else "")
            + (f"  complexity-weighting(tau={args.complexity_tau:g}, l={args.complexity_neighbors})"
               if args.complexity_weighting else "")
            + (f"  cluster-balance(alpha={args.cluster_balance:g})"
               if args.cluster_balance is not None else "")
            if use_grad_cluster else "  (grad-cluster disabled)"
        )
        print(
            f"train samples/epoch={epoch_size} [{mix}]{val_desc}"
            f"  {image_hw[0]}x{image_hw[1]}"
            + (f" ->fov-crop {args.fov_crop[0]}x{args.fov_crop[1]}" if args.fov_crop else "")
            + (f" (fov-jitter down to {args.fov_jitter[0]}x{args.fov_jitter[1]})" if args.fov_jitter else "")
            + f"  frames={args.num_frames}  sampling={args.sampling}"
            + (f"/{args.window_size}" if args.sampling == "window" else "")
            + f"  world_size={world_size}"
            + cluster_desc,
            flush=True,
        )

    # ---- optimiser ----
    param_groups = build_param_groups(model, args.weight_decay, args.lr, args.backbone_lr_mult)
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(args.betas))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(args.warmup_steps, args.max_steps, args.min_lr_ratio)
    )

    start_step = 0
    if args.resume is not None and args.resume.exists():
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = payload["step"]
        if is_main(rank):
            print(f"resumed from {args.resume} at step {start_step}")

    log_path = args.out / "log.jsonl"
    writer = None
    if is_main(rank):
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps(vars(args), indent=1, default=str))
        writer = SummaryWriter(log_dir=str(args.out / "tb"))

    # ---- training loop ----
    ddp_model.train()
    step = start_step
    micro = 0
    epoch = 0
    running: dict[str, float] = {}
    running_count = 0
    per_source: dict[str, float] = {}
    per_source_count: dict[str, int] = {}
    started = time.time()

    while step < args.max_steps:
        # ---- per-epoch sampler setup ----
        grad_sampler.set_epoch(epoch)
        epoch += 1

        # ---- inner training loop (one epoch) ----
        for batch in train_loader:
            if step >= args.max_steps:
                break

            if args.freeze_backbone_steps:
                requires_grad = step >= args.freeze_backbone_steps
                for param in model.aggregator.patch_embed.parameters():
                    param.requires_grad_(requires_grad)

            batch = move_to_device(batch, device)

            micro += 1
            is_last_micro = (micro % args.grad_accum) == 0
            sync_context = (
                ddp_model.no_sync()
                if (world_size > 1 and not is_last_micro)
                else nullcontext()
            )
            with sync_context:
                predictions = ddp_model(batch["images"])
                loss, logs = criterion(predictions, batch)
                (loss / args.grad_accum).backward()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item() / args.grad_accum
            if len(source_names) > 1:
                share = 1.0 / len(batch["dataset"])
                for source in batch["dataset"]:
                    per_source_count[source] = per_source_count.get(source, 0) + share
                if args.batch_size == 1:
                    source = batch["dataset"][0]
                    per_source[source] = per_source.get(source, 0.0) + logs["loss"].item()

            if not is_last_micro:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            running["grad_norm"] = running.get("grad_norm", 0.0) + grad_norm.item()
            running_count += 1

            if is_main(rank) and args.log_every and step % args.log_every == 0:
                means = {k: v / running_count for k, v in running.items()}
                seen_total = max(sum(per_source_count.values()), 1e-9)
                for source, seen in per_source_count.items():
                    means[f"frac_{source}"] = seen / seen_total
                for source, total in per_source.items():
                    means[f"loss_{source}"] = total / max(per_source_count.get(source, 0), 1e-9)
                elapsed = time.time() - started
                lrs = scheduler.get_last_lr()
                record = {
                    "step": step,
                    "epoch": epoch,
                    "lr": max(lrs),
                    "lr_backbone": min(lrs),
                    "steps_per_sec": args.log_every / max(elapsed, 1e-9),
                    **{k: round(v, 5) for k, v in means.items()},
                }
                print(
                    f"step {step:>7d}/{args.max_steps}  epoch {epoch}"
                    f"  loss {means.get('loss', 0):.4f}"
                    f"  cam {means.get('loss_camera', 0):.4f}"
                    f"  depth {means.get('loss_depth', 0):.4f}"
                    f"  point {means.get('loss_point', 0):.4f}"
                    f"  cover {means.get('depth_valid', 0) * 100:.1f}%  "
                    + "".join(
                        f"{n}{means.get(f'frac_{n}', 0) * 100:.0f}%"
                        + (f"/{means[f'loss_{n}']:.3f}" if f"loss_{n}" in means else "")
                        + "  "
                        for n in (source_names if len(source_names) > 1 else [])
                    )
                    + f"lr {record['lr']:.2e}  {record['steps_per_sec']:.2f} it/s",
                    flush=True,
                )
                with log_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                if writer is not None:
                    for k, v in record.items():
                        if k == "step" or not isinstance(v, (int, float)):
                            continue
                        writer.add_scalar(f"train/{k}", v, step)
                running, running_count, started = {}, 0, time.time()
                per_source, per_source_count = {}, {}

            if val_loaders and args.val_every and step % args.val_every == 0:
                metrics = evaluate_all(model, val_loaders, criterion, device, args.val_batches)
                if is_main(rank):
                    nan = float("nan")

                    def val_line(label: str, prefix: str = "") -> str:
                        get = lambda k: metrics.get(f"{prefix}{k}", nan)  # noqa
                        return (
                            f"  [val @ {step}] {label}loss {get('loss'):.4f}  "
                            f"AUC@30 {get('auc_at_30'):.3f}  "
                            f"rot_err {get('rot_err_deg_median'):.2f}deg  "
                            f"abs_rel {get('abs_rel'):.3f}"
                        )

                    if len(val_loaders) > 1:
                        for name in val_loaders:
                            print(val_line(f"{name:>8s} ", f"{name}/"), flush=True)
                        print(val_line("    mean "), flush=True)
                    else:
                        print(val_line(""), flush=True)
                    with log_path.open("a") as fh:
                        fh.write(json.dumps({"step": step, "split": "val", **metrics}) + "\n")
                    if writer is not None:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and math.isfinite(v):
                                writer.add_scalar(f"val/{k}", v, step)

            if is_main(rank) and args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    args.out / "latest.pt", ddp_model, optimizer, scheduler, step, args
                )
            if is_main(rank) and step in args.keep_steps:
                save_weights(args.out / f"step{step:06d}.pt", ddp_model, step, args)

        # ---- end of epoch: gradient cluster recomputation ----
        if use_grad_cluster and epoch % args.grad_cluster_every == 0:
            if is_main(rank):
                print(
                    f"\n[epoch {epoch}] recomputing gradient clusters "
                    f"(every {args.grad_cluster_every} epochs)",
                    flush=True,
                )
            scene_weights = recompute_clusters(
                model=model,           # raw model, not DDP-wrapped
                criterion=criterion,
                grad_dataset=grad_dataset,
                device=device,
                k=args.grad_cluster_k,
                proj_dim=args.proj_dim,
                proj_seed=args.proj_seed,
                proj_workers=args.proj_workers,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                out_dir=args.out,
                seed=args.seed,
                clip_grad=args.clip_grad,
                metric=args.cluster_metric,
                cosine_center=args.cosine_center,
                complexity_tau=args.complexity_tau if args.complexity_weighting else None,
                complexity_neighbors=args.complexity_neighbors,
                cluster_balance=args.cluster_balance,
                scene_groups=grad_groups,
                cluster_balance_shuffle=args.cluster_balance_shuffle,
            )
            if resample:
                # Rank 0 computed the weights; every rank must draw the same epoch.
                w = torch.zeros(len(train_set), dtype=torch.float64, device=device)
                if is_main(rank):
                    if scene_weights is None or len(scene_weights) != len(train_set):
                        raise RuntimeError(
                            f"resampling weights cover {None if scene_weights is None else len(scene_weights)} "
                            f"scenes, training set has {len(train_set)}"
                        )
                    w.copy_(torch.from_numpy(scene_weights))
                if world_size > 1:
                    dist.broadcast(w, src=0)
                grad_sampler.set_scene_weights(w.cpu().numpy())
                if is_main(rank):
                    print(
                        "  [resample] next epochs draw "
                        + "  ".join(
                            f"{n}={s:.1%}" for n, s in
                            source_shares(grad_sampler.scene_weights, train_set, sizes, source_names).items()
                        ),
                        flush=True,
                    )
            # Restore training mode after the gradient pass (recompute_clusters
            # leaves the model in train() already, but be explicit).
            ddp_model.train()

    # ---- done ----
    if is_main(rank):
        save_checkpoint(args.out / "final.pt", ddp_model, optimizer, scheduler, step, args)
        print(f"done at step {step}; wrote {args.out / 'final.pt'}")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
