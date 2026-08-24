#!/usr/bin/env python3
"""Train VGGT-Omega from scratch on preprocessed DL3DV and/or ScanNet v2.

Single GPU:

    python training/train.py --data-root ~/dl3dv-train --preset small \
        --dinov3 checkpoints/dinov3_vits16.pt --out runs/small \
        --num-frames 16 --max-steps 100000 --checkpointing

Both datasets at once (see `mixed_dataset.py`):

    python training/train.py \
        --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth \
        --scannet-root ~/scannet-train \
        --dl3dv-weight 1 --scannet-weight 1 --preset small --out runs/mixed

At 1/1 the mix is proportional to scene count, so DL3DV's ~4900 train scenes
dominate ScanNet's ~1500. There are two ways to even that out, and they are not
the same experiment:

    --scannet-weight 3      ScanNet's 1500 scenes are each seen 3x per epoch
    --dl3dv-scenes 1500     the first 1500 DL3DV scenes, the rest never seen

The first keeps all the data and repeats the smaller set; the second keeps the
step count per scene equal and throws data away. Prefer the first unless the
point is to hold DL3DV's *diversity* down. `--dl3dv-weight 0.5` is a third
option, but it picks its half at random under --seed rather than taking a
prefix, so it is not reproducible across a --seed change.

Loss and val metrics are reported per dataset as well as pooled. Both sets must
be preprocessed at the same resolution, since a batch spanning two shapes cannot
be stacked -- see `assert_stackable` in `mixed_dataset.py`.

Multi GPU:

    torchrun --nproc_per_node=4 training/train.py --data-root ... (same flags)

Sanity check before committing to a long run -- overfit a handful of scenes and
watch the loss collapse:

    python training/train.py --data-root ~/dl3dv-train --preset small \
        --overfit 2 --num-frames 8 --max-steps 200 --log-every 20 --val-every 0

Notes on choices that are easy to get wrong:

* The model autocasts to bf16 internally (`VGGTOmega.forward`) and forces fp32
  for the heads, so this script does *not* wrap the forward in another autocast
  and does not use a GradScaler -- bf16 does not need loss scaling.
* Weight decay is applied only to >=2D parameters. Norm weights, biases,
  LayerScale gammas and the camera/register tokens are excluded; decaying those
  measurably hurts ViT training.
* The DINOv3 trunk is pretrained, so it gets its own lower learning rate
  (`--backbone-lr-mult`).
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

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dl3dv_dataset import DL3DVDataset  # noqa: E402
from training.mixed_dataset import (  # noqa: E402
    TaggedDataset,
    assert_stackable,
    build_concat_trainset,
    collate_mixed,
)
from training.losses import VGGTOmegaLoss, depth_metrics, pose_metrics  # noqa: E402
from training.model_config import build_model, parameter_summary  # noqa: E402


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #


def setup_distributed() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Works unmodified under torchrun."""
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
    """Four groups: {backbone, rest} x {decay, no-decay}."""
    groups = {
        ("backbone", True): {"params": [], "weight_decay": weight_decay, "lr": lr * backbone_lr_mult},
        ("backbone", False): {"params": [], "weight_decay": 0.0, "lr": lr * backbone_lr_mult},
        ("head", True): {"params": [], "weight_decay": weight_decay, "lr": lr},
        ("head", False): {"params": [], "weight_decay": 0.0, "lr": lr},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "aggregator.patch_embed" in name
        # 1-D parameters are norms, biases, LayerScale gammas and the learned
        # camera/register tokens -- none of these should be decayed.
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
    """`evaluate` per dataset, flattened to `<name>/<metric>` plus a plain mean.

    Kept separate rather than pooled because the two datasets have genuinely
    different depth: ScanNet's is a metric sensor at ~95% coverage, DL3DV's is
    estimated. A single averaged abs_rel would hide one regressing while the
    other improves.
    """
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


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="output of preprocess_dl3dv.py")
    p.add_argument("--depth-root", type=Path, default=None, help="output of fetch_dl3dv_depth.py (dense GT)")
    p.add_argument("--scannet-root", type=Path, default=None, help="output of preprocess_scannet.py")
    p.add_argument("--scannet-depth-root", type=Path, default=None,
                   help="dense ScanNet depth; defaults to <scannet-root>/depth")
    p.add_argument("--dl3dv-weight", type=float, default=1.0,
                   help="times each DL3DV scene is listed per epoch (1.0 = once)")
    p.add_argument("--scannet-weight", type=float, default=1.0,
                   help="times each ScanNet scene is listed per epoch. At 1/1 the mix is simply\n"
                        "proportional to scene count; raise this to give the smaller set more weight")
    p.add_argument("--dense-only", action="store_true", help="drop scenes with no dense depth (~12%%)")
    p.add_argument("--out", type=Path, default=Path("runs/vggt-omega-small"))

    p.add_argument("--preset", default="small", choices=("small", "base", "large"))
    p.add_argument("--dinov3", type=Path, default=None, help="converted DINOv3 trunk (convert_dinov3.py)")
    p.add_argument("--checkpointing", action="store_true", help="activation checkpointing (~4x less memory)")
    p.add_argument("--freeze-backbone-steps", type=int, default=0)

    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=None, help="resize at load time; default keeps stored size")
    p.add_argument("--sampling", default="covisibility", choices=("covisibility", "contiguous", "random"))
    p.add_argument("--batch-size", type=int, default=1, help="scenes per step per GPU")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    # After warmup, pre-clip grad norms sit at ~11 (p99 ~17, max ~20 on
    # small-v15). A clip of 1.0 renormalises every step; 10.0 still rescales the
    # typical ~11-norm update and heavily cuts warmup (norms 13-31). 50.0 leaves
    # both alone and only catches genuine spikes (~180, twice in 38k steps on
    # small-v3).
    p.add_argument("--clip-grad", type=float, default=50.0)

    p.add_argument("--weight-camera", type=float, default=5.0)
    p.add_argument("--weight-depth", type=float, default=1.0)
    p.add_argument("--weight-point", type=float, default=0.5, help="L_point on unprojected depth")
    p.add_argument("--weight-gradient", type=float, default=1.0,
                   help="the ||c*grad(e)|| sub-term; set 0 without --depth-root, where it is noise")
    p.add_argument("--conf-alpha", type=float, default=0.2,
                   help="the -alpha*log(c) term in L_depth and L_point. Below ~2.0 the confidence "
                        "head saturates at its floor of 1.0 and depth_conf is not a usable "
                        "uncertainty output; see losses.depth_loss for why 0.2 is still the default")

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=2000, help="0 disables validation")
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", type=int, default=0, help="train on N scenes only; loss should go to ~0")
    p.add_argument("--scene-list", type=Path, default=None,
                   help="train on only the `subset/scene` lines in this file (clustering/subset.py)")
    p.add_argument("--dl3dv-scenes", type=int, default=0,
                   help="train on only the first N DL3DV scenes (0 = all). The index order is\n"
                        "arbitrary, so this is a sample, not a filter -- but note it is a prefix,\n"
                        "so N < 976 stays inside the 1K subset")
    p.add_argument("--scannet-scenes", type=int, default=0, help="cap ScanNet at N train scenes (0 = all)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    # ---- model ----
    # Every rank loads the trunk, so all replicas start from identical weights.
    model = build_model(args.preset, use_checkpoint=args.checkpointing, dinov3_checkpoint=args.dinov3).to(device)

    if is_main(rank):
        summary = parameter_summary(model)
        print(f"preset={args.preset}  " + "  ".join(f"{k}={v:.1f}M" for k, v in summary.items()))

    ddp_model = model
    if world_size > 1:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            # A frozen backbone leaves its parameters out of the graph, which
            # trips DDP's default all-params-were-used assertion.
            find_unused_parameters=args.freeze_backbone_steps > 0,
        )

    criterion = VGGTOmegaLoss(
        weight_camera=args.weight_camera,
        weight_depth=args.weight_depth,
        weight_point=args.weight_point,
        weight_gradient=args.weight_gradient,
        depth_kwargs={"alpha": args.conf_alpha},
    )

    # ---- data ----
    # DL3DV and ScanNet share one loader class because `preprocess_scannet.py`
    # writes the same on-disk contract; only the roots differ. Preprocess ScanNet
    # with `--target-hw 224 384 --fit crop` and the two are the same shape, so
    # mixing them needs nothing more than a ConcatDataset.
    if args.data_root is None and args.scannet_root is None:
        raise SystemExit("pass --data-root (DL3DV), --scannet-root (ScanNet), or both")

    def make_split(split: str, *, augment: bool, seed: int | None, sampling: str) -> dict:
        built = {}
        sources = (
            ("dl3dv", args.data_root, args.depth_root, args.dense_only),
            # ScanNet depth always exists, so `dense_only` would be a no-op that
            # only risks dropping scenes over a missing flag.
            ("scannet", args.scannet_root, args.scannet_depth_root or
             (args.scannet_root / "depth" if args.scannet_root else None), False),
        )
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
                    sampling=sampling,
                    augment=augment,
                    seed=seed,
                    depth_root=depth_root,
                    dense_only=dense_only,
                )
            except ValueError as exc:
                # An empty val split is normal for a set preprocessed without a
                # holdout; an empty *train* split is not.
                if split == "train":
                    raise
                if is_main(rank):
                    print(f"[{name}] no {split} split ({exc}); skipping its {split} loader")
        return built

    train_parts = make_split("train", augment=True, seed=None, sampling=args.sampling)

    for name, dataset in train_parts.items():
        if args.scene_list is not None:
            # After construction, so the dataset's own filters (dense_only,
            # num_frames, image size) still apply -- a listed scene they drop
            # stays dropped.
            wanted = set(args.scene_list.read_text().split())
            dataset.scenes = [e for e in dataset.scenes if f"{e['subset']}/{e['scene']}" in wanted]
            if is_main(rank):
                print(f"[scene-list] {name}: {len(dataset.scenes)}/{len(wanted)} listed scenes kept")
        cap = {"dl3dv": args.dl3dv_scenes, "scannet": args.scannet_scenes}.get(name, 0)
        if cap and cap < len(dataset.scenes):
            # A plain prefix. DL3DV's 1K..5K subsets are release batches, not
            # strata -- scene content is already unordered within and across
            # them -- so the first N is as good a sample as a shuffle, and it has
            # the advantage of being the same N on every rank and every rerun
            # without depending on a seed.
            dataset.scenes = dataset.scenes[:cap]
            if is_main(rank):
                print(f"[{name}] capped to the first {len(dataset.scenes)} train scenes")
        if args.overfit:
            dataset.scenes = dataset.scenes[: args.overfit]
            dataset.seed = 0  # deterministic frame choice, so the target is fixed
            dataset.augment = False
    train_parts = {n: d for n, d in train_parts.items() if len(d.scenes) > 0}
    if not train_parts:
        raise SystemExit("no training scenes survived the dataset filters")

    # Fails here rather than mid-epoch inside collate_scenes if the two sets were
    # preprocessed at different resolutions.
    image_hw = assert_stackable(train_parts, args.batch_size)

    weights = {"dl3dv": args.dl3dv_weight, "scannet": args.scannet_weight}
    train_set, source_names, sizes, epoch_counts = build_concat_trainset(
        train_parts, weights, seed=args.seed
    )

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_mixed,
    )

    val_loaders: dict[str, DataLoader] = {}
    if args.val_every and not args.overfit:
        for name, dataset in make_split(
            "val", augment=False, seed=1234, sampling="covisibility"
        ).items():
            # seed=1234 above fixes frame selection, so val numbers are
            # comparable across steps.
            val_loaders[name] = DataLoader(
                TaggedDataset(dataset, name),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(args.workers // 2, 1),
                pin_memory=True,
                collate_fn=collate_mixed,
            )

    if is_main(rank):
        mix = "  ".join(
            f"{n}={s}" + (f"(x{c / s:.2g})" if c != s else "")
            for n, s, c in zip(source_names, sizes, epoch_counts)
        )
        val_desc = ("  val " + " ".join(f"{n}={len(l.dataset)}" for n, l in val_loaders.items())
                    if val_loaders else "  (no val)")
        print(
            f"train samples/epoch={len(train_set)} [{mix}]{val_desc}"
            + f"  {image_hw[0]}x{image_hw[1]}  frames/sample={args.num_frames}"
            + f"  world_size={world_size}"
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

    # ---- loop ----
    ddp_model.train()
    step = start_step  # optimiser steps, not micro-batches
    micro = 0
    epoch = 0
    running: dict[str, float] = {}
    running_count = 0
    per_source: dict[str, float] = {}
    per_source_count: dict[str, int] = {}
    started = time.time()

    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1

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
            # Skip the gradient all-reduce on every micro-batch but the last.
            sync_context = ddp_model.no_sync() if (world_size > 1 and not is_last_micro) else nullcontext()
            with sync_context:
                predictions = ddp_model(batch["images"])
                loss, logs = criterion(predictions, batch)
                (loss / args.grad_accum).backward()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item() / args.grad_accum
            # Track the realised mix. The *share* is always exact, but a
            # per-dataset loss is only meaningful when a batch cannot span the
            # two: the criterion returns one scalar for the whole batch, so
            # splitting it would report the same number for both datasets and
            # look like a breakdown while carrying no information. At
            # --batch-size 1 each batch is one dataset and it is exact.
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
            # One scheduler step per *optimiser* step, so --grad-accum does not
            # silently run the schedule N times too fast.
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
                    "lr": max(lrs),  # head groups
                    "lr_backbone": min(lrs),
                    "steps_per_sec": args.log_every / max(elapsed, 1e-9),
                    **{k: round(v, 5) for k, v in means.items()},
                }
                print(
                    f"step {step:>7d}/{args.max_steps}  loss {means.get('loss', 0):.4f}  "
                    f"cam {means.get('loss_camera', 0):.4f}  depth {means.get('loss_depth', 0):.4f}  "
                    f"point {means.get('loss_point', 0):.4f}  cover {means.get('depth_valid', 0) * 100:.1f}%  "
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
                        get = lambda k: metrics.get(f"{prefix}{k}", nan)  # noqa: E731
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
                save_checkpoint(args.out / "latest.pt", ddp_model, optimizer, scheduler, step, args)

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
