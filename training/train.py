#!/usr/bin/env python3
"""Train VGGT-Omega from scratch on preprocessed DL3DV.

Single GPU:

    python training/train.py --data-root ~/dl3dv-train --preset small \
        --dinov3 checkpoints/dinov3_vits16.pt --out runs/small \
        --num-frames 16 --max-steps 100000 --checkpointing

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dl3dv_dataset import DL3DVDataset, collate_scenes  # noqa: E402
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
    p.add_argument("--data-root", type=Path, required=True, help="output of preprocess_dl3dv.py")
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
    p.add_argument("--clip-grad", type=float, default=1.0)

    p.add_argument("--weight-camera", type=float, default=5.0)
    p.add_argument("--weight-depth", type=float, default=1.0)
    p.add_argument("--weight-gradient", type=float, default=0.0, help="needs dense depth; useless on sparse GT")

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=2000, help="0 disables validation")
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", type=int, default=0, help="train on N scenes only; loss should go to ~0")
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
        weight_gradient=args.weight_gradient,
    )

    # ---- data ----
    train_set = DL3DVDataset(
        args.data_root,
        split="train",
        num_frames=args.num_frames,
        resolution=args.resolution,
        sampling=args.sampling,
        augment=True,
    )
    if args.overfit:
        train_set.scenes = train_set.scenes[: args.overfit]
        train_set.seed = 0  # deterministic frame choice, so the target is fixed
        train_set.augment = False

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
        collate_fn=collate_scenes,
    )

    val_loader = None
    if args.val_every and not args.overfit:
        val_set = DL3DVDataset(
            args.data_root,
            split="val",
            num_frames=args.num_frames,
            resolution=args.resolution,
            sampling="covisibility",
            augment=False,
            seed=1234,  # fixed frame selection so val numbers are comparable across steps
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(args.workers // 2, 1),
            pin_memory=True,
            collate_fn=collate_scenes,
        )

    if is_main(rank):
        print(
            f"train scenes={len(train_set)}"
            + (f"  val scenes={len(val_loader.dataset)}" if val_loader else "  (no val)")
            + f"  frames/sample={args.num_frames}  world_size={world_size}"
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
    if is_main(rank):
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps(vars(args), indent=1, default=str))

    # ---- loop ----
    ddp_model.train()
    step = start_step  # optimiser steps, not micro-batches
    micro = 0
    epoch = 0
    running: dict[str, float] = {}
    running_count = 0
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
                    f"lr {record['lr']:.2e}  {record['steps_per_sec']:.2f} it/s",
                    flush=True,
                )
                with log_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                running, running_count, started = {}, 0, time.time()

            if val_loader is not None and args.val_every and step % args.val_every == 0:
                metrics = evaluate(model, val_loader, criterion, device, args.val_batches)
                if is_main(rank):
                    print(
                        f"  [val @ {step}] loss {metrics.get('loss', float('nan')):.4f}  "
                        f"AUC@30 {metrics.get('auc_at_30', float('nan')):.3f}  "
                        f"rot_err {metrics.get('rot_err_deg_median', float('nan')):.2f}deg  "
                        f"abs_rel {metrics.get('abs_rel', float('nan')):.3f}",
                        flush=True,
                    )
                    with log_path.open("a") as fh:
                        fh.write(json.dumps({"step": step, "split": "val", **metrics}) + "\n")

            if is_main(rank) and args.save_every and step % args.save_every == 0:
                save_checkpoint(args.out / "latest.pt", ddp_model, optimizer, scheduler, step, args)

    if is_main(rank):
        save_checkpoint(args.out / "final.pt", ddp_model, optimizer, scheduler, step, args)
        print(f"done at step {step}; wrote {args.out / 'final.pt'}")

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
