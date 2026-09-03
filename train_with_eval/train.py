#!/usr/bin/env python3
"""Train VGGT-Omega while tracing TracIn-style probe loss reduction vs batch diversity.

Each optimizer step (when --trace-every divides the step) we:

1. Evaluate total loss on a fixed ETH3D probe set *before* the update  -> L_{t-1}
2. Apply the usual training step on the current batch
3. Re-evaluate on the same ETH3D windows *after* the update            -> L_t
4. Record delta_L = L_{t-1} - L_t

The probe set matches `training/evaluate.py` on ETH3D (same roots, split,
repeats, frame sampling). Example:

    python train_with_eval/extract_features.py \\
        --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only \\
        --scannet-root ~/scannet-train --num-frames 8
    python train_with_eval/train.py --data-root ~/dl3dv-train --preset small \\
        --dinov3 checkpoints/dinov3_vits16.pt --out runs/trace-test \\
        --num-frames 8 --max-steps 5000 --trace-every 50 \\
        --eth3d-root ~/eth3d-eval --eth3d-depth-root ~/eth3d-eval/depth \\
        --eth3d-split all --eth3d-repeats 3 \\
        --diversity-features train_with_eval/layer30_features.npz

In parallel we measure **cross-scene** diversity: one layer-30 patch descriptor
per scene, mean pairwise cosine similarity across the scenes in the batch
(use --batch-size 8 or similar; batch_size=1 gives no diversity metric).

Records go to `<out>/trace.jsonl` for offline correlation analysis:

    python train_with_eval/analyze.py runs/my_run/trace.jsonl
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

from train_with_eval.batch_diversity import BatchDiversityTracker, DEFAULT_FINAL_LN  # noqa: E402
from train_with_eval.eth3d_probe import Eth3dProbeConfig, collect_eth3d_probe_batches  # noqa: E402
from training.dl3dv_dataset import DL3DVDataset  # noqa: E402
from training.mixed_dataset import (  # noqa: E402
    TaggedDataset,
    assert_stackable,
    build_concat_trainset,
    collate_mixed,
)
from training.losses import VGGTOmegaLoss, depth_metrics, pose_metrics  # noqa: E402
from training.model_config import build_model, parameter_summary  # noqa: E402
from training.train import (  # noqa: E402
    build_param_groups,
    evaluate_all,
    is_main,
    lr_lambda_factory,
    move_to_device,
    save_checkpoint,
    setup_distributed,
)


@torch.no_grad()
def eth3d_probe_loss(model, probe_batches: list[dict], criterion, device) -> float:
    """Mean VGGTOmegaLoss over fixed ETH3D windows (same objective as evaluate.py)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in probe_batches:
        batch = move_to_device(batch, device)
        predictions = model(batch["images"])
        loss, _ = criterion(predictions, batch)
        if math.isfinite(loss.item()):
            total += loss.item()
            count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


def build_parser() -> argparse.ArgumentParser:
    # Reuse training/train.py's full CLI by importing its parser factory.
    from training import train as base_train  # noqa: WPS433

    p = base_train.build_parser()
    p.description = __doc__
    p.formatter_class = argparse.RawDescriptionHelpFormatter

    g = p.add_argument_group("trace / TracIn-style ETH3D eval")
    g.add_argument(
        "--trace-every",
        type=int,
        default=1,
        help="log ETH3D delta-L and batch diversity every N optimizer steps (0 disables)",
    )
    g.add_argument("--eth3d-root", type=Path, default=None,
                   help="preprocessed ETH3D root (required when --trace-every > 0)")
    g.add_argument("--eth3d-depth-root", type=Path, default=None,
                   help="ETH3D dense depth root, e.g. ~/eth3d-eval/depth")
    g.add_argument("--eth3d-split", default="all", choices=("all", "train", "val"),
                   help="ETH3D split (default: all, same as evaluate.py)")
    g.add_argument("--eth3d-repeats", type=int, default=3,
                   help="frame windows per scene; matches evaluate.py --repeats")
    g.add_argument("--eth3d-seed", type=int, default=1234,
                   help="frame-sampling seed for ETH3D probe windows")
    g.add_argument("--eth3d-sampling", default="covisibility",
                   choices=("covisibility", "contiguous", "random"))
    g.add_argument("--eth3d-num-frames", type=int, default=None,
                   help="frames per ETH3D window (default: --num-frames from training)")
    g.add_argument("--eth3d-max-windows", type=int, default=0,
                   help="cap probe windows for speed (0 = all scenes x repeats)")
    g.add_argument(
        "--no-diversity",
        action="store_true",
        help="skip DINOv3 patch-token diversity (faster; delta-L only)",
    )
    g.add_argument(
        "--diversity-model",
        default="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        help="HF DINOv3 for batch diversity (matches patch_avg_clustering default)",
    )
    g.add_argument(
        "--diversity-layers",
        type=int,
        nargs="+",
        default=[30],
        metavar="L",
        help="hidden-state indices for patch-token means (default: 30)",
    )
    g.add_argument(
        "--diversity-device",
        default="cpu",
        help="device for the diversity DINOv3 (cpu avoids competing with training GPU)",
    )
    g.add_argument(
        "--diversity-max-frames",
        type=int,
        default=4,
        help="frames sampled per scene when building its descriptor (matches clustering n_images)",
    )
    g.add_argument(
        "--diversity-features",
        type=Path,
        nargs="+",
        default=None,
        metavar="NPZ",
        help="precomputed layer-30 npz from train_with_eval/extract_features.py "
             "(skips the live ViT-7B forward)",
    )
    g.add_argument(
        "--final-ln",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FINAL_LN,
        help="apply DINOv3's final layernorm to patch tokens before averaging "
             "(live diversity only; npz already bakes this in). Default on; "
             "--no-final-ln for raw hidden states",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    model = build_model(args.preset, use_checkpoint=args.checkpointing, dinov3_checkpoint=args.dinov3).to(device)

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

    if args.data_root is None and args.scannet_root is None:
        raise SystemExit("pass --data-root (DL3DV), --scannet-root (ScanNet), or both")

    def make_split(split: str, *, augment: bool, seed: int | None, sampling: str) -> dict:
        built = {}
        sources = (
            ("dl3dv", args.data_root, args.depth_root, args.dense_only),
            (
                "scannet",
                args.scannet_root,
                args.scannet_depth_root or (args.scannet_root / "depth" if args.scannet_root else None),
                False,
            ),
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
                if split == "train":
                    raise
                if is_main(rank):
                    print(f"[{name}] no {split} split ({exc}); skipping its {split} loader")
        return built

    train_parts = make_split("train", augment=True, seed=None, sampling=args.sampling)

    for name, dataset in train_parts.items():
        if args.scene_list is not None:
            wanted = set(args.scene_list.read_text().split())
            dataset.scenes = [e for e in dataset.scenes if f"{e['subset']}/{e['scene']}" in wanted]
            if is_main(rank):
                print(f"[scene-list] {name}: {len(dataset.scenes)}/{len(wanted)} listed scenes kept")
        cap = {"dl3dv": args.dl3dv_scenes, "scannet": args.scannet_scenes}.get(name, 0)
        if cap and cap < len(dataset.scenes):
            dataset.scenes = dataset.scenes[:cap]
            if is_main(rank):
                print(f"[{name}] capped to the first {len(dataset.scenes)} train scenes")
        if args.overfit:
            dataset.scenes = dataset.scenes[: args.overfit]
            dataset.seed = 0
            dataset.augment = False
    train_parts = {n: d for n, d in train_parts.items() if len(d.scenes) > 0}
    if not train_parts:
        raise SystemExit("no training scenes survived the dataset filters")

    image_hw = assert_stackable(train_parts, args.batch_size)

    weights = {"dl3dv": args.dl3dv_weight, "scannet": args.scannet_weight}
    train_set, source_names, sizes, epoch_counts = build_concat_trainset(train_parts, weights, seed=args.seed)

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
        for name, dataset in make_split("val", augment=False, seed=1234, sampling="covisibility").items():
            val_loaders[name] = DataLoader(
                TaggedDataset(dataset, name),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(args.workers // 2, 1),
                pin_memory=True,
                collate_fn=collate_mixed,
            )

    probe_batches: list[dict] = []
    probe_meta: dict | None = None
    diversity_tracker: BatchDiversityTracker | None = None
    if is_main(rank) and args.trace_every:
        if args.eth3d_root is None:
            raise SystemExit("--eth3d-root is required when --trace-every > 0")
        eth3d_cfg = Eth3dProbeConfig(
            data_root=args.eth3d_root,
            depth_root=args.eth3d_depth_root,
            split=args.eth3d_split,
            num_frames=args.eth3d_num_frames,
            resolution=args.resolution,
            sampling=args.eth3d_sampling,
            dense_only=False,
            repeats=args.eth3d_repeats,
            seed=args.eth3d_seed,
            max_windows=args.eth3d_max_windows,
            batch_size=1,
        )
        probe_batches, probe_meta = collect_eth3d_probe_batches(
            eth3d_cfg,
            train_num_frames=args.num_frames,
            train_resolution=args.resolution,
        )
        print(
            f"[trace] ETH3D probe: {probe_meta['n_windows']} windows "
            f"({probe_meta['n_scenes']} scenes x {probe_meta['repeats']} repeats, "
            f"{probe_meta['num_frames']} frames, hw={probe_meta['image_hw']})"
        )

        if not args.no_diversity:
            if args.diversity_features:
                diversity_tracker = BatchDiversityTracker.from_npz(
                    args.diversity_features,
                    layers=args.diversity_layers,
                )
                meta = diversity_tracker.cache_meta or {}
                npz_ln = bool(meta.get("final_ln", False))
                if npz_ln != args.final_ln:
                    print(
                        f"[trace] warning: npz has final_ln={npz_ln} but "
                        f"--{'final-ln' if args.final_ln else 'no-final-ln'} was set; "
                        "using the npz (re-extract to change it)",
                        flush=True,
                    )
                print(
                    f"[trace] cross-scene diversity: {meta.get('n_scenes', 0)} cached scenes "
                    f"layers {list(meta.get('layers', args.diversity_layers))} "
                    f"{'final-ln' if npz_ln else 'no-final-ln'} "
                    f"from {', '.join(meta.get('paths', []))}  "
                    f"(needs batch_size >= 2, got {args.batch_size})"
                )
            else:
                diversity_tracker = BatchDiversityTracker(
                    model_id=args.diversity_model,
                    layers=args.diversity_layers,
                    device=args.diversity_device,
                    max_frames=args.diversity_max_frames,
                    final_ln=args.final_ln,
                )
                print(
                    f"[trace] cross-scene diversity: {args.diversity_model} layers {args.diversity_layers} "
                    f"{'final-ln' if args.final_ln else 'no-final-ln'} "
                    f"on {args.diversity_device}  (needs batch_size >= 2, got {args.batch_size}; "
                    f"pass --diversity-features to skip the live 7B forward)"
                )

    if is_main(rank):
        mix = "  ".join(
            f"{n}={s}" + (f"(x{c / s:.2g})" if c != s else "") for n, s, c in zip(source_names, sizes, epoch_counts)
        )
        val_desc = (
            "  val " + " ".join(f"{n}={len(l.dataset)}" for n, l in val_loaders.items()) if val_loaders else "  (no val)"
        )
        print(
            f"train samples/epoch={len(train_set)} [{mix}]{val_desc}"
            + f"  {image_hw[0]}x{image_hw[1]}  frames/sample={args.num_frames}"
            + f"  world_size={world_size}  trace_every={args.trace_every}"
        )

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
    trace_path = args.out / "trace.jsonl"
    writer = None
    if is_main(rank):
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps(vars(args), indent=1, default=str))
        if probe_meta is not None:
            (args.out / "eth3d_probe.json").write_text(json.dumps(probe_meta, indent=1))
        writer = SummaryWriter(log_dir=str(args.out / "tb"))

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
            sync_context = ddp_model.no_sync() if (world_size > 1 and not is_last_micro) else nullcontext()

            train_loss_value = None
            with sync_context:
                predictions = ddp_model(batch["images"])
                loss, logs = criterion(predictions, batch)
                if is_last_micro:
                    train_loss_value = loss.item()
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

            do_trace = (
                is_main(rank)
                and args.trace_every
                and (step + 1) % args.trace_every == 0
                and probe_batches
            )
            probe_before = eth3d_probe_loss(model, probe_batches, criterion, device) if do_trace else None

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            running["grad_norm"] = running.get("grad_norm", 0.0) + grad_norm.item()
            running_count += 1

            if do_trace:
                probe_after = eth3d_probe_loss(model, probe_batches, criterion, device)
                delta_l = probe_before - probe_after
                record: dict[str, object] = {
                    "step": step,
                    "loss_eth3d_before": round(probe_before, 6),
                    "loss_eth3d_after": round(probe_after, 6),
                    "delta_loss_eth3d": round(delta_l, 6),
                    # legacy keys for older analyze scripts
                    "loss_probe_before": round(probe_before, 6),
                    "loss_probe_after": round(probe_after, 6),
                    "delta_loss_probe": round(delta_l, 6),
                    "train_loss": round(train_loss_value, 6) if train_loss_value is not None else None,
                    "batch_size": args.batch_size,
                    "num_frames": args.num_frames,
                    "eth3d_windows": probe_meta["n_windows"] if probe_meta else None,
                    "scene_id": batch.get("scene_id"),
                    "dataset": batch.get("dataset"),
                }
                if diversity_tracker is not None:
                    div = diversity_tracker.from_batch(batch)
                    record.update({k: round(v, 6) if isinstance(v, float) else v for k, v in div.items()})
                with trace_path.open("a") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
                if writer is not None:
                    writer.add_scalar("trace/delta_loss_eth3d", delta_l, step)
                    writer.add_scalar("trace/loss_eth3d_before", probe_before, step)
                    writer.add_scalar("trace/loss_eth3d_after", probe_after, step)
                    if "avg_cos_sim" in record and isinstance(record["avg_cos_sim"], float):
                        writer.add_scalar("trace/avg_cos_sim", record["avg_cos_sim"], step)
                        writer.add_scalar("trace/diversity", record.get("diversity", float("nan")), step)
                div_str = ""
                if "avg_cos_sim" in record:
                    div_str = f"  cos {record['avg_cos_sim']:.3f}  div {record.get('diversity', float('nan')):.3f}"
                print(
                    f"  [trace @ {step}]  dL_eth3d {delta_l:+.5f}  "
                    f"L {probe_before:.4f}->{probe_after:.4f}{div_str}",
                    flush=True,
                )

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
                    "lr": max(lrs),
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
        if trace_path.exists():
            print(f"trace log: {trace_path}")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
