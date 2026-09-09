#!/usr/bin/env python3
"""DoReMi over the patch-average clusters: the reference, proxy and final runs.

Three runs of the same trainer, in this order (see `doremi/README.md` for the full
commands):

    --mode reference   train the baseline-mixture model the excess loss is measured
                       against. Identical to `training/train.py` except that the
                       training pool is pinned to the clusters json, so the reference
                       and the proxy see the same scenes.
    --mode proxy       train a small model with Group DRO on the domain weights,
                       against `--reference`. Writes `domain_weights.json`.
    --mode final       train the model you actually want, sampling scenes from the
                       learned mixture in `--domain-weights`.

What each step of the proxy run does, on top of a normal step:

1. per-scene losses for the proxy and for the frozen reference on the *same* batch
   (`per_sample_loss.py` -- the criterion evaluated one scene at a time),
2. `max(0, L_proxy - L_ref)` accumulated per domain over the whole
   `--grad-accum` window, all-reduced across ranks,
3. one exponentiated-gradient update of the domain weights (`dro.py`),
4. a backward pass on `sum_j alpha_j * mean_j(L_proxy)` instead of the batch mean.

The proxy step therefore costs one extra forward pass. `--reference-device cpu`
trades speed for memory if the reference will not fit alongside the proxy;
`--reference-losses` skips the reference model entirely and reads cached per-scene
losses from an `evaluate.py` jsonl, at the cost of comparing the proxy against a
*different frame window* than the one it just trained on.

Everything except the data pool and the loop body is imported from
`training/train.py` -- optimiser groups, schedule, validation, checkpointing -- so
this stays a variant of that trainer rather than a fork of it. `build_train_parts`
below is the exception: `make_split` is nested inside its `main()`, so it is
restated here.

Sanity check with no GPU time, before either real run:

    python3 doremi/train.py --dry-run --scannet-root ~/scannet-train \\
        --data-root ~/dl3dv-train --mode proxy --reference runs/ref/final.pt
"""

from __future__ import annotations

import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doremi.domains import format_table, load_domains, load_weights, save_weights, scene_factors  # noqa: E402
from doremi.dro import DomainReweighter, auto_eta, effective_sample_size  # noqa: E402
from doremi.per_sample_loss import domain_weighted_loss, excess_losses, per_sample_losses  # noqa: E402
from doremi.sampling import WeightedEpochSampler, concat_scene_keys, index_weights  # noqa: E402
from training.dl3dv_dataset import DL3DVDataset  # noqa: E402
from training.losses import VGGTOmegaLoss  # noqa: E402
from training.mixed_dataset import (  # noqa: E402
    TaggedDataset,
    assert_stackable,
    build_concat_trainset,
    collate_mixed,
)
from training.model_config import build_model, parameter_summary  # noqa: E402
from training.train import (  # noqa: E402
    build_param_groups,
    build_parser as base_parser,
    evaluate_all,
    is_main,
    lr_lambda_factory,
    move_to_device,
    save_checkpoint,
    setup_distributed,
)


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def build_parser():
    p = base_parser()
    p.description = __doc__

    g = p.add_argument_group("doremi")
    g.add_argument("--mode", default="proxy", choices=("reference", "proxy", "final"),
                   help="which of the three runs this is")
    g.add_argument("--clusters", type=Path, default=None,
                   help="cluster.py json defining the domains (default: patch_avg_clustering's)")
    g.add_argument("--min-domain-size", type=int, default=10,
                   help="drop clusters with fewer scenes than this, and their scenes with them.\n"
                        "A cluster of 2 cannot support a weight: its excess loss is noise, and\n"
                        "max(0, .) rectifies noise upward, so it runs away (see dro.py)")
    g.add_argument("--reference", type=Path, default=None,
                   help="[proxy] checkpoint of the --mode reference run")
    g.add_argument("--reference-losses", type=Path, default=None,
                   help="[proxy] instead of a reference model, cached per-scene losses from\n"
                        "`evaluate.py --out` (windows-*.jsonl). Cheaper and worse: the cached\n"
                        "loss is for a different frame window than the proxy just saw")
    g.add_argument("--reference-loss-key", default="loss",
                   help="which jsonl field to read as the reference loss")
    g.add_argument("--reference-device", default=None,
                   help="[proxy] where to hold the reference model (default: the training device)")
    g.add_argument("--domain-weights", type=Path, default=None,
                   help="[final] a domain_weights.json from the proxy run")

    g.add_argument("--dro-lr", default="auto",
                   help="eta, the domain weight step size, in units of 1/excess-loss. `auto`\n"
                        "(the default) picks it from --dro-budget and the measured spread of the\n"
                        "excess loss each step, because a fixed eta is not transferable across\n"
                        "loss scales: DoReMi's 1.0 is calibrated to per-token cross-entropy and\n"
                        "saturates this objective's weights within a few steps (see dro.auto_eta)")
    g.add_argument("--dro-budget", type=float, default=4.0,
                   help="[--dro-lr auto] a domain that stays one cross-domain standard deviation\n"
                        "above average for the whole run ends at this multiple of its prior share")
    g.add_argument("--dro-smoothing", type=float, default=1e-3,
                   help="c: mass mixed back toward --dro-prior every step")
    g.add_argument("--dro-ema", type=float, default=0.9,
                   help="decay of each domain's excess-loss estimate. Domains absent from a step\n"
                        "keep their last estimate rather than reading as zero-excess")
    g.add_argument("--dro-prior", default="baseline", choices=("baseline", "uniform"),
                   help="what the weights are initialised at and smoothed toward. `baseline` is\n"
                        "proportional to cluster size, i.e. the mixture a plain run trains on")
    g.add_argument("--max-ratio", type=float, default=4.0,
                   help="cap on alpha / baseline, both ways. 0 disables (DoReMi's own behaviour)")
    g.add_argument("--dro-start-step", type=int, default=0,
                   help="train the proxy this many steps before the weights start moving.\n"
                        "Absolute, so on a --resume it is the resumed step count plus the\n"
                        "warmup you want: the excess-loss EMA is filled by `observe` below it\n"
                        "and the weights only move above it, with --dro-budget spread over\n"
                        "(--max-steps - --dro-start-step) updates rather than over the run")
    g.add_argument("--reset-dro", action="store_true",
                   help="[proxy + --resume] start the mixture from --dro-prior instead of the\n"
                        "checkpoint's reweighter state. Needed to resume a *finished* proxy for\n"
                        "a fresh DRO phase: `alpha_bar` is a running mean, so a checkpoint with\n"
                        "30k updates in it moves by 1/30001 per new step and the mixture cannot\n"
                        "leave where it was. Also drops the excess-loss EMA, which is the point\n"
                        "when the old estimates were measured under a different --dro-objective")
    g.add_argument("--constant-lr", type=float, default=None,
                   help="[--resume] hold the learning rate flat at this value for the whole\n"
                        "continuation, keeping --backbone-lr-mult's ratio between groups.\n"
                        "Resuming re-derives the cosine from the *new* --max-steps, so step\n"
                        "30000 of a 33500-step schedule gets ~4x the LR the original\n"
                        "30000-step run ended at -- a bump landing exactly where the excess\n"
                        "loss is being measured. Pass the LR the run ended on")
    g.add_argument("--dro-log-every", type=int, default=200,
                   help="append the weight vector to domain_weights.jsonl this often")
    g.add_argument("--dro-objective", default="scene-ratio",
                   choices=("scene-ratio", "domain-mean"),
                   help="[proxy] how alpha weights the objective. `scene-ratio` weights each\n"
                        "scene by alpha_j/baseline_j, so a baseline mixture is exactly the\n"
                        "reference run's objective and a scene's weight does not depend on its\n"
                        "cluster's size. `domain-mean` is DoReMi's literal per-domain mean and\n"
                        "is what runs/doremi-proxy used -- it makes the per-scene gradient\n"
                        "weight proportional to cluster size (3.2x over these 48 clusters), so\n"
                        "the proxy undertrains small clusters and the reweighter then reads the\n"
                        "gap it created as excess loss. Kept only to reproduce that run")
    g.add_argument("--excess-clamp", default="none", choices=("scene", "none"),
                   help="[proxy] how to turn L_proxy - L_ref into the excess loss DRO sees.\n"
                        "`scene` is DoReMi's literal per-scene max(0, .) and is what\n"
                        "runs/doremi-proxy-raw used; on that run the clamped domain excess was\n"
                        "90%% predicted by the *variance* of the per-window difference alone\n"
                        "(corr +0.95), because rectifying a difference centred near zero turns\n"
                        "E[max(0,d)] into a measure of sd(d) -- which in turn tracks the\n"
                        "domain's loss level, hence the dataset. `none` passes the signed\n"
                        "difference through. The exponentiated-gradient update is shift\n"
                        "invariant, so a proxy that is uniformly worse costs nothing here and\n"
                        "only genuine per-domain structure moves the mixture")
    g.add_argument("--per-sample-objective", action="store_true",
                   help="[reference/final] weight scenes equally rather than valid pixels, i.e.\n"
                        "use the same normalisation the proxy objective uses")
    g.add_argument("--dry-run", action="store_true",
                   help="build the domains, pool, sampler and mixture, print them, and stop")
    return p


def resolve_args(args) -> None:
    """Reject mode/flag combinations that would otherwise fail deep into a run."""
    if isinstance(args.dro_lr, str):
        if args.dro_lr == "auto":
            args.dro_lr = None
        else:
            try:
                args.dro_lr = float(args.dro_lr)
            except ValueError:
                raise SystemExit(f"--dro-lr must be a number or 'auto', got {args.dro_lr!r}") from None
    if args.dro_budget <= 1.0:
        raise SystemExit("--dro-budget is a ratio and must be > 1")
    if args.mode == "proxy":
        if (args.reference is None) == (args.reference_losses is None):
            raise SystemExit("--mode proxy needs exactly one of --reference / --reference-losses")
        if args.reference is not None and not args.reference.exists():
            raise SystemExit(f"--reference {args.reference} does not exist")
        if args.reference_losses is not None and not args.reference_losses.exists():
            raise SystemExit(f"--reference-losses {args.reference_losses} does not exist")
    if args.mode == "final":
        if args.domain_weights is None:
            raise SystemExit("--mode final needs --domain-weights (the proxy run's json)")
        if not args.domain_weights.exists():
            raise SystemExit(f"--domain-weights {args.domain_weights} does not exist")
    if args.mode != "proxy" and (args.reference or args.reference_losses):
        print(f"[doremi] note: --mode {args.mode} ignores the reference")
    if args.resume is None:
        for flag, value in (("--reset-dro", args.reset_dro), ("--constant-lr", args.constant_lr)):
            if value:
                raise SystemExit(f"{flag} only means anything with --resume")
    if args.reset_dro and args.mode != "proxy":
        raise SystemExit("--reset-dro only applies to --mode proxy")
    if args.mode == "proxy" and args.dro_objective == "domain-mean":
        print("[doremi] warning: --dro-objective domain-mean makes a scene's gradient weight\n"
              "          proportional to its cluster's size (3.2x over these 48 clusters), which\n"
              "          is what made runs/doremi-proxy learn cluster size instead of hardness.\n"
              "          Use it only to reproduce that run.")
    if args.mode == "proxy" and args.excess_clamp == "scene":
        print("[doremi] warning: --excess-clamp scene makes the domain excess ~90% a measure of\n"
              "          the per-window difference's variance rather than its mean, which is what\n"
              "          made runs/doremi-proxy-raw learn the DL3DV/ScanNet split (corr -0.77)\n"
              "          instead of hardness. Use it only to reproduce that run.")
    if args.mode == "proxy" and args.batch_size * args.grad_accum < 8:
        print(f"[doremi] warning: {args.batch_size} x {args.grad_accum} = "
              f"{args.batch_size * args.grad_accum} scenes per weight update. Each domain's excess "
              f"loss is then estimated from ~1 scene; raise --grad-accum")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def build_train_parts(args, split: str, *, augment: bool, seed: int | None, sampling: str,
                      rank: int = 0) -> dict:
    """`training/train.py`'s `make_split`, restated because it is nested in its main()."""
    built: dict[str, DL3DVDataset] = {}
    sources = (
        ("dl3dv", args.data_root, args.depth_root, args.dense_only),
        ("scannet", args.scannet_root,
         args.scannet_depth_root or (args.scannet_root / "depth" if args.scannet_root else None),
         False),
    )
    for name, root, depth_root, dense_only in sources:
        if root is None:
            continue
        try:
            built[name] = DL3DVDataset(
                root, name=name, split=split, num_frames=args.num_frames,
                resolution=args.resolution, sampling=sampling, augment=augment, seed=seed,
                depth_root=depth_root, dense_only=dense_only,
            )
        except ValueError as exc:
            if split == "train":
                raise
            if is_main(rank):
                print(f"[{name}] no {split} split ({exc}); skipping its {split} loader")
    return built


def pin_pool_to_domains(train_parts: dict, domains, args, rank: int) -> None:
    """Restrict every dataset to the scenes the clusters json covers.

    All three runs do this, which is the point: an excess loss only means something if
    the reference model was trained on the same pool, and a learned weight only means
    something if the scenes it applies to are the ones that were weighted. Scenes
    outside the clusters json (or inside a cluster `--min-domain-size` dropped) leave
    the pool here rather than being trained on with an undefined domain.
    """
    for name, dataset in train_parts.items():
        before = len(dataset.scenes)
        keep = [e for e in dataset.scenes if domains.domain_of(e["subset"], e["scene"]) >= 0]
        if args.scene_list is not None:
            wanted = set(args.scene_list.read_text().split())
            keep = [e for e in keep if f"{e['subset']}/{e['scene']}" in wanted]
        dataset.scenes = keep
        if is_main(rank):
            print(f"[domains] {name}: {len(keep)}/{before} scenes are in "
                  f"{domains.path.name}" + (" and --scene-list" if args.scene_list else ""))


# --------------------------------------------------------------------------- #
# reference losses
# --------------------------------------------------------------------------- #


class ReferenceModel:
    """A frozen reference model, scored per scene on the batch the proxy just saw."""

    def __init__(self, path: Path, device: torch.device, preset: str | None = None) -> None:
        from training.evaluate import load_checkpoint

        self.model, self.step, self.train_args = load_checkpoint(path, preset, device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.device = device
        self.path = path

    def describe(self) -> str:
        return (f"reference {self.path} (step {self.step}, preset "
                f"{self.train_args.get('preset', '?')}) on {self.device}")

    @torch.no_grad()
    def losses(self, criterion: VGGTOmegaLoss, batch: dict) -> torch.Tensor:
        moved = batch if self.device == batch["images"].device else move_to_device(batch, self.device)
        predictions = self.model(moved["images"])
        return per_sample_losses(criterion, predictions, moved).loss.detach()


class ReferenceCache:
    """Per-scene reference losses read from an `evaluate.py` windows jsonl.

    Keyed by `scene_id` alone, which is what that jsonl carries. DL3DV hashes and
    ScanNet `sceneXXXX_YY` ids do not collide with each other, so this is
    unambiguous for these two datasets and would not be for a third.
    """

    def __init__(self, path: Path, key: str = "loss") -> None:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                value = row.get(key)
                if value is None or not math.isfinite(value):
                    continue
                scene = row["scene_id"]
                sums[scene] = sums.get(scene, 0.0) + float(value)
                counts[scene] = counts.get(scene, 0) + 1
        self.mean = {k: sums[k] / counts[k] for k in sums}
        self.path, self.key = Path(path), key
        self.windows = sum(counts.values())
        self.missing: set[str] = set()

    def describe(self) -> str:
        return (f"reference cache {self.path.name}: {len(self.mean)} scenes, "
                f"{self.windows} windows, key={self.key!r}")

    def losses(self, criterion: VGGTOmegaLoss, batch: dict) -> torch.Tensor:
        values = []
        for scene in batch["scene_id"]:
            if scene not in self.mean:
                self.missing.add(scene)
            values.append(self.mean.get(scene, float("nan")))
        return torch.tensor(values, dtype=torch.float32, device=batch["depth"].device)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = build_parser().parse_args()
    resolve_args(args)
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    domains = load_domains(args.clusters, min_size=args.min_domain_size)
    baseline = domains.baseline()
    prior = baseline if args.dro_prior == "baseline" else domains.uniform()
    if is_main(rank):
        print(f"[domains] {domains.path.name}: {len(domains)} domains over {domains.n_scenes} "
              f"scenes (min_size={args.min_domain_size}, dropped "
              f"{domains.index_meta['dropped_by_min_size']} scenes)")

    criterion = VGGTOmegaLoss(
        weight_camera=args.weight_camera,
        weight_depth=args.weight_depth,
        weight_point=args.weight_point,
        weight_gradient=args.weight_gradient,
        depth_kwargs={"alpha": args.conf_alpha},
    )

    # ---- data ----
    if args.data_root is None and args.scannet_root is None:
        raise SystemExit("pass --data-root (DL3DV), --scannet-root (ScanNet), or both")

    train_parts = build_train_parts(args, "train", augment=True, seed=None,
                                    sampling=args.sampling, rank=rank)
    pin_pool_to_domains(train_parts, domains, args, rank)
    if args.overfit:
        for dataset in train_parts.values():
            dataset.scenes = dataset.scenes[: args.overfit]
            dataset.seed = 0
            dataset.augment = False
    train_parts = {n: d for n, d in train_parts.items() if len(d.scenes) > 0}
    if not train_parts:
        raise SystemExit("no training scenes survived the domain filter")

    image_hw = assert_stackable(train_parts, args.batch_size)
    # No --dl3dv-weight / --scannet-weight here: in reference and proxy mode the
    # mixture has to be the natural one for the excess loss to mean anything, and in
    # final mode the learned per-scene weights already subsume a per-dataset weight.
    train_set, source_names, sizes, _ = build_concat_trainset(train_parts, None, seed=args.seed)

    # ---- the mixture the epoch is drawn from ----
    alpha = load_weights(args.domain_weights, domains) if args.mode == "final" else prior.copy()
    train_sampler = None
    if args.mode == "final":
        factors = scene_factors(domains, alpha)
        weights = index_weights(train_set, factors)
        train_sampler = WeightedEpochSampler(
            weights, len(train_set), rank=rank, world_size=world_size, seed=args.seed
        )
        if is_main(rank):
            ratio = alpha / baseline
            print(f"[final] sampling from {args.domain_weights.name}: ratio "
                  f"min={ratio.min():.2f} max={ratio.max():.2f} "
                  f"ess={effective_sample_size(alpha):.1f}/{len(domains)}, "
                  f"{int((weights == 0).sum())} indices excluded")
    elif world_size > 1:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)

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

    if args.dry_run:
        if is_main(rank):
            keys = concat_scene_keys(train_set)
            per_domain = np.zeros(len(domains), dtype=np.int64)
            for subset, scene in keys:
                per_domain[domains.domain_of(subset, scene)] += 1
            print(f"[dry-run] mode={args.mode}  pool={len(train_set)} indices "
                  f"[{dict(zip(source_names, sizes))}]  {image_hw[0]}x{image_hw[1]}")
            print(f"[dry-run] domains covered={int((per_domain > 0).sum())}/{len(domains)}  "
                  f"scenes/domain min={per_domain.min()} max={per_domain.max()}")
            print(format_table(domains, alpha, limit=6))
            if args.mode == "proxy":
                print(f"[dry-run] proxy: {args.batch_size * args.grad_accum} scenes per weight "
                      f"update, eta={'auto (budget %g)' % args.dro_budget if args.dro_lr is None else args.dro_lr}, "
                      f"prior={args.dro_prior}, max_ratio={args.max_ratio or 'off'}, "
                      f"objective={args.dro_objective}")
                if args.reference_losses:
                    cache = ReferenceCache(args.reference_losses, args.reference_loss_key)
                    covered = sum(1 for _, s in concat_scene_keys(train_set) if s in cache.mean)
                    print(f"[dry-run] {cache.describe()}; covers {covered}/{len(train_set)} indices")
        if world_size > 1:
            dist.destroy_process_group()
        return 0

    # ---- model ----
    model = build_model(args.preset, use_checkpoint=args.checkpointing,
                        dinov3_checkpoint=args.dinov3).to(device)
    if is_main(rank):
        summary = parameter_summary(model)
        print(f"preset={args.preset}  " + "  ".join(f"{k}={v:.1f}M" for k, v in summary.items()))

    ddp_model = model
    if world_size > 1:
        ddp_model = DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=args.freeze_backbone_steps > 0,
        )

    reference = None
    if args.mode == "proxy":
        if args.reference is not None:
            ref_device = torch.device(args.reference_device) if args.reference_device else device
            reference = ReferenceModel(args.reference, ref_device, preset=None)
        else:
            reference = ReferenceCache(args.reference_losses, args.reference_loss_key)
        if is_main(rank):
            print("[proxy] " + reference.describe())

    dro = DomainReweighter(
        # eta is set per step below when --dro-lr is `auto`; 0.0 here so a run that
        # somehow never measures two domains leaves the mixture at its prior rather
        # than moving it on one domain's evidence.
        len(domains), eta=args.dro_lr or 0.0, smoothing=args.dro_smoothing, ema=args.dro_ema,
        prior=prior, max_ratio=args.max_ratio or None,
    ) if args.mode == "proxy" else None
    dro_planned_steps = max(args.max_steps - args.dro_start_step, 1)

    val_loaders: dict[str, DataLoader] = {}
    if args.val_every and not args.overfit:
        for name, dataset in build_train_parts(args, "val", augment=False, seed=1234,
                                               sampling="covisibility", rank=rank).items():
            val_loaders[name] = DataLoader(
                TaggedDataset(dataset, name), batch_size=args.batch_size, shuffle=False,
                num_workers=max(args.workers // 2, 1), pin_memory=True, collate_fn=collate_mixed,
            )

    if is_main(rank):
        val_desc = ("  val " + " ".join(f"{n}={len(l.dataset)}" for n, l in val_loaders.items())
                    if val_loaders else "  (no val)")
        print(f"[{args.mode}] train samples/epoch={len(train_set)} "
              f"[{dict(zip(source_names, sizes))}]{val_desc}  {image_hw[0]}x{image_hw[1]}  "
              f"frames/sample={args.num_frames}  world_size={world_size}")

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
        if dro is not None and payload.get("dro") and not args.reset_dro:
            dro.load_state(payload["dro"])
            if is_main(rank) and dro.steps > 100:
                print(f"[doremi] warning: resumed a reweighter with {dro.steps} updates in it. "
                      f"`alpha_bar`\n"
                      f"          is a running mean, so each new step moves the reported "
                      f"mixture by 1/{dro.steps + 1}\n"
                      f"          and it cannot leave where it already was. Pass --reset-dro "
                      f"for a fresh\n          DRO phase off a finished proxy.")
        if args.constant_lr is not None:
            # Rebuild the schedule as a constant rather than letting the cosine be
            # re-derived from the new --max-steps. `initial_lr` is what LambdaLR
            # reads as `base_lrs`, and it carries --backbone-lr-mult, so scale the
            # groups by their existing ratio to the peak instead of assuming which
            # group is which.
            peak = max(g.get("initial_lr", g["lr"]) for g in optimizer.param_groups)
            for group in optimizer.param_groups:
                mult = group.get("initial_lr", group["lr"]) / peak if peak > 0 else 1.0
                group["initial_lr"] = args.constant_lr * mult
                group["lr"] = group["initial_lr"]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        if is_main(rank):
            lr_desc = (f", lr held at {args.constant_lr:g}" if args.constant_lr is not None
                       else f", lr {optimizer.param_groups[0]['lr']:g} from the cosine over "
                            f"{args.max_steps} steps")
            dro_desc = ""
            if dro is not None:
                dro_desc = (", mixture reset to --dro-prior" if args.reset_dro
                            else f" (domain weights at step {dro.steps})")
            print(f"resumed from {args.resume} at step {start_step}{dro_desc}{lr_desc}")

    log_path = args.out / "log.jsonl"
    weights_log = args.out / "domain_weights.jsonl"
    weights_json = args.out / "domain_weights.json"
    writer = None
    if is_main(rank):
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps(vars(args), indent=1, default=str))
        writer = SummaryWriter(log_dir=str(args.out / "tb"))

    def write_weights(step: int) -> None:
        save_weights(
            weights_json, domains, dro.alpha_bar if dro.steps else dro.alpha,
            extra={
                "mode": args.mode, "step": step, "dro_steps": dro.steps,
                "reference": str(args.reference or args.reference_losses),
                "eta": dro.eta, "eta_mode": "auto" if args.dro_lr is None else "fixed",
                "budget": args.dro_budget if args.dro_lr is None else None,
                "smoothing": args.dro_smoothing, "ema": args.dro_ema,
                "prior": args.dro_prior, "max_ratio": args.max_ratio or None,
                "objective": args.dro_objective,
                "scenes_per_update": args.batch_size * args.grad_accum * world_size,
                "ess": effective_sample_size(dro.alpha_bar if dro.steps else dro.alpha),
                "excess_loss": dro.excess.tolist(), "seen": dro.seen.tolist(),
            },
        )

    # ---- loop ----
    ddp_model.train()
    step = start_step
    micro = 0
    epoch = 0
    running: dict[str, float] = {}
    running_count = 0
    # Per-domain excess-loss sums for the current --grad-accum window. One weight
    # update per optimiser step, over every scene that contributed to it.
    window_excess = torch.zeros(len(domains), device=device, dtype=torch.float64)
    window_counts = torch.zeros(len(domains), device=device, dtype=torch.float64)
    window_skipped = 0
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

            with sync_context:
                predictions = ddp_model(batch["images"])

                if args.mode == "proxy":
                    ids = domains.ids_for_batch(batch)
                    proxy = per_sample_losses(criterion, predictions, batch)
                    ref_loss = reference.losses(criterion, batch).to(device)
                    excess = excess_losses(proxy.loss, ref_loss,
                                          clamp=args.excess_clamp == "scene")

                    usable = proxy.valid & torch.isfinite(ref_loss)
                    usable_np = usable.detach().cpu().numpy() & (ids >= 0)
                    window_skipped += int((~usable_np).sum())
                    for d in np.unique(ids[usable_np]):
                        rows = np.flatnonzero(usable_np & (ids == d))
                        window_excess[d] += float(excess[rows].sum())
                        window_counts[d] += len(rows)

                    loss, domain_means, domain_counts = domain_weighted_loss(
                        proxy.loss, ids, dro.alpha, baseline=baseline, valid=usable,
                        scheme=args.dro_objective,
                    )
                    logs = {k: v[usable].mean().detach() if usable.any() else v.mean().detach()
                            for k, v in proxy.logs.items()}
                    logs["loss"] = loss.detach()
                    logs["excess"] = excess[usable].mean().detach() if usable.any() else excess.sum() * 0
                    logs["ref_loss"] = (ref_loss[usable].mean().detach() if usable.any()
                                        else ref_loss.sum() * 0)
                    logs["domains_in_batch"] = torch.tensor(float(len(domain_counts)), device=device)
                elif args.per_sample_objective:
                    per_sample = per_sample_losses(criterion, predictions, batch)
                    keep = per_sample.valid
                    loss = (per_sample.loss[keep].mean() if keep.any()
                            else per_sample.loss.sum() * 0.0)
                    logs = {k: v[keep].mean().detach() if keep.any() else v.mean().detach()
                            for k, v in per_sample.logs.items()}
                    logs["loss"] = loss.detach()
                else:
                    loss, logs = criterion(predictions, batch)

                (loss / args.grad_accum).backward()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item() / args.grad_accum

            if not is_last_micro:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            running["grad_norm"] = running.get("grad_norm", 0.0) + grad_norm.item()
            running_count += 1

            # ---- one domain weight update per optimiser step ----
            if dro is not None:
                if world_size > 1:
                    # Every rank must end the step with the same alpha, so the
                    # statistics are pooled before the update rather than after.
                    stacked = torch.stack([window_excess, window_counts])
                    dist.all_reduce(stacked)
                    window_excess, window_counts = stacked[0], stacked[1]
                if step > args.dro_start_step:
                    if args.dro_lr is None:
                        dro.eta = auto_eta(dro.excess, dro.seen, total_logit=math.log(args.dro_budget),
                                           planned_steps=dro_planned_steps)
                    dro.step(window_excess.cpu().numpy(), window_counts.cpu().numpy())
                else:
                    dro.observe(window_excess.cpu().numpy(), window_counts.cpu().numpy())
                window_excess = torch.zeros_like(window_excess)
                window_counts = torch.zeros_like(window_counts)

                if is_main(rank) and args.dro_log_every and step % args.dro_log_every == 0:
                    with weights_log.open("a") as fh:
                        fh.write(json.dumps({
                            "step": step,
                            "alpha": [round(v, 6) for v in dro.alpha.tolist()],
                            "alpha_bar": [round(v, 6) for v in dro.alpha_bar.tolist()],
                            "excess": [round(v, 6) for v in dro.excess.tolist()],
                            "seen": dro.seen.tolist(),
                            "clusters": domains.ids,
                        }) + "\n")
                    write_weights(step)

            if is_main(rank) and args.log_every and step % args.log_every == 0:
                means = {k: v / running_count for k, v in running.items()}
                elapsed = time.time() - started
                lrs = scheduler.get_last_lr()
                record = {
                    "step": step,
                    "mode": args.mode,
                    "lr": max(lrs),
                    "lr_backbone": min(lrs),
                    "steps_per_sec": args.log_every / max(elapsed, 1e-9),
                    **{k: round(v, 5) for k, v in means.items()},
                }
                extra = ""
                if dro is not None:
                    # Against the baseline, not --dro-prior: `ratio` is read as an
                    # oversampling factor, and that is relative to the natural mixture.
                    ratio = dro.alpha_bar / np.maximum(baseline, 1e-12) if dro.steps else np.ones(1)
                    record.update(
                        dro_steps=dro.steps,
                        ess=effective_sample_size(dro.alpha),
                        ratio_max=float(ratio.max()),
                        ratio_min=float(ratio.min()),
                        excess_mean=float(dro.excess.mean()),
                        eta=dro.eta,
                        domains_seen=int((dro.seen > 0).sum()),
                        skipped=window_skipped,
                    )
                    extra = (f"ref {means.get('ref_loss', 0):.4f}  "
                             f"excess {means.get('excess', 0):.4f}  "
                             f"ess {record['ess']:.1f}/{len(domains)}  "
                             f"ratio {record['ratio_min']:.2f}-{record['ratio_max']:.2f}  ")
                    window_skipped = 0
                print(
                    f"[{args.mode}] step {step:>7d}/{args.max_steps}  "
                    f"loss {means.get('loss', 0):.4f}  cam {means.get('loss_camera', 0):.4f}  "
                    f"depth {means.get('loss_depth', 0):.4f}  point {means.get('loss_point', 0):.4f}  "
                    + extra
                    + f"lr {record['lr']:.2e}  {record['steps_per_sec']:.2f} it/s",
                    flush=True,
                )
                with log_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                if writer is not None:
                    for k, v in record.items():
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"train/{k}", v, step)
                    if dro is not None:
                        for d, cid in enumerate(domains.ids):
                            writer.add_scalar(f"alpha/c{cid:02d}", dro.alpha[d], step)
                running, running_count, started = {}, 0, time.time()

            if val_loaders and args.val_every and step % args.val_every == 0:
                metrics = evaluate_all(model, val_loaders, criterion, device, args.val_batches)
                if is_main(rank):
                    nan = float("nan")
                    print(f"  [val @ {step}] loss {metrics.get('loss', nan):.4f}  "
                          f"AUC@30 {metrics.get('auc_at_30', nan):.3f}  "
                          f"rot_err {metrics.get('rot_err_deg_median', nan):.2f}deg  "
                          f"abs_rel {metrics.get('abs_rel', nan):.3f}", flush=True)
                    with log_path.open("a") as fh:
                        fh.write(json.dumps({"step": step, "split": "val", **metrics}) + "\n")
                    if writer is not None:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and math.isfinite(v):
                                writer.add_scalar(f"val/{k}", v, step)

            if is_main(rank) and args.save_every and step % args.save_every == 0:
                path = args.out / "latest.pt"
                save_checkpoint(path, ddp_model, optimizer, scheduler, step, args)
                if dro is not None:
                    # `save_checkpoint` does not know about the reweighter, so the
                    # weight state is added to the payload it just wrote -- resuming a
                    # proxy run without it would restart the mixture from the prior.
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    payload["dro"] = dro.state()
                    torch.save(payload, path)

    if is_main(rank):
        save_checkpoint(args.out / "final.pt", ddp_model, optimizer, scheduler, step, args)
        if dro is not None:
            payload = torch.load(args.out / "final.pt", map_location="cpu", weights_only=False)
            payload["dro"] = dro.state()
            torch.save(payload, args.out / "final.pt")
            write_weights(step)
            print(f"\n[proxy] learned mixture after {dro.steps} weight updates -- {dro.summary(baseline)}")
            print(format_table(domains, dro.alpha_bar if dro.steps else dro.alpha,
                               excess=dro.excess, limit=8))
            print(f"wrote {weights_json}")
            if isinstance(reference, ReferenceCache) and reference.missing:
                print(f"[proxy] warning: {len(reference.missing)} scenes were absent from the "
                      f"reference cache and were skipped")
            print(f"\nnext:\n    python doremi/train.py --mode final "
                  f"--domain-weights {weights_json} \\\n        --preset <target> ...")
        print(f"done at step {step}; wrote {args.out / 'final.pt'}")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
