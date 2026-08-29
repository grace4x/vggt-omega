#!/usr/bin/env python3
"""Score a trained VGGT-Omega checkpoint on a preprocessed DL3DV set.

The eval set is built exactly like the training set -- `preprocess_dl3dv.py` for
images/COLMAP, then `fetch_dl3dv_depth.py` for dense depth -- so a held-out
download (e.g. DL3DV's 960P benchmark scenes) drops straight in:

    python training/evaluate.py --checkpoint runs/small-v5/latest.pt \
        --data-root ~/dl3dv-eval --depth-root ~/dl3dv-eval-depth --split all

Every setting the numbers depend on (preset, num_frames, resolution, dense_only)
defaults to whatever the checkpoint was trained with, read back out of the
checkpoint's own `args`, so the eval matches the run unless you say otherwise.

Several checkpoints in one go, sharing one pass over the data per checkpoint:

    python training/evaluate.py --checkpoint runs/small-v*/latest.pt \
        --data-root ~/dl3dv-eval --depth-root ~/dl3dv-eval-depth \
        --out runs/eval-2026-08

What the numbers mean, and the two traps in reading them:

* **Pose** metrics are pooled over every frame pair of every window, not averaged
  over per-window summaries -- a median of medians is not a median. `auc_at_30`
  (mAA@30) is the headline the VGGT line of work reports; `rra_at_5` / `rta_at_5`
  break it into rotation and translation, and translation is the harder half.

* **Depth** is reported raw, median-aligned (`*_aligned`), and scale-shift
  aligned (`*_ss`). The model predicts in the loader's unit space -- scene
  normalised so the mean point distance is 1 -- so raw `abs_rel` is a fair
  number *and* includes any global scale error. Median alignment multiplies by
  `median(gt)/median(pred)` (scale only). Scale-shift fits `s * pred + t` by
  least squares, which also removes a constant depth bias. A big gap between
  raw and aligned means the model has the geometry but not the scale.

* A window whose scale normalisation found nothing to key off (`scale_ok=False`)
  has no depth targets at all. Those windows are counted and excluded from the
  depth and point statistics rather than averaged in as zeros.

`--repeats` re-samples different frame windows per scene (deterministically, off
`--seed`). One window per scene is a noisy estimate on a small eval set; 3-5
tightens it at a proportional cost.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dl3dv_dataset import DL3DVDataset, collate_scenes  # noqa: E402
from training.losses import (  # noqa: E402
    VGGTOmegaLoss,
    depth_metrics,
    pose_auc,
    pose_pair_errors,
)
from training.model_config import build_model  # noqa: E402

# Metrics that only exist when the window had depth targets. Set to nan (and so
# dropped from the aggregate) on a window where the loader found none.
DEPTH_KEYS = (
    "loss_depth",
    "loss_point",
    "depth_err",
    "point_err",
    "depth_conf",
    "abs_rel",
    "delta_1.25",
    "scale_ratio",
    "abs_rel_aligned",
    "delta_1.25_aligned",
    "abs_rel_ss",
    "delta_1.25_ss",
    "ss_scale",
    "ss_shift",
)


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #


def read_payload(path: Path) -> tuple[dict, dict, int]:
    """Returns (state_dict, the args the run was launched with, step)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in payload:
        return payload["model"], payload.get("args") or {}, payload.get("step", -1)
    return payload, {}, -1  # a bare state_dict, e.g. one stripped for release


def load_checkpoint(path: Path, preset: str | None, device: torch.device):
    """Returns (model in eval mode, step, the args the run was launched with)."""
    state_dict, train_args, step = read_payload(path)
    resolved = preset or train_args.get("preset") or "small"
    # No `dinov3_checkpoint`: the trunk weights in the state dict replace it anyway,
    # and loading the converted trunk first just doubles the load time.
    model = build_model(resolved, use_checkpoint=False)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(
            f"{path} does not fit preset {resolved!r}. Pass --preset explicitly if the "
            f"checkpoint predates the `args` payload.\n{exc}"
        ) from exc
    return model.to(device).eval(), step, train_args


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def window_metrics(predictions: dict, batch: dict, index: int, criterion) -> dict[str, float]:
    """Every metric for one scene window, i.e. batch element `index`."""
    single_pred = {k: v[index : index + 1] for k, v in predictions.items()}
    single_batch = {
        k: (v[index : index + 1] if torch.is_tensor(v) else v[index : index + 1])
        for k, v in batch.items()
    }
    _, logs = criterion(single_pred, single_batch)
    record = {k: v.item() for k, v in logs.items()}

    depth, mask = single_batch["depth"], single_batch["depth_mask"]
    record.update(depth_metrics(single_pred["depth"], depth, mask))
    aligned = depth_metrics(single_pred["depth"], depth, mask, align_median=True)
    record.update({f"{k}_aligned": v for k, v in aligned.items() if k in ("abs_rel", "delta_1.25")})
    ss = depth_metrics(single_pred["depth"], depth, mask, align_scale_shift=True)
    record["abs_rel_ss"] = ss["abs_rel"]
    record["delta_1.25_ss"] = ss["delta_1.25"]
    record["ss_scale"] = ss["ss_scale"]
    record["ss_shift"] = ss["ss_shift"]

    if not bool(single_batch["scale_ok"][0]) or not mask.any():
        for key in DEPTH_KEYS:
            record[key] = float("nan")
    return record


def summarise(records: list[dict], rotation_deg: torch.Tensor, translation_deg: torch.Tensor) -> dict:
    """Per-window means, plus pose statistics pooled over every frame pair."""
    out: dict[str, float] = {}
    for key in sorted({k for r in records for k in r}):
        values = [r[key] for r in records if isinstance(r.get(key), float) and math.isfinite(r[key])]
        if values:
            out[key] = sum(values) / len(values)
            out[f"{key}_n"] = float(len(values))

    if rotation_deg.numel():
        out.update(
            {
                "auc_at_30": pose_auc(rotation_deg, translation_deg),
                "rra_at_5": (rotation_deg < 5).float().mean().item(),
                "rra_at_15": (rotation_deg < 15).float().mean().item(),
                "rta_at_5": (translation_deg < 5).float().mean().item(),
                "rta_at_15": (translation_deg < 15).float().mean().item(),
                "rot_err_deg_median": rotation_deg.median().item(),
                "rot_err_deg_mean": rotation_deg.mean().item(),
                "trans_err_deg_median": translation_deg.median().item(),
                "trans_err_deg_mean": translation_deg.mean().item(),
                "num_pairs": float(rotation_deg.numel()),
            }
        )
    out["num_windows"] = float(len(records))
    out["num_windows_no_depth"] = float(sum(1 for r in records if not math.isfinite(r.get("abs_rel", float("nan")))))
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


@torch.inference_mode()
def evaluate_checkpoint(model, dataset, criterion, device, args) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    rotations, translations = [], []
    started = time.time()

    for repeat in range(args.repeats):
        # Re-seeding between repeats gives each scene a different frame window while
        # keeping the whole evaluation reproducible. Workers snapshot the dataset when
        # the iterator is created, so this has to happen before the loop below (and is
        # why `persistent_workers` stays off).
        dataset.seed = args.seed + 100_003 * repeat
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_scenes,
        )

        for batch in loader:
            tensors = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            predictions = model(tensors["images"])

            rot, trans = pose_pair_errors(predictions["pose_enc"], tensors["pose_enc"])
            rotations.append(rot.flatten().cpu())
            translations.append(trans.flatten().cpu())

            for i, scene_id in enumerate(batch["scene_id"]):
                record = window_metrics(predictions, tensors, i, criterion)
                record.update(
                    scene_id=scene_id,
                    repeat=repeat,
                    dense_depth=bool(tensors["dense_depth"][i]),
                    scene_scale=float(tensors["scene_scale"][i]),
                )
                # Per-window pose numbers too, so a single bad scene is findable in
                # the jsonl even though the headline figures are pooled.
                record["rot_err_deg_median_window"] = rot[i].median().item()
                record["trans_err_deg_median_window"] = trans[i].median().item()
                records.append(record)

            if args.progress and len(records) % (args.progress * max(args.batch_size, 1)) == 0:
                rate = len(records) / max(time.time() - started, 1e-9)
                print(f"  {len(records)} windows  ({rate:.2f} windows/s)", flush=True)

    rotation_deg = torch.cat(rotations) if rotations else torch.zeros(0)
    translation_deg = torch.cat(translations) if translations else torch.zeros(0)
    summary = summarise(records, rotation_deg, translation_deg)
    summary["seconds"] = time.time() - started

    # The dense/sparse split is worth keeping separate: ~1%-coverage COLMAP depth
    # and ~100% DA3 depth are not the same measurement, and mixing them makes the
    # depth numbers depend on the composition of the eval set.
    for name, subset in (
        ("dense", [r for r in records if r["dense_depth"]]),
        ("sparse", [r for r in records if not r["dense_depth"]]),
    ):
        if subset and len(subset) != len(records):
            summary[f"by_{name}"] = summarise(subset, torch.zeros(0), torch.zeros(0))
    return summary, records


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

TABLE = (
    ("AUC@30", "auc_at_30", "{:.3f}"),
    ("RRA@5", "rra_at_5", "{:.3f}"),
    ("RTA@5", "rta_at_5", "{:.3f}"),
    ("rot_med", "rot_err_deg_median", "{:.2f}"),
    ("trans_med", "trans_err_deg_median", "{:.2f}"),
    ("fov_err", "cam_fov", "{:.4f}"),
    ("abs_rel", "abs_rel", "{:.3f}"),
    ("abs_rel*", "abs_rel_aligned", "{:.3f}"),
    ("abs_rel†", "abs_rel_ss", "{:.3f}"),
    ("d<1.25", "delta_1.25", "{:.3f}"),
    ("d<1.25†", "delta_1.25_ss", "{:.3f}"),
    ("scale", "scale_ratio", "{:.3f}"),
    ("point_err", "point_err", "{:.3f}"),
    ("loss", "loss", "{:.4f}"),
)


def print_table(rows: list[tuple[str, int, dict]]) -> None:
    name_width = max(len(name) for name, _, _ in rows)
    header = f"{'checkpoint':<{name_width}}  {'step':>7}  " + "  ".join(f"{h:>9}" for h, _, _ in TABLE)
    print("\n" + header)
    print("-" * len(header))
    for name, step, summary in rows:
        cells = []
        for _, key, fmt in TABLE:
            value = summary.get(key)
            cells.append(f"{fmt.format(value):>9}" if isinstance(value, float) and math.isfinite(value) else f"{'--':>9}")
        print(f"{name:<{name_width}}  {step:>7}  " + "  ".join(cells))
    print(
        "\n  * = median-aligned (scale only).  † = scale-shift aligned (s*pred+t, least squares)."
        "  scale = median(gt)/median(pred), 1.0 is correct."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="one or more *.pt from train.py")
    p.add_argument("--data-root", type=Path, required=True, help="output of preprocess_dl3dv.py")
    p.add_argument("--depth-root", type=Path, default=None, help="output of fetch_dl3dv_depth.py")
    p.add_argument("--split", default="all", choices=("all", "train", "val"))
    p.add_argument("--out", type=Path, default=None, help="write summary.json + windows.jsonl here")

    p.add_argument("--preset", default=None, help="override; defaults to the checkpoint's own")
    p.add_argument("--num-frames", type=int, default=None, help="override; defaults to the checkpoint's own")
    p.add_argument("--resolution", type=int, default=None)
    # Tri-state: unset follows the checkpoint, which trained with one or the other.
    p.add_argument("--dense-only", action="store_true", default=None, help="drop scenes with no dense depth")
    p.add_argument("--no-dense-only", action="store_false", dest="dense_only",
                   help="keep them, scoring sparse COLMAP depth where DA3 is missing")
    p.add_argument("--sampling", default="covisibility", choices=("covisibility", "contiguous", "random"))
    p.add_argument("--scene-list", type=Path, default=None, help="restrict to `subset/scene` lines in this file")
    p.add_argument("--max-scenes", type=int, default=0, help="first N scenes only; for a smoke test")
    p.add_argument("--repeats", type=int, default=1, help="frame windows sampled per scene")

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--progress", type=int, default=25, help="print every N batches; 0 silences it")
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    for path in args.checkpoint:
        if not path.exists():
            raise SystemExit(f"no such checkpoint: {path}")

    # Dataset settings follow the first checkpoint's run unless overridden, so the
    # eval reproduces the conditions the model was trained under by default. Only
    # its `args` are needed here; the weights are loaded one checkpoint at a time
    # below so a glob of them does not all sit in memory at once.
    reference = read_payload(args.checkpoint[0])[1]
    num_frames = args.num_frames or reference.get("num_frames") or 16
    resolution = args.resolution if args.resolution is not None else reference.get("resolution")
    dense_only = args.dense_only if args.dense_only is not None else bool(reference.get("dense_only"))
    depth_root = args.depth_root
    if depth_root is None and reference.get("depth_root"):
        print(f"[eval] no --depth-root given; the run used {reference['depth_root']} -- falling back to sparse GT")

    dataset = DL3DVDataset(
        args.data_root,
        split=args.split,
        num_frames=num_frames,
        resolution=resolution,
        sampling=args.sampling,
        augment=False,
        seed=args.seed,
        depth_root=depth_root,
        dense_only=dense_only,
    )
    if args.scene_list is not None:
        wanted = set(args.scene_list.read_text().split())
        dataset.scenes = [e for e in dataset.scenes if f"{e['subset']}/{e['scene']}" in wanted]
        if not dataset.scenes:
            raise SystemExit(f"none of the {len(wanted)} scenes in {args.scene_list} survived the dataset filters")
    if args.max_scenes:
        dataset.scenes = dataset.scenes[: args.max_scenes]

    criterion = VGGTOmegaLoss(
        weight_camera=reference.get("weight_camera", 5.0),
        weight_depth=reference.get("weight_depth", 1.0),
        weight_point=reference.get("weight_point", 0.5),
        weight_gradient=reference.get("weight_gradient", 1.0),
        depth_kwargs={"alpha": reference.get("conf_alpha", 0.2)},
    )

    print(
        f"[eval] {len(dataset)} scenes x {args.repeats} window(s)  frames={num_frames}  "
        f"hw={dataset.image_hw}  sampling={args.sampling}  depth={'dense' if depth_root else 'sparse'}"
    )

    rows, results = [], {}
    for path in args.checkpoint:
        model, step, train_args = load_checkpoint(path, args.preset, device)
        print(f"\n[eval] {path}  (step {step}, preset {args.preset or train_args.get('preset', 'small')})")
        summary, records = evaluate_checkpoint(model, dataset, criterion, device, args)
        summary["checkpoint"], summary["step"] = str(path), step
        rows.append((str(path), step, summary))
        results[str(path)] = summary

        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            stem = path.parent.name + "-" + path.stem  # runs/small-v5/latest.pt -> small-v5-latest
            with (args.out / f"windows-{stem}.jsonl").open("w") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_table(rows)

    if args.out is not None:
        payload = {
            "data_root": str(args.data_root),
            "depth_root": str(depth_root) if depth_root else None,
            "split": args.split,
            "num_frames": num_frames,
            "resolution": resolution,
            "dense_only": dense_only,
            "sampling": args.sampling,
            "repeats": args.repeats,
            "seed": args.seed,
            "num_scenes": len(dataset),
            "results": results,
        }
        (args.out / "summary.json").write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
