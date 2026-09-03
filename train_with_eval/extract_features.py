#!/usr/bin/env python3
"""Precompute layer-30 DINOv3 patch-mean descriptors for train_with_eval diversity.

One forward of ViT-7B per sampled frame, written to an npz that
`train_with_eval/train.py --diversity-features` looks up by
`dataset/subset/scene`. Same recipe as `BatchDiversityTracker`: the model's
final layernorm on each selected layer's tokens, then mean of the patch
tokens, L2-normalize per frame, then L2-normalize the mean across frames
into one scene vector. `hidden_states[l]` is pre-norm; only `last_hidden_state`
has the layernorm applied, so `--final-ln` (on by default) is how DINOv3
intermediate features are normally consumed. Pass `--no-final-ln` for the
old raw-hidden-state recipe. Layernorm is per token and not linear, so it
cannot be recovered from a stored patch mean -- re-extract to change it.

    python train_with_eval/extract_features.py \\
        --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only \\
        --scannet-root ~/scannet-train --num-frames 8 \\
        --out train_with_eval/layer30_features.npz

Then train with `--diversity-features train_with_eval/layer30_features.npz`
instead of running the 7B on CPU every trace step. Interrupted runs continue
with `--resume`.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_with_eval.batch_diversity import (  # noqa: E402
    DEFAULT_FINAL_LN,
    DEFAULT_LAYERS,
    DEFAULT_MODEL,
    scene_key,
)
from training.dl3dv_dataset import DL3DVDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
N_IMAGES = 4
SEED = 0
SAVE_EVERY = 50


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="preprocessed DL3DV root")
    p.add_argument("--depth-root", type=Path, default=None, help="DL3DV dense depth root")
    p.add_argument("--scannet-root", type=Path, default=None, help="preprocessed ScanNet root")
    p.add_argument("--scannet-depth-root", type=Path, default=None,
                   help="ScanNet depth root (default: <scannet-root>/depth)")
    p.add_argument("--dense-only", action="store_true", help="drop DL3DV scenes with no dense depth")
    p.add_argument("--num-frames", type=int, default=8,
                   help="same as train.py: scenes shorter than this are dropped")
    p.add_argument("--dl3dv-scenes", type=int, default=0, help="cap DL3DV at first N train scenes (0 = all)")
    p.add_argument("--scannet-scenes", type=int, default=0, help="cap ScanNet at first N train scenes (0 = all)")
    p.add_argument("--scene-list", type=Path, default=None,
                   help="optional subset/scene allow-list, same as train.py --scene-list")
    p.add_argument("--split", default="train", choices=("train", "val", "all"))
    p.add_argument("--out", type=Path, default=HERE / "layer30_features.npz")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS), metavar="L",
                   help="hidden-state indices whose patch-token means are concatenated (default: 30)")
    p.add_argument(
        "--final-ln",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FINAL_LN,
        help="apply the model's final layernorm to every selected layer's tokens "
             "before the patch mean (default: on; --no-final-ln for raw hidden states)",
    )
    p.add_argument("--n-images", type=int, default=N_IMAGES, metavar="N",
                   help="frames embedded per scene (default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED, help="frame-sampling seed (per-scene)")
    p.add_argument("--device", default="cuda", help="device for DINOv3 (default: cuda)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing npz")
    p.add_argument("--resume", action="store_true",
                   help="skip scenes already present in --out and rewrite the full file")
    return p


def _collect_sources(args) -> list[tuple[str, Path, Path | None, bool]]:
    sources = []
    if args.data_root is not None:
        sources.append(("dl3dv", args.data_root, args.depth_root, args.dense_only))
    if args.scannet_root is not None:
        depth = args.scannet_depth_root or (args.scannet_root / "depth")
        sources.append(("scannet", args.scannet_root, depth, False))
    if not sources:
        raise SystemExit("pass --data-root (DL3DV), --scannet-root (ScanNet), or both")
    return sources


def resolve_scenes(args) -> tuple[list[dict], tuple[int, int]]:
    """Scene entries with `dataset` set, plus the stored image shape (must be one)."""
    wanted = set(args.scene_list.read_text().split()) if args.scene_list is not None else None
    caps = {"dl3dv": args.dl3dv_scenes, "scannet": args.scannet_scenes}
    scenes: list[dict] = []
    shapes: set[tuple[int, int]] = set()
    for name, root, depth_root, dense_only in _collect_sources(args):
        dataset = DL3DVDataset(
            root,
            name=name,
            split=args.split,
            num_frames=args.num_frames,
            depth_root=depth_root,
            dense_only=dense_only,
            augment=False,
            sampling="random",
        )
        entries = list(dataset.scenes)
        if wanted is not None:
            entries = [e for e in entries if f"{e['subset']}/{e['scene']}" in wanted]
        cap = caps.get(name, 0)
        if cap and cap < len(entries):
            entries = entries[:cap]
            print(f"[{name}] capped to the first {len(entries)} {args.split} scenes")
        else:
            print(f"[{name}] {len(entries)} {args.split} scenes")
        for entry in entries:
            scenes.append({**entry, "dataset": name, "root": root})
        shapes.add(tuple(int(v) for v in dataset.image_hw))
    if not scenes:
        raise SystemExit("no scenes survived the dataset filters")
    if len(shapes) > 1:
        raise SystemExit(f"datasets are stored at different resolutions: {sorted(shapes)}")
    return scenes, shapes.pop()


def pick_frames(entry: dict, n_images: int, seed: int) -> list[Path]:
    image_dir = Path(entry["root"]) / entry["path"] / "images"
    available = sorted(image_dir.glob("*.jpg"))
    if not available:
        raise SystemExit(f"no frames under {image_dir}")
    rng = random.Random(f"{seed}:{scene_key(entry['dataset'], entry['subset'], entry['scene'])}")
    if len(available) >= n_images:
        return rng.sample(available, n_images)
    return available + [rng.choice(available) for _ in range(n_images - len(available))]


def _payload(
    *,
    keys,
    patch,
    scene,
    datasets,
    subsets,
    scenes,
    frames,
    image_hw,
    args,
    layers,
) -> dict:
    return {
        "keys": np.array(keys),
        "patch": np.stack(patch).astype(np.float16) if patch else np.zeros((0, args.n_images, 0), np.float16),
        "scene": np.stack(scene).astype(np.float16) if scene else np.zeros((0, 0), np.float16),
        "datasets": np.array(datasets),
        "subsets": np.array(subsets),
        "scenes": np.array(scenes),
        "frames": np.array(frames),
        "image_hw": np.array(image_hw),
        "n_images": np.array(args.n_images),
        "patch_layers": np.array(layers),
        "final_ln": np.array(args.final_ln),
        "features": np.array("patch-avg"),
        "model": np.array(args.model),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write through a file handle so numpy does not append a second `.npz`
    # (it does that when the path does not already end in `.npz`).
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **payload)
    tmp.replace(path)


def load_existing(path: Path, args, layers: tuple[int, ...]) -> dict[str, dict]:
    with np.load(path, allow_pickle=True) as z:
        if str(z["model"]) != args.model:
            raise SystemExit(f"{path} was extracted with model {z['model']!r}, not {args.model!r}")
        recorded = tuple(int(v) for v in z["patch_layers"])
        if recorded != layers:
            raise SystemExit(f"{path} has layers {recorded}, not {layers}")
        if int(z["n_images"]) != args.n_images:
            raise SystemExit(f"{path} has n_images={int(z['n_images'])}, not {args.n_images}")
        recorded_ln = bool(z["final_ln"]) if "final_ln" in z.files else False
        if recorded_ln != args.final_ln:
            raise SystemExit(
                f"{path} has final_ln={recorded_ln}, not {args.final_ln}; "
                "pass --overwrite, or --no-final-ln / --final-ln to match the cache"
            )
        done = {}
        for i, key in enumerate(z["keys"]):
            done[str(key)] = {
                "patch": z["patch"][i],
                "scene": z["scene"][i],
                "dataset": str(z["datasets"][i]),
                "subset": str(z["subsets"][i]),
                "scene_id": str(z["scenes"][i]),
                "frames": [str(f) for f in z["frames"][i]],
                "image_hw": np.asarray(z["image_hw"][i]),
            }
    return done


def embed_frame(
    model, processor, skip, layers, device, frame: Path, height: int, width: int, *, final_ln: bool,
) -> np.ndarray:
    image = processor(
        load_image(str(frame)),
        size={"height": height, "width": width},
        do_center_crop=False,
        return_tensors="pt",
    )
    with torch.inference_mode():
        out = model(**{k: v.to(device, torch.float16) for k, v in image.items()}, output_hidden_states=True)
        vecs = []
        for layer in layers:
            hidden = out.hidden_states[layer][0]  # (1 + registers + patches, dim)
            if final_ln:
                hidden = model.norm(hidden)
            patch_mean = hidden[skip:].mean(0).float().cpu().numpy()
            vecs.append(patch_mean / max(np.linalg.norm(patch_mean), 1e-12))
    return np.concatenate(vecs)


def main() -> int:
    args = build_parser().parse_args()
    layers = tuple(args.layers)
    if args.out.exists() and not args.overwrite and not args.resume:
        raise SystemExit(f"{args.out} exists; pass --overwrite or --resume")

    scenes, (height, width) = resolve_scenes(args)
    if height % 16 or width % 16:
        raise SystemExit(f"stored shape {(height, width)} is not a multiple of the patch size")

    done: dict[str, dict] = {}
    if args.resume and args.out.exists():
        done = load_existing(args.out, args, layers)
        print(f"[resume] {len(done)} scenes already in {args.out}")

    todo = [e for e in scenes if scene_key(e["dataset"], e["subset"], e["scene"]) not in done]
    print(
        f"{len(scenes)} scenes ({len(todo)} to embed) x {args.n_images} frames at {(height, width)}, "
        f"patch-mean layers {layers} on {args.device}"
        + (" (post-final-layernorm)" if args.final_ln else "")
    )
    if not todo:
        print(f"nothing to do; {args.out} is complete")
        return 0

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    skip = 1 + model.config.num_register_tokens
    n_layers = model.config.num_hidden_layers
    if not all(-n_layers <= layer <= n_layers for layer in layers):
        raise SystemExit(f"--layers must be in [-{n_layers}, {n_layers}]")

    def snapshot() -> dict:
        keys, patch, scene_vecs, datasets, subsets, scene_ids, frames, image_hw = (
            [], [], [], [], [], [], [], [],
        )
        order = [scene_key(e["dataset"], e["subset"], e["scene"]) for e in scenes]
        for key in order:
            row = done.get(key)
            if row is None:
                continue
            keys.append(key)
            patch.append(row["patch"])
            scene_vecs.append(row["scene"])
            datasets.append(row["dataset"])
            subsets.append(row["subset"])
            scene_ids.append(row["scene_id"])
            frames.append(row["frames"])
            image_hw.append(row["image_hw"])
        return _payload(
            keys=keys,
            patch=patch,
            scene=scene_vecs,
            datasets=datasets,
            subsets=subsets,
            scenes=scene_ids,
            frames=frames,
            image_hw=image_hw,
            args=args,
            layers=layers,
        )

    short = 0
    for i, entry in enumerate(todo):
        key = scene_key(entry["dataset"], entry["subset"], entry["scene"])
        frame_paths = pick_frames(entry, args.n_images, args.seed)
        if len(sorted((Path(entry["root"]) / entry["path"] / "images").glob("*.jpg"))) < args.n_images:
            short += 1
        frame_vecs = [
            embed_frame(
                model, processor, skip, layers, device, frame, height, width, final_ln=args.final_ln,
            )
            for frame in frame_paths
        ]
        patch = np.stack(frame_vecs)
        scene_vec = patch.mean(axis=0)
        scene_vec = scene_vec / max(np.linalg.norm(scene_vec), 1e-12)
        done[key] = {
            "patch": patch.astype(np.float16),
            "scene": scene_vec.astype(np.float16),
            "dataset": entry["dataset"],
            "subset": entry["subset"],
            "scene_id": entry["scene"],
            "frames": [str(f) for f in frame_paths],
            "image_hw": np.array([height, width]),
        }
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"{i + 1}/{len(todo)} {key}", flush=True)
        if (i + 1) % SAVE_EVERY == 0:
            _write(args.out, snapshot())
            print(f"  checkpointed {len(done)} scenes -> {args.out}", flush=True)

    if short:
        print(f"{short}/{len(todo)} scenes had fewer than {args.n_images} frames; padded by resampling")

    _write(args.out, snapshot())
    print(
        f"wrote {args.out} ({len(done)} scenes, {args.n_images} frames/scene, "
        f"layers {layers}{', final-ln' if args.final_ln else ''})"
    )
    print(f"  next: python train_with_eval/train.py ... --diversity-features {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
