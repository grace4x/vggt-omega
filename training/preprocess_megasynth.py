#!/usr/bin/env python3
"""Turn raw MegaSynth scenes into the same training set format as DL3DV.

MegaSynth (Jiang et al., CVPR 2025) is 67K procedurally generated indoor scenes
rendered in Blender, published as 100 zips of ~672 scenes each at
`hf.co/datasets/hwjiang/MegaSynth`. Input here is one directory per scene, as
`download_megasynth.py` stages it out of those zips:

    <src-root>/<scene>/hanwen/opencv_cameras.json
    <src-root>/<scene>/hanwen/renderings/<idx:08d>_rgba.png     512x512 RGBA
    <src-root>/<scene>/hanwen/renderings/<idx:08d>_depth.exr    512x512 float32

Output is the contract `preprocess_dl3dv.py` writes, so `DL3DVDataset` loads
MegaSynth with no changes at all:

    <out>/index.json
    <out>/scenes/megasynth/<scene>/meta.npz
    <out>/scenes/megasynth/<scene>/images/<stem>.jpg
    <out>/depth/megasynth/<scene>/<stem>.png      -- pass as --depth-root
    <out>/depth/megasynth/<scene>/meta.json

Why this is so much less work than the ScanNet or DL3DV paths: the renderer
already emits exactly what the contract wants. `opencv_cameras.json` stores
`w2c` in OpenCV convention (`render_scenes_rgbd.py` builds it as
`diag(1,-1,-1,1) @ inv(c2w)`, i.e. Blender's OpenGL frame already converted),
depth is Blender's Z pass on the *same* 512x512 grid as the colour, and there is
no COLMAP stage, no undistortion and no rig transform to reconcile. So every
scene takes the loader's dense branch and the sparse arrays (`points_xyz`,
`obs_frame`, `obs_point`) are written empty, exactly as for ScanNet.

Three things are specific to a Blender render and are the only real work here:

* **The Z pass is planar depth, not ray length.** Verified by reprojecting one
  frame's depth into a neighbouring frame: as planar z the median disagreement
  is 0.06% with no radial trend, as ray length it is 0.09% and drifts with
  distance from the principal point. Planar is what the loader's pinhole
  unprojection assumes, so the values go straight through.
* **Background pixels are `1e10`, not zero.** `render_scenes_rgbd.py` sets
  `film_transparent`, so a pixel that hits no geometry gets alpha 0, white RGB
  and the Z pass's sentinel. Those become mask zeros. Partially transparent
  pixels (anti-aliased silhouette edges, ~0.5% of a frame) go too: their colour
  is a blend of two surfaces and their depth is whichever one won.
* **Intrinsics vary per frame.** Every scene is 512x512 with the principal point
  dead centre, but the renderer jitters the focal length between views (fx
  ranges ~380-490), so `intrinsics` is genuinely per-frame rather than one K
  repeated. `plan_output` is therefore called per frame and only the *plan* --
  which depends on the shape and the principal point, not the focal -- is
  required to agree across the scene.

Scale: MegaSynth normalises each scene into a [-1, 1] box before rendering, so
depths are in arbitrary scene units, like DL3DV's COLMAP units and unlike
ScanNet's metres. The loader divides all three by their own mean point distance,
so all three arrive at the model in the same unit space.

Resolution: the source is square, so `--resolution 384` stores 384x384, which
stacks with nothing else in this repo. Use `--target-hw 224 384 --fit crop` to
make MegaSynth interchangeable with a DL3DV set built at `--resolution 384`;
that is what `download_megasynth.py` passes by default. The crop keeps the full
horizontal field of view (~65 deg) and cuts the vertical to ~39 deg.

Example:

    python training/preprocess_megasynth.py \
        --src-root ~/megasynth-raw --out ~/megasynth-train \
        --target-hw 224 384 --fit crop --workers 8

Usually you do not run this directly: `download_megasynth.py` calls it per scene
so each scene's ~60 MB of PNG/EXR can be deleted right after it is consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# opencv-python ships the OpenEXR codec disabled unless this is set, and it is
# read once at import time -- so it has to happen before `import cv2`, here and
# in every process the pool forks.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_scannet import geometric_covisibility, plan_output, quantise_depth  # noqa: E402

SUBSET = "megasynth"

# Blender's Z pass writes this where the ray hits nothing. Anything above the
# threshold is that sentinel, not a surface: the scene is normalised into a
# [-1, 1] box, so no real depth exceeds single digits.
DEPTH_SKY = 1e6

# `load_depth` treats a quantised value of 0 as "no ground truth", so the bottom
# rail has to sit strictly below anything the renderer can report. Blender's
# default near clip is 0.1 in these scenes (`render_scenes_rgbd.py` leaves
# `clip_start` at its default) and the observed minimum is 0.101, so 0.02 is
# comfortably under it with room for a scene rendered at a closer clip.
DEPTH_QUANT_LO = 0.02


# --------------------------------------------------------------------------- #
# per-scene worker
# --------------------------------------------------------------------------- #


def read_frame(rgba_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(BGR uint8, depth float32 with 0 = invalid) at the source 512x512, or None.

    Zeroing the invalid pixels here rather than carrying a separate mask is what
    lets the rest of the pipeline -- the resize, `geometric_covisibility`,
    `quantise_depth`, `load_depth` -- use the same "0 means no ground truth"
    convention the DL3DV and ScanNet paths already use.
    """
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]  # the Z pass is written to all three channels
    depth = depth.astype(np.float32)
    if rgba.shape[:2] != depth.shape[:2]:
        return None

    valid = np.isfinite(depth) & (depth > 0) & (depth < DEPTH_SKY)
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        # Strictly ==255: a partially transparent pixel is a silhouette edge, so
        # its depth is one of two surfaces and its colour is a blend of both.
        valid &= rgba[..., 3] == 255
        rgba = rgba[..., :3]
    return np.ascontiguousarray(rgba), np.where(valid, depth, 0.0).astype(np.float32)


def resample(bgr: np.ndarray, depth: np.ndarray, plan: tuple, out_hw: tuple[int, int]):
    """Apply `plan_output`'s plan to one frame's colour and depth."""
    out_h, out_w = out_hw
    _, inter_h, inter_w, crop_y, crop_x = plan
    if (bgr.shape[0], bgr.shape[1]) != (inter_h, inter_w):
        interp = cv2.INTER_AREA if inter_h < bgr.shape[0] else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, (inter_w, inter_h), interpolation=interp)
        # Nearest for depth: averaging across a discontinuity invents a surface
        # in front of both neighbours, and averaging across the 0 rail would
        # drag valid depths towards zero.
        depth = cv2.resize(depth, (inter_w, inter_h), interpolation=cv2.INTER_NEAREST)
    if (inter_h, inter_w) != (out_h, out_w):
        bgr = bgr[crop_y : crop_y + out_h, crop_x : crop_x + out_w]
        depth = depth[crop_y : crop_y + out_h, crop_x : crop_x + out_w]
    return bgr, depth


def process_scene(task: dict) -> dict:
    scene = task["scene"]
    out_dir = Path(task["out_dir"])
    depth_dir = Path(task["depth_dir"])
    meta_path = out_dir / "meta.npz"

    if meta_path.exists() and (depth_dir / "meta.json").exists() and not task["overwrite"]:
        with np.load(meta_path, allow_pickle=True) as data:
            return {
                "status": "cached",
                "scene": scene,
                "num_frames": int(len(data["frame_names"])),
                "image_hw": [int(v) for v in data["image_hw"]],
            }

    renderings = Path(task["scene_root"]) / "hanwen" / "renderings"
    cameras = json.loads((Path(task["scene_root"]) / "hanwen" / "opencv_cameras.json").read_text())

    # The downloader fetches a subset of the 48 rendered views to keep the
    # transfer down, so the camera file lists more frames than are on disk.
    # Pair by `file_path` and keep what is actually here.
    available = []
    for frame in cameras["frames"]:
        rgba = renderings / Path(frame["file_path"]).name
        depth = rgba.with_name(rgba.name.replace("_rgba.png", "_depth.exr"))
        if rgba.exists() and depth.exists():
            available.append((frame, rgba, depth))
    if len(available) < task["min_frames"]:
        return {
            "status": "skip",
            "reason": "too_few_rendered_frames",
            "scene": scene,
            "detail": f"{len(available)} of {len(cameras['frames'])}",
        }

    stride = max(1, math.ceil(len(available) / task["max_frames"]))
    available = available[::stride]

    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    frame_names, extrinsics, intrinsics, depths, depth_stats = [], [], [], [], []
    out_hw, plan = None, None
    for frame, rgba_path, depth_path in available:
        loaded = read_frame(rgba_path, depth_path)
        if loaded is None:
            continue
        bgr, depth = loaded

        src_h, src_w = int(frame["h"]), int(frame["w"])
        if (bgr.shape[0], bgr.shape[1]) != (src_h, src_w):
            return {
                "status": "skip",
                "reason": "render_shape_mismatch",
                "scene": scene,
                "detail": f"{bgr.shape[:2]} vs {(src_h, src_w)} in opencv_cameras.json",
            }
        K_src = np.array(
            [[frame["fx"], 0.0, frame["cx"]], [0.0, frame["fy"], frame["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        out_h, out_w, K_out, frame_plan = plan_output(
            K_src,
            (src_h, src_w),
            resolution=task["resolution"],
            mode=task["mode"],
            patch_size=task["patch_size"],
            target_hw=task["target_hw"],
            fit=task["fit"],
        )
        # Only the focal length is allowed to vary within a scene. The plan
        # carries the stored shape and the crop offset, which every frame has to
        # share or the frames cannot be stacked and the crop would move the
        # principal point away from the centre `K_out` claims it is at.
        if plan is None:
            out_hw, plan = (out_h, out_w), frame_plan
        elif frame_plan != plan:
            return {
                "status": "skip",
                "reason": "inconsistent_frame_geometry",
                "scene": scene,
                "detail": f"{frame_plan} vs {plan}",
            }

        bgr, depth = resample(bgr, depth, plan, out_hw)
        good = depth > 0
        if good.mean() < task["min_depth_frac"]:
            continue

        stem = Path(frame["file_path"]).name.replace("_rgba.png", "")
        cv2.imwrite(str(images_out / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, task["jpeg_quality"]])

        w2c = np.asarray(frame["w2c"], dtype=np.float64)
        frame_names.append(f"{stem}.jpg")
        extrinsics.append(w2c[:3])
        intrinsics.append(K_out)
        depths.append(depth)
        depth_stats.append(np.percentile(depth[good], [5, 50, 95]))

    if len(frame_names) < task["min_frames"]:
        return {
            "status": "skip",
            "reason": "too_few_usable_frames",
            "scene": scene,
            "detail": f"{len(frame_names)} kept",
        }

    out_h, out_w = out_hw
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)
    depth_stack = np.stack(depths)

    covisibility = geometric_covisibility(
        depth_stack, extrinsics, intrinsics, grid=tuple(task["covis_grid"]), rel_tol=task["covis_rel_tol"]
    )

    quantised, lo, hi = quantise_depth(depth_stack, lo=DEPTH_QUANT_LO)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for name, plane in zip(frame_names, quantised):
        cv2.imwrite(str(depth_dir / f"{Path(name).stem}.png"), plane)
    (depth_dir / "meta.json").write_text(
        json.dumps(
            {
                "lo": lo,
                "hi": hi,
                "out_hw": [out_h, out_w],
                "frames": [Path(n).stem for n in frame_names],
                "source": "megasynth_blender_z",
            }
        )
    )

    np.savez(
        meta_path,
        frame_names=np.array(frame_names),
        extrinsics=extrinsics.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        image_hw=np.array([out_h, out_w], dtype=np.int32),
        # Empty by design: a render has no triangulated points and never needs
        # the loader's sparse fallback. See the module docstring.
        points_xyz=np.zeros((0, 3), dtype=np.float32),
        points_error=np.zeros((0,), dtype=np.float32),
        obs_frame=np.zeros((0,), dtype=np.int32),
        obs_point=np.zeros((0,), dtype=np.int32),
        depth_stats=np.stack(depth_stats).astype(np.float32),
        covisibility=covisibility,
        subset=np.array(SUBSET),
        scene_id=np.array(scene),
    )

    return {
        "status": "ok",
        "scene": scene,
        "num_frames": len(frame_names),
        "image_hw": [out_h, out_w],
        "depth_valid": float((depth_stack > 0).mean()),
        "median_depth": float(np.median(depth_stats, axis=0)[1]),
    }


def safe_process(task: dict) -> dict:
    """`process_scene`, with any exception turned into an error result.

    Public because `download_megasynth.py` is a second caller: one malformed
    scene out of thousands must not take a multi-hour run down.
    """
    try:
        return process_scene(task)
    except Exception as exc:  # a half-staged scene shouldn't kill the run
        return {
            "status": "error",
            "scene": task["scene"],
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def discover_scenes(src_root: Path) -> list[tuple[str, Path]]:
    """Find `<scene>/hanwen/opencv_cameras.json`, at any depth under `src_root`.

    Any depth because the zips nest scenes one level down (`split_7/<uuid>/...`),
    so an extracted tree and a staged one both work without a --layout flag.
    """
    found = {}
    for path in sorted(src_root.rglob("hanwen/opencv_cameras.json")):
        scene_root = path.parent.parent
        found.setdefault(scene_root.name, scene_root)
    return sorted(found.items())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", type=Path, required=True, help="dir containing <scene>/hanwen/...")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth-out", type=Path, default=None, help="default: <out>/depth")
    p.add_argument("--scenes", nargs="*", default=None, help="process only these scene ids")

    p.add_argument("--resolution", type=int, default=384,
                   help="aspect-preserving sizing; the source is square, so this stores 384x384")
    p.add_argument("--target-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="force an exact stored shape, overriding --resolution. Use 224 384 to make "
                        "MegaSynth interchangeable with a DL3DV set built at --resolution 384")
    p.add_argument("--fit", choices=("crop", "squash"), default="crop",
                   help="how --target-hw is reached: 'crop' keeps natural proportions and loses "
                        "field of view; 'squash' keeps the full view but stretches the image")
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--jpeg-quality", type=int, default=95)

    p.add_argument("--max-frames", type=int, default=48, help="cap frames per scene (48 are rendered)")
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--min-depth-frac", type=float, default=0.2,
                   help="drop frames this empty; a MegaSynth frame is mostly enclosed geometry")
    p.add_argument("--covis-grid", type=int, nargs=2, default=(24, 32))
    p.add_argument("--covis-rel-tol", type=float, default=0.1)

    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true", help="one line per scene, for the streaming driver")
    p.add_argument("--no-index", action="store_true",
                   help="skip the index write; for parallel single-scene workers")
    p.add_argument("--index-only", action="store_true",
                   help="rebuild index.json from the meta.npz already on disk, then exit")
    p.add_argument("--dry-run", action="store_true")
    return p


def assign_splits(scene_ids, val_frac: float, seed: int) -> dict[str, str]:
    """Random holdout, hashed off the scene id rather than drawn positionally.

    There is no official split and no notion of two scans of one space to keep
    together -- every scene is an independent procedural sample. Hashing means a
    scene lands on the same side however many scenes were downloaded before it,
    so growing the set never reshuffles the holdout.
    """
    if val_frac <= 0:
        return {s: "train" for s in scene_ids}
    out = {}
    for scene in scene_ids:
        digest = hashlib.blake2b(f"{seed}:{scene}".encode(), digest_size=8).digest()
        out[scene] = "val" if int.from_bytes(digest, "big") / 2.0**64 < val_frac else "train"
    return out


def run_config(args, depth_out: Path) -> dict:
    return {
        "src_root": str(args.src_root),
        "depth_root": str(depth_out),
        "resolution": args.resolution,
        "target_hw": list(args.target_hw) if args.target_hw else None,
        "fit": args.fit,
        "mode": args.mode,
        "patch_size": args.patch_size,
        "max_frames": args.max_frames,
        "min_depth_frac": args.min_depth_frac,
        "splits": f"random({args.val_frac})",
    }


def entry_for(subset: str, scene: str, split: str, num_frames, image_hw) -> dict:
    return {
        "subset": subset,
        "scene": scene,
        "split": split,
        "path": f"scenes/{subset}/{scene}",
        "num_frames": num_frames,
        "num_points": 0,
        "image_hw": image_hw,
        # Always true: a render's depth is exact, so every scene takes the
        # loader's dense branch.
        "has_depth": True,
    }


def write_index(out_root: Path, entries: list[dict], config: dict) -> dict:
    index = {
        "config": config,
        "num_train": sum(e["split"] == "train" for e in entries),
        "num_val": sum(e["split"] == "val" for e in entries),
        "scenes": entries,
    }
    tmp = (out_root / "index.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=1))
    tmp.replace(out_root / "index.json")  # atomic, so a reader never sees a half file
    return index


def merge_index(out_root: Path, results: list[dict], config: dict) -> dict:
    """Rewrite `index.json`, folding in whatever a previous run already recorded."""
    index_path = out_root / "index.json"
    existing = {}
    if index_path.exists():
        try:
            for entry in json.loads(index_path.read_text()).get("scenes", []):
                existing[entry["scene"]] = entry
        except json.JSONDecodeError:
            pass  # a run killed mid-write; rebuild from what we have
    for entry in results:
        existing[entry["scene"]] = entry
    return write_index(out_root, sorted(existing.values(), key=lambda e: e["scene"]), config)


def rebuild_index(out_root: Path, val_frac: float, seed: int, config: dict) -> dict:
    """Assemble `index.json` from the `meta.npz` files already on disk.

    The streaming driver runs many single-scene workers concurrently, and a
    read-modify-write of one shared index from N of them loses entries no matter
    how atomic the write is. So workers pass `--no-index` and write nothing
    shared, and the index is derived from the per-scene artefacts afterwards,
    which is idempotent and order-independent.
    """
    scenes = sorted(p.parent.name for p in (out_root / "scenes" / SUBSET).glob("*/meta.npz"))
    splits = assign_splits(scenes, val_frac, seed)
    entries = []
    for scene in scenes:
        try:
            with np.load(out_root / "scenes" / SUBSET / scene / "meta.npz", allow_pickle=True) as data:
                num_frames = int(len(data["frame_names"]))
                image_hw = [int(v) for v in data["image_hw"]]
        except Exception:
            continue  # a worker killed mid-write; the next run will redo it
        entries.append(entry_for(SUBSET, scene, splits[scene], num_frames, image_hw))
    return write_index(out_root, entries, config)


def consolidate_failures(out_root: Path) -> int:
    """Fold the per-scene failure sidecars into one `failures.json`."""
    sidecars = out_root / "failures"
    if not sidecars.is_dir():
        return 0
    by_scene = {}
    path = out_root / "failures.json"
    if path.exists():
        try:
            by_scene = {f["scene"]: f for f in json.loads(path.read_text())}
        except json.JSONDecodeError:
            pass
    for sidecar in sidecars.glob("*.json"):
        try:
            failure = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        by_scene[failure["scene"]] = failure
    if by_scene:
        path.write_text(json.dumps(sorted(by_scene.values(), key=lambda f: f["scene"]), indent=1))
    return len(by_scene)


def make_task(args, scene: str, scene_root: Path, depth_out: Path) -> dict:
    return {
        "scene": scene,
        "scene_root": str(scene_root),
        "out_dir": str(args.out / "scenes" / SUBSET / scene),
        "depth_dir": str(depth_out / SUBSET / scene),
        "resolution": args.resolution,
        "target_hw": tuple(args.target_hw) if args.target_hw else None,
        "fit": args.fit,
        "mode": args.mode,
        "patch_size": args.patch_size,
        "jpeg_quality": args.jpeg_quality,
        "max_frames": max(1, args.max_frames),
        "min_frames": args.min_frames,
        "min_depth_frac": args.min_depth_frac,
        "covis_grid": list(args.covis_grid),
        "covis_rel_tol": args.covis_rel_tol,
        "overwrite": args.overwrite,
    }


def main() -> int:
    args = build_parser().parse_args()
    depth_out = args.depth_out or (args.out / "depth")

    if args.index_only:
        # The staged renders are long gone by now, so take the scene universe
        # from the output tree rather than from --src-root.
        index = rebuild_index(args.out, args.val_frac, args.seed, run_config(args, depth_out))
        if not index["scenes"]:
            print(f"no processed scenes under {args.out}", file=sys.stderr)
            return 1
        consolidate_failures(args.out)
        total = sum(e["num_frames"] or 0 for e in index["scenes"])
        print(
            f"wrote {args.out / 'index.json'}: {len(index['scenes'])} scenes "
            f"({index['num_train']} train / {index['num_val']} val), {total} frames"
        )
        return 0

    scenes = discover_scenes(args.src_root)
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [(s, p) for s, p in scenes if s in wanted]
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        print(f"no <scene>/hanwen/opencv_cameras.json under {args.src_root}", file=sys.stderr)
        return 1

    splits = assign_splits([s for s, _ in scenes], args.val_frac, args.seed)

    if args.dry_run:
        for scene, path in scenes[:20]:
            print(f"  [{splits[scene]}] {scene}  {path}")
        print(f"  ... {len(scenes)} scenes, {sum(v == 'val' for v in splits.values())} val")
        return 0

    tasks = [make_task(args, scene, path, depth_out) for scene, path in scenes]

    entries, failures = [], []
    counts = {"ok": 0, "cached": 0, "skip": 0, "error": 0}
    started = time.time()

    def record(result: dict) -> None:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        scene = result["scene"]
        if result["status"] in ("ok", "cached"):
            entries.append(
                entry_for(SUBSET, scene, splits[scene], result.get("num_frames"), result.get("image_hw"))
            )
            if args.quiet:
                print(
                    f"{scene} {result['status']} frames={result.get('num_frames')} "
                    f"valid={100 * result.get('depth_valid', 0):.1f}%",
                    flush=True,
                )
        else:
            failures.append({k: result.get(k) for k in ("scene", "status", "reason", "detail")})
            if args.quiet:
                print(f"{scene} {result['status']} {result.get('reason')}", flush=True)

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            record(safe_process(task))
            if not args.quiet:
                print(f"\r[{i + 1}/{len(tasks)}] {counts}", end="", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(safe_process, task) for task in tasks]
            for i, future in enumerate(as_completed(futures)):
                record(future.result())
                if not args.quiet:
                    elapsed = time.time() - started
                    rate = (i + 1) / max(elapsed, 1e-6)
                    print(
                        f"\r[{i + 1}/{len(tasks)}] {counts} {rate:.2f} scenes/s "
                        f"eta {(len(tasks) - i - 1) / max(rate, 1e-6) / 60:.1f}m   ",
                        end="",
                        flush=True,
                    )
    if not args.quiet:
        print()

    args.out.mkdir(parents=True, exist_ok=True)

    # Under --no-index every shared file is off limits, because N single-scene
    # workers run concurrently. Failures go to one sidecar per scene instead;
    # --index-only folds them back into failures.json.
    if failures:
        if args.no_index:
            sidecars = args.out / "failures"
            sidecars.mkdir(parents=True, exist_ok=True)
            for failure in failures:
                (sidecars / f"{failure['scene']}.json").write_text(json.dumps(failure))
        else:
            path = args.out / "failures.json"
            prior = json.loads(path.read_text()) if path.exists() else []
            by_scene = {f["scene"]: f for f in prior}
            by_scene.update({f["scene"]: f for f in failures})
            path.write_text(json.dumps(sorted(by_scene.values(), key=lambda f: f["scene"]), indent=1))

    if args.no_index:
        return 0

    index = merge_index(args.out, entries, run_config(args, depth_out))
    if not args.quiet:
        total_frames = sum(e["num_frames"] or 0 for e in index["scenes"])
        print(
            f"wrote {args.out / 'index.json'}: {len(index['scenes'])} scenes cumulative "
            f"({index['num_train']} train / {index['num_val']} val), {total_frames} frames; "
            f"this run {counts} in {(time.time() - started) / 60:.1f}m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
