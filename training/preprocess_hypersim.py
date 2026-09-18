#!/usr/bin/env python3
"""Turn staged Hypersim frames into the same training set format as DL3DV.

Hypersim (McCormac et al., 2021) is 461 photorealistic synthetic indoor scenes
rendered in V-Ray, published as one ~4 GB zip per scene at
`docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/`.
Input here is one directory per `<scene>/<camera>`, as `download_hypersim.py`
stages it out of those zips (only the members actually needed, never the whole
4 GB -- see that script's docstring):

    <src-root>/<scene>/<camera>/frame.<frame_id>.tonemap.jpg           1024x768 RGB
    <src-root>/<scene>/<camera>/frame.<frame_id>.depth_meters.npy      1024x768 f32, ray distance
    <src-root>/<scene>/<camera>/positions_m.npy                        (F,3) f64, camera position, metres
    <src-root>/<scene>/<camera>/orientations.npy                       (F,3,3) f64, cam-to-world, OpenGL axes
    <src-root>/<scene>/<camera>/frame_ids.json                         (F,) the frame_id each row corresponds to

Output is the contract `preprocess_dl3dv.py` writes, so `DL3DVDataset` loads
Hypersim with no changes at all:

    <out>/index.json
    <out>/scenes/hypersim/<scene>_<camera>/meta.npz
    <out>/scenes/hypersim/<scene>_<camera>/images/<stem>.jpg
    <out>/depth/hypersim/<scene>_<camera>/<stem>.png      -- pass as --depth-root
    <out>/depth/hypersim/<scene>_<camera>/meta.json

Scene unit. A Hypersim "scene" carries several cameras, and two cameras in one
scene are typically different rooms or trajectories through the same asset --
the same situation as ScanNet's `scene0011_00`/`scene0011_01`, which that
preprocessor also treats as independent scans. So the training-set "scene" here
is `<scene>_<camera>`, not `<scene>` alone, and every array in this file is
built from one camera's frames only.

Three things are specific to this dataset and are the only real work here:

* **Depth is Euclidean ray distance, not planar z.** Hypersim's
  `depth_meters.hdf5` (already unpacked to `.npy` by the downloader) is the
  distance from the camera centre to the surface along the ray, not the
  z-coordinate the loader's pinhole unprojection assumes. `distance_to_z`
  converts with the fixed intrinsics below -- this is the standard conversion
  used by every public Hypersim loader (e.g. Depth-Anything-V2's metric_depth
  training set).
* **Intrinsics are a fixed approximation, not read per-scene.** Every public
  loader for this dataset (again e.g. Depth-Anything-V2) uses one pinhole for
  all of Hypersim: 1024x768, fx=fy=886.81, principal point at image-centre-
  minus-half-pixel. A minority of scenes actually use non-standard tilt-shift
  projection (`metadata_camera_parameters.csv` in the ml-hypersim repo), which
  this silently gets slightly wrong for -- accepted here as a known,
  field-standard approximation rather than something to "fix" by re-deriving a
  per-scene projection matrix from an undocumented ray-casting formula.
* **Extrinsics are OpenGL axes, not OpenCV.** `orientations.npy` is
  camera-to-world with +y up, +z backward (the camera looks down -z); the
  contract wants OpenCV's +y down, +z forward, world-to-camera. See
  `opengl_to_opencv_extrinsic`.

Depth quantisation, geometric covisibility and `plan_output`-style output
sizing are dataset-agnostic and imported from `preprocess_scannet.py` rather
than reimplemented.

Every scene takes the loader's dense depth branch (like ScanNet and MegaSynth):
there is no sparse COLMAP reconstruction, so `points_xyz`/`obs_frame`/
`obs_point` are written empty.

Scale: Hypersim's positions are metres, like ScanNet's and unlike DL3DV's
arbitrary COLMAP units. The loader normalises every scene by its own mean point
distance, so the metric scale is divided out the same way ScanNet's is.

Example:

    python training/preprocess_hypersim.py \
        --src-root ~/hypersim-raw --out ~/hypersim-train \
        --target-hw 224 384 --fit crop --workers 8

Usually you do not run this directly: `download_hypersim.py` calls it per
`<scene>/<camera>` so each one's staged frames can be deleted right after use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_scannet import geometric_covisibility, plan_output, quantise_depth  # noqa: E402

SUBSET = "hypersim"

# The fixed pinhole every public Hypersim loader uses (Depth-Anything-V2's
# metric_depth/dataset/hypersim.py, among others). See the module docstring for
# why this is an accepted approximation rather than a per-scene read.
SRC_W, SRC_H = 1024, 768
FOCAL = 886.81

# `load_depth` treats a quantised value of 0 as "no ground truth", so the bottom
# rail has to sit strictly below anything a real render reports. Hypersim rooms
# are interiors with sub-metre near surfaces possible but not below this.
DEPTH_QUANT_LO = 0.02

# Rows of `depth_meters.hdf5` that never hit geometry (background through a
# window, etc.) come back as NaN, not a sentinel value like MegaSynth's 1e10.
_FLIP_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def distance_to_z(distance: np.ndarray) -> np.ndarray:
    """Euclidean ray distance (metres) -> planar z-depth, for the fixed pinhole.

    `distance` is |ray| from the camera centre to the surface; the loader's
    pinhole unprojection wants the z-component of that ray instead. The ratio
    z/|ray| is exactly focal / |plane pixel ray|, a per-pixel constant that does
    not depend on the scene, so it is computed once and reused for every frame.
    """
    x = np.linspace((-0.5 * SRC_W) + 0.5, (0.5 * SRC_W) - 0.5, SRC_W)
    y = np.linspace((-0.5 * SRC_H) + 0.5, (0.5 * SRC_H) - 0.5, SRC_H)
    xx, yy = np.meshgrid(x, y)
    ray_len = np.sqrt(xx**2 + yy**2 + FOCAL**2)
    return distance / ray_len * FOCAL


def opengl_to_opencv_extrinsic(R_world_from_cam_gl: np.ndarray, position_m: np.ndarray) -> np.ndarray:
    """(3,4) world-to-camera, OpenCV convention, from Hypersim's raw camera pose.

    `orientations.npy` is camera-to-world rotation in OpenGL axes (+x right,
    +y up, +z backward -- the camera looks down -z). `flip = diag(1,-1,-1)`
    re-expresses the same rotation in OpenCV axes (+x right, +y down, +z
    forward) without changing which way is "right" in the image; inverting that
    (transpose, since it is a rotation) gives cam-from-world, and the
    translation follows from `t = -R @ position`.
    """
    R_cv = R_world_from_cam_gl @ _FLIP_GL_TO_CV
    R_cam_from_world = R_cv.T
    t = -R_cam_from_world @ position_m
    return np.concatenate([R_cam_from_world, t[:, None]], axis=1)


def fixed_intrinsics() -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = FOCAL
    K[1, 1] = FOCAL
    K[0, 2] = SRC_W / 2.0 - 0.5
    K[1, 2] = SRC_H / 2.0 - 0.5
    return K


# --------------------------------------------------------------------------- #
# per-scene worker
# --------------------------------------------------------------------------- #


def process_scene(task: dict) -> dict:
    scene = task["scene"]
    src_dir = Path(task["src_dir"])
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

    frame_ids = json.loads((src_dir / "frame_ids.json").read_text())
    positions = np.load(src_dir / "positions_m.npy")
    orientations = np.load(src_dir / "orientations.npy")
    if len(frame_ids) < task["min_frames"]:
        return {
            "status": "skip",
            "reason": "too_few_available_frames",
            "scene": scene,
            "detail": f"{len(frame_ids)} listed",
        }

    stride = max(1, math.ceil(len(frame_ids) / task["max_frames"]))
    keep = list(range(0, len(frame_ids), stride))

    K_src = fixed_intrinsics()
    out_h, out_w, K_out, plan = plan_output(
        K_src,
        (SRC_H, SRC_W),
        resolution=task["resolution"],
        mode=task["mode"],
        patch_size=task["patch_size"],
        target_hw=task["target_hw"],
        fit=task["fit"],
    )
    _, inter_h, inter_w, crop_y, crop_x = plan

    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    frame_names, extrinsics, depths, depth_stats = [], [], [], []
    for i in keep:
        frame_id = frame_ids[i]
        jpg_path = src_dir / f"frame.{frame_id}.tonemap.jpg"
        depth_path = src_dir / f"frame.{frame_id}.depth_meters.npy"
        if not (jpg_path.exists() and depth_path.exists()):
            continue

        bgr = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
        distance = np.load(depth_path).astype(np.float32)
        # A scene whose render does not match the fixed pinhole's assumed shape
        # would silently mis-decode depth through the wrong ray geometry, so
        # refuse it rather than guess.
        assert distance.shape == (SRC_H, SRC_W), f"{scene}: depth_meters shape {distance.shape}"
        assert bgr is not None and bgr.shape[:2] == (SRC_H, SRC_W), f"{scene}: tonemap shape mismatch"

        depth_z = distance_to_z(distance)
        good = np.isfinite(depth_z) & (depth_z > 0)
        depth_z = np.where(good, depth_z, 0.0).astype(np.float32)

        if (bgr.shape[0], bgr.shape[1]) != (inter_h, inter_w):
            interp = cv2.INTER_AREA if inter_h < bgr.shape[0] else cv2.INTER_CUBIC
            bgr = cv2.resize(bgr, (inter_w, inter_h), interpolation=interp)
            depth_z = cv2.resize(depth_z, (inter_w, inter_h), interpolation=cv2.INTER_NEAREST)
        if (inter_h, inter_w) != (out_h, out_w):
            bgr = bgr[crop_y : crop_y + out_h, crop_x : crop_x + out_w]
            depth_z = depth_z[crop_y : crop_y + out_h, crop_x : crop_x + out_w]

        good = depth_z > 0
        if good.mean() < task["min_depth_frac"]:
            continue

        stem = f"frame_{frame_id}"
        cv2.imwrite(str(images_out / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, task["jpeg_quality"]])

        cam_from_world = opengl_to_opencv_extrinsic(orientations[i], positions[i])
        frame_names.append(f"{stem}.jpg")
        extrinsics.append(cam_from_world)
        depths.append(depth_z)
        depth_stats.append(np.percentile(depth_z[good], [5, 50, 95]))

    if len(frame_names) < task["min_frames"]:
        return {
            "status": "skip",
            "reason": "too_few_usable_frames",
            "scene": scene,
            "detail": f"{len(frame_names)} kept",
        }

    extrinsics = np.stack(extrinsics)
    intrinsics = np.repeat(K_out[None], len(frame_names), axis=0)
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
                "source": "hypersim_ray_distance",
            }
        )
    )

    np.savez(
        meta_path,
        frame_names=np.array(frame_names),
        extrinsics=extrinsics.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        image_hw=np.array([out_h, out_w], dtype=np.int32),
        # Empty by design: a V-Ray render has no triangulated points and never
        # needs the loader's sparse fallback. See the module docstring.
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
    try:
        return process_scene(task)
    except Exception as exc:  # one malformed scene shouldn't kill the run
        return {
            "status": "error",
            "scene": task["scene"],
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def discover_scans(src_root: Path) -> list[tuple[str, Path]]:
    """Find `<scene>/<camera>/frame_ids.json`, i.e. one entry per staged camera."""
    found = {}
    for path in sorted(src_root.glob("*/*/frame_ids.json")):
        camera_dir = path.parent
        scene = f"{camera_dir.parent.name}_{camera_dir.name}"
        found.setdefault(scene, camera_dir)
    return sorted(found.items())


def load_split(split_csv: Path | None) -> dict[str, str]:
    """Hypersim's official train/test partition, keyed by `<scene>_<camera>`.

    `split_csv` is `metadata_images_split_scene_v1.csv`
    (apple/ml-hypersim, `evermotion_dataset/analysis/`), which
    `download_hypersim.py` fetches once. `split_partition_name` is assigned per
    scene in the source, not per camera, but this reads it per row and takes the
    majority per `<scene>_<camera>` in case of the rare inconsistency, rather
    than assuming uniformity holds everywhere.
    """
    if split_csv is None or not split_csv.exists():
        return {}
    import csv

    votes: dict[str, dict[str, int]] = {}
    with split_csv.open() as f:
        for row in csv.DictReader(f):
            if row["included_in_public_release"].strip().lower() != "true":
                continue
            key = f"{row['scene_name']}_{row['camera_name']}"
            partition = "val" if row["split_partition_name"].strip().lower() == "test" else "train"
            votes.setdefault(key, {"train": 0, "val": 0})[partition] += 1
    return {key: max(counts, key=counts.get) for key, counts in votes.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", type=Path, required=True, help="dir containing <scene>/<camera>/...")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth-out", type=Path, default=None, help="default: <out>/depth")
    p.add_argument("--scenes", nargs="*", default=None, help="process only these <scene>_<camera> ids")
    p.add_argument("--split-csv", type=Path, default=None,
                   help="metadata_images_split_scene_v1.csv; falls back to --val-frac")

    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--target-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="force an exact stored shape, overriding --resolution. Use 224 384 to make "
                        "Hypersim interchangeable with a DL3DV set built at --resolution 384")
    p.add_argument("--fit", choices=("crop", "squash"), default="crop",
                   help="how --target-hw is reached: 'crop' keeps natural proportions and loses "
                        "field of view; 'squash' keeps the full view but stretches the image")
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--jpeg-quality", type=int, default=95)

    p.add_argument("--max-frames", type=int, default=100, help="cap frames per camera")
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--min-depth-frac", type=float, default=0.2)
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


def assign_splits(scene_ids, split_csv: Path | None, val_frac: float, seed: int) -> dict[str, str]:
    official = load_split(split_csv)
    if official:
        return {s: official.get(s, "train") for s in scene_ids}
    import hashlib

    out = {}
    for scene in scene_ids:
        digest = hashlib.blake2b(f"{seed}:{scene}".encode(), digest_size=8).digest()
        out[scene] = "val" if int.from_bytes(digest, "big") / 2.0**64 < val_frac else "train"
    return out


def run_config(args, depth_out: Path, split_csv_used: bool) -> dict:
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
        "splits": "official" if split_csv_used else f"random({args.val_frac})",
    }


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

    scenes = sorted(existing.values(), key=lambda e: e["scene"])
    index = {
        "config": config,
        "num_train": sum(e["split"] == "train" for e in scenes),
        "num_val": sum(e["split"] == "val" for e in scenes),
        "scenes": scenes,
    }
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=1))
    tmp.replace(index_path)  # atomic, so a concurrent reader never sees a half file
    return index


def rebuild_index(out_root: Path, splits: dict[str, str], config: dict) -> dict:
    """Assemble `index.json` from the `meta.npz` files already on disk.

    Mirrors `preprocess_scannet.py`'s `rebuild_index`: the streaming driver runs
    many single-scan workers in parallel, so the index is derived from the
    per-scene artefacts afterwards rather than read-modify-written by N workers.
    """
    entries = []
    for meta_path in sorted((out_root / "scenes" / SUBSET).glob("*/meta.npz")):
        scene = meta_path.parent.name
        try:
            with np.load(meta_path, allow_pickle=True) as data:
                num_frames = int(len(data["frame_names"]))
                image_hw = [int(v) for v in data["image_hw"]]
        except Exception:
            continue  # a worker killed mid-write; the next run will redo it
        entries.append(
            {
                "subset": SUBSET,
                "scene": scene,
                "split": splits.get(scene, "train"),
                "path": f"scenes/{SUBSET}/{scene}",
                "num_frames": num_frames,
                "num_points": 0,
                "image_hw": image_hw,
                "has_depth": True,
            }
        )
    index = {
        "config": config,
        "num_train": sum(e["split"] == "train" for e in entries),
        "num_val": sum(e["split"] == "val" for e in entries),
        "scenes": entries,
    }
    tmp = (out_root / "index.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=1))
    tmp.replace(out_root / "index.json")
    return index


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


def make_task(args, scene: str, src_dir: Path, depth_out: Path) -> dict:
    return {
        "scene": scene,
        "src_dir": str(src_dir),
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
        # The staged frames are long gone by now, so take the scene universe
        # from the output tree rather than from --src-root.
        found = sorted(d.name for d in (args.out / "scenes" / SUBSET).glob("*") if (d / "meta.npz").exists())
        if not found:
            print(f"no processed scenes under {args.out}", file=sys.stderr)
            return 1
        splits = assign_splits(found, args.split_csv, args.val_frac, args.seed)
        index = rebuild_index(args.out, splits, run_config(args, depth_out, bool(load_split(args.split_csv))))
        consolidate_failures(args.out)
        total = sum(e["num_frames"] or 0 for e in index["scenes"])
        print(
            f"wrote {args.out / 'index.json'}: {len(index['scenes'])} scenes "
            f"({index['num_train']} train / {index['num_val']} val), {total} frames"
        )
        return 0

    scans = discover_scans(args.src_root)
    if args.scenes:
        wanted = set(args.scenes)
        scans = [(s, p) for s, p in scans if s in wanted]
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        print(f"no <scene>/<camera>/frame_ids.json under {args.src_root}", file=sys.stderr)
        return 1

    assignment = load_split(args.split_csv)
    splits = assign_splits([s for s, _ in scans], args.split_csv, args.val_frac, args.seed)

    if args.dry_run:
        for scene, path in scans[:20]:
            print(f"  [{splits[scene]}] {scene}  {path}")
        print(f"  ... {len(scans)} scans, {sum(v == 'val' for v in splits.values())} val")
        return 0

    tasks = [make_task(args, scene, path, depth_out) for scene, path in scans]

    entries, failures = [], []
    counts = {"ok": 0, "cached": 0, "skip": 0, "error": 0}
    started = time.time()

    def record(result: dict) -> None:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        scene = result["scene"]
        if result["status"] in ("ok", "cached"):
            entries.append(
                {
                    "subset": SUBSET,
                    "scene": scene,
                    "split": splits[scene],
                    "path": f"scenes/{SUBSET}/{scene}",
                    "num_frames": result.get("num_frames"),
                    "num_points": 0,
                    "image_hw": result.get("image_hw"),
                    # Always true: a render's depth is exact, so every scene
                    # takes the loader's dense branch.
                    "has_depth": True,
                }
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
                        f"\r[{i + 1}/{len(tasks)}] {counts} {rate:.2f} scans/s "
                        f"eta {(len(tasks) - i - 1) / max(rate, 1e-6) / 60:.1f}m   ",
                        end="",
                        flush=True,
                    )
    if not args.quiet:
        print()

    args.out.mkdir(parents=True, exist_ok=True)

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

    index = merge_index(args.out, entries, run_config(args, depth_out, bool(assignment)))
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
