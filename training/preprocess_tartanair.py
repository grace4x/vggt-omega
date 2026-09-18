#!/usr/bin/env python3
"""Turn staged TartanAir frames into the same training set format as DL3DV.

TartanAir (Wang et al., 2020) is ~1000 stereo trajectories across ~30
photorealistic AirSim environments. Input here is one directory per
`<env>/<difficulty>/<traj>`, as `download_tartanair.py` stages it out of the
`image_left.zip`/`depth_left.zip` pair for that env/difficulty (only the
members actually needed, never either whole zip -- see that script's
docstring):

    <src-root>/<env>/<difficulty>/<traj>/<frame>_left.png              640x480 RGB
    <src-root>/<env>/<difficulty>/<traj>/<frame>_left_depth.npy        640x480 f32, planar z-depth, metres
    <src-root>/<env>/<difficulty>/<traj>/poses.npy                     (F,7) f64, tx ty tz qx qy qz qw
    <src-root>/<env>/<difficulty>/<traj>/frame_ids.json                (F,) the frame id each row corresponds to

Output is the contract `preprocess_dl3dv.py` writes, so `DL3DVDataset` loads
TartanAir with no changes at all:

    <out>/index.json
    <out>/scenes/tartanair/<env>_<difficulty>_<traj>/meta.npz
    <out>/scenes/tartanair/<env>_<difficulty>_<traj>/images/<stem>.jpg
    <out>/depth/tartanair/<env>_<difficulty>_<traj>/<stem>.png      -- pass as --depth-root
    <out>/depth/tartanair/<env>_<difficulty>_<traj>/meta.json

Scene unit. Each `<env>/<difficulty>` holds several independent trajectories
(`P000`, `P001`, ...) through the same environment, the same situation as
Hypersim's several cameras per scene -- so the training-set "scene" here is
`<env>_<difficulty>_<traj>`, not `<env>` alone.

Three things are specific to this dataset and are the only real work here:

* **Depth is already planar z-depth, not a ray distance.** Unlike Hypersim's
  `depth_meters.hdf5`, TartanAir's `*_depth.npy` is exactly what the loader's
  pinhole unprojection wants -- no `distance_to_z`-style conversion. Sky/other
  infinitely-distant pixels are written as a large sentinel (documented as
  10000) rather than inf or NaN, so anything past `DEPTH_MAX` is masked out
  as invalid instead of being trusted as real geometry.
* **Intrinsics are a fixed pinhole, not read per-scene.** Every trajectory in
  the dataset shares one camera: 640x480, fx=fy=320, principal point at the
  image centre (see `data_type.md` in castacks/tartanair_tools) -- there is no
  per-scene calibration to read.
* **Extrinsics are NED, not OpenCV.** `poses.npy` rows are `tx ty tz qx qy qz
  qw`: `t` is the camera's position in a world-NED frame, and the quaternion
  is the camera's orientation in that same frame -- but critically the
  camera's own local axes are also NED-style (+x forward, +y right, +z down),
  not vision-style (+x right, +y down, +z forward). `ned_to_opencv_extrinsic`
  converts.

Depth quantisation, geometric covisibility and `plan_output`-style output
sizing are dataset-agnostic and imported from `preprocess_scannet.py` rather
than reimplemented.

Every scene takes the loader's dense depth branch (like ScanNet, MegaSynth and
Hypersim): there is no sparse COLMAP reconstruction, so `points_xyz`/
`obs_frame`/`obs_point` are written empty.

Scale: TartanAir's positions are metres, like ScanNet's and Hypersim's and
unlike DL3DV's arbitrary COLMAP units. The loader normalises every scene by its
own mean point distance, so the metric scale is divided out the same way.

Example:

    python training/preprocess_tartanair.py \
        --src-root ~/tartanair-raw --out ~/tartanair-train \
        --target-hw 224 384 --fit crop --workers 8

Usually you do not run this directly: `download_tartanair.py` calls it per
`<env>/<difficulty>/<traj>` so each one's staged frames can be deleted right
after use.
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

SUBSET = "tartanair"

# The one fixed pinhole every TartanAir trajectory shares (castacks/
# tartanair_tools `data_type.md`).
SRC_W, SRC_H = 640, 480
FOCAL = 320.0

# `depth_left.zip` writes far-away background (sky, etc.) as a large sentinel
# (documented as 10000) rather than inf/NaN. No real indoor/outdoor scene in
# the dataset has geometry anywhere near this far, so it is a safe cutoff.
DEPTH_MAX = 1000.0

# `load_depth` treats a quantised value of 0 as "no ground truth", so the
# bottom rail has to sit strictly below anything a real render reports.
DEPTH_QUANT_LO = 0.02

# NED camera axes are (+x forward, +y right, +z down); OpenCV's are (+x right,
# +y down, +z forward). Reordering columns of a NED-frame rotation matrix by
# this permutation re-expresses the same rotation in OpenCV axes, without
# touching the (still-NED) world frame it is expressed in -- the loader only
# ever normalises translation scale, so an arbitrary-but-consistent world
# frame is fine, the same way it accepts Hypersim's OpenGL one.
_NED_TO_CV_COLS = [1, 2, 0]


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """(3,3) rotation matrix from a unit quaternion, world-from-local convention."""
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def ned_to_opencv_extrinsic(pose_row: np.ndarray) -> np.ndarray:
    """(3,4) world-to-camera, OpenCV convention, from one `poses.npy` row.

    `pose_row` is `tx ty tz qx qy qz qw`: position and camera-to-world
    orientation in NED axes (see module docstring). This permutes the
    rotation's columns to OpenCV axes, then inverts (transpose, since it is a
    rotation) to get camera-from-world, with `t = -R @ position` to match.
    """
    t, q = pose_row[:3], pose_row[3:]
    R_world_from_cam_ned = quat_to_matrix(*q)
    R_world_from_cam_cv = R_world_from_cam_ned[:, _NED_TO_CV_COLS]
    R_cam_from_world = R_world_from_cam_cv.T
    t_cam = -R_cam_from_world @ t
    return np.concatenate([R_cam_from_world, t_cam[:, None]], axis=1)


def fixed_intrinsics() -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = FOCAL
    K[1, 1] = FOCAL
    K[0, 2] = SRC_W / 2.0
    K[1, 2] = SRC_H / 2.0
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
    poses = np.load(src_dir / "poses.npy")
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
        png_path = src_dir / f"{frame_id}_left.png"
        depth_path = src_dir / f"{frame_id}_left_depth.npy"
        if not (png_path.exists() and depth_path.exists()):
            continue

        bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        depth_z = np.load(depth_path).astype(np.float32)
        assert depth_z.shape == (SRC_H, SRC_W), f"{scene}: depth shape {depth_z.shape}"
        assert bgr is not None and bgr.shape[:2] == (SRC_H, SRC_W), f"{scene}: image shape mismatch"

        good = np.isfinite(depth_z) & (depth_z > 0) & (depth_z < DEPTH_MAX)
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

        cam_from_world = ned_to_opencv_extrinsic(poses[i])
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
                "source": "tartanair_planar_z",
            }
        )
    )

    np.savez(
        meta_path,
        frame_names=np.array(frame_names),
        extrinsics=extrinsics.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        image_hw=np.array([out_h, out_w], dtype=np.int32),
        # Empty by design: an AirSim render has no triangulated points and
        # never needs the loader's sparse fallback. See the module docstring.
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
    except Exception as exc:  # one malformed trajectory shouldn't kill the run
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
    """Find `<env>/<difficulty>/<traj>/frame_ids.json`, one entry per staged trajectory."""
    found = {}
    for path in sorted(src_root.glob("*/*/*/frame_ids.json")):
        traj_dir = path.parent
        scene = f"{traj_dir.parent.parent.name}_{traj_dir.parent.name}_{traj_dir.name}"
        found.setdefault(scene, traj_dir)
    return sorted(found.items())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", type=Path, required=True, help="dir containing <env>/<difficulty>/<traj>/...")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth-out", type=Path, default=None, help="default: <out>/depth")
    p.add_argument("--scenes", nargs="*", default=None, help="process only these <env>_<difficulty>_<traj> ids")

    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--target-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="force an exact stored shape, overriding --resolution. Use 224 384 to make "
                        "TartanAir interchangeable with a DL3DV set built at --resolution 384")
    p.add_argument("--fit", choices=("crop", "squash"), default="crop",
                   help="how --target-hw is reached: 'crop' keeps natural proportions and loses "
                        "field of view; 'squash' keeps the full view but stretches the image")
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--jpeg-quality", type=int, default=95)

    p.add_argument("--max-frames", type=int, default=100, help="cap frames per trajectory")
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--min-depth-frac", type=float, default=0.2)
    p.add_argument("--covis-grid", type=int, nargs=2, default=(24, 32))
    p.add_argument("--covis-rel-tol", type=float, default=0.1)

    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of *environments* held out, not trajectories -- TartanAir has no "
                        "official split and consecutive frames within an env are highly correlated")
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


def assign_splits(scene_ids: list[str], val_frac: float, seed: int) -> dict[str, str]:
    """Whole environments held out, never individual trajectories (see
    `--val-frac`'s help)."""
    import hashlib

    envs = sorted({scene.rsplit("_", 2)[0] for scene in scene_ids})
    env_split = {}
    for env in envs:
        digest = hashlib.blake2b(f"{seed}:{env}".encode(), digest_size=8).digest()
        env_split[env] = "val" if int.from_bytes(digest, "big") / 2.0**64 < val_frac else "train"
    return {scene: env_split[scene.rsplit("_", 2)[0]] for scene in scene_ids}


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
        "splits": f"whole_env_holdout({args.val_frac})",
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

    Mirrors `preprocess_hypersim.py`'s `rebuild_index`: the streaming driver
    runs many single-trajectory workers in parallel, so the index is derived
    from the per-scene artefacts afterwards rather than read-modify-written by
    N workers.
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
        found = sorted(d.name for d in (args.out / "scenes" / SUBSET).glob("*") if (d / "meta.npz").exists())
        if not found:
            print(f"no processed scenes under {args.out}", file=sys.stderr)
            return 1
        splits = assign_splits(found, args.val_frac, args.seed)
        index = rebuild_index(args.out, splits, run_config(args, depth_out))
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
        print(f"no <env>/<difficulty>/<traj>/frame_ids.json under {args.src_root}", file=sys.stderr)
        return 1

    splits = assign_splits([s for s, _ in scans], args.val_frac, args.seed)

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
