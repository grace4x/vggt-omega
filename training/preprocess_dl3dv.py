#!/usr/bin/env python3
"""Turn raw DL3DV downloads into a compact training set for VGGT-Omega.

Inputs (two separate HuggingFace repos, matched by scene hash):

    <images-root>/{1K,2K,3K}/<hash>/images_4/frame_XXXXX.png
    <images-root>/{1K,2K,3K}/<hash>/transforms.json
    <colmap-root>/{1K,2K,3K}/<hash>/colmap/sparse/0/{cameras,images,points3D}.bin

The COLMAP models were reconstructed on the 3840x2160 originals with an OPENCV
(distorted) camera, while `images_4` holds the 4x downsampled frames -- and only
a subset of them. This script reconciles the two: it intersects the frame lists,
rescales the intrinsics, undistorts to a true pinhole camera with the principal
point pinned to the image centre (which is what VGGT-Omega's 9D FoV pose
encoding assumes), resizes, and writes per-scene metadata.

Output:

    <out>/index.json
    <out>/scenes/<subset>/<hash>/meta.npz
    <out>/scenes/<subset>/<hash>/images/frame_XXXXX.jpg

`meta.npz` contents (N frames, M points, K observations):

    frame_names      (N,)      str   -- basename of the jpg under images/
    extrinsics       (N,3,4)   f32   -- camera-from-world, OpenCV convention
    intrinsics       (N,3,3)   f32   -- pinhole, for the stored image resolution
    image_hw         (2,)      i32   -- (H, W) of the stored jpgs
    points_xyz       (M,3)     f32   -- triangulated points, world frame
    points_error     (M,)      f32   -- COLMAP reprojection error (full-res px)
    obs_frame        (K,)      i32   -- index into frame_names
    obs_point        (K,)      i32   -- index into points_xyz
    depth_stats      (N,3)     f32   -- per-frame sparse depth (p05, median, p95)
    covisibility     (N,N)     f16   -- IoU of shared point tracks
    scene_scale      scalar    f32   -- median sparse depth over the scene
    subset, scene_id           str

Depth supervision is sparse: rebuild it at load time by projecting `points_xyz`
with the stored extrinsics/intrinsics (see `dl3dv_dataset.py`). Nothing here is
scale-normalised -- `scene_scale` is provided so the training code can pick its
own convention.

Example:

    python training/preprocess_dl3dv.py \
        --images-root ~/dl3dv-images-960P \
        --colmap-root ~/dl3dv-cache-960P \
        --out ~/dl3dv-train \
        --resolution 512 --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras_bin, read_images_bin, read_points3D_bin  # noqa: E402


SUBSETS = ("1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")


# --------------------------------------------------------------------------- #
# scene discovery
# --------------------------------------------------------------------------- #


def sparse_dir_for(colmap_root: Path, subset: str, scene: str) -> Path | None:
    """DL3DV ships `colmap/sparse/0`, but tolerate a flat `sparse/0` too."""
    for candidate in (
        colmap_root / subset / scene / "colmap" / "sparse" / "0",
        colmap_root / subset / scene / "sparse" / "0",
        colmap_root / subset / scene / "colmap" / "sparse",
    ):
        if all((candidate / f"{n}.bin").exists() for n in ("cameras", "images", "points3D")):
            return candidate
    return None


def image_dir_for(images_root: Path, subset: str, scene: str, prefer: str) -> Path | None:
    scene_dir = images_root / subset / scene
    for name in (prefer, "images_4", "images_8", "images_2", "images"):
        candidate = scene_dir / name
        if candidate.is_dir() and any(p.suffix in IMAGE_EXTS for p in candidate.iterdir()):
            return candidate
    return None


def discover_scenes(images_root: Path, colmap_root: Path, subsets, image_dir_name: str):
    """Yield (subset, scene_hash, image_dir, sparse_dir) for scenes that have both halves.

    The download is expected to still be running, so anything incomplete is
    silently skipped rather than treated as an error.
    """
    found, skipped = [], {"no_images": 0, "no_colmap": 0}
    for subset in subsets:
        if not (images_root / subset).is_dir():
            continue
        for scene in sorted(os.listdir(images_root / subset)):
            if scene.startswith("."):
                continue
            image_dir = image_dir_for(images_root, subset, scene, image_dir_name)
            if image_dir is None:
                skipped["no_images"] += 1
                continue
            sparse_dir = sparse_dir_for(colmap_root, subset, scene)
            if sparse_dir is None:
                skipped["no_colmap"] += 1
                continue
            found.append((subset, scene, image_dir, sparse_dir))
    return found, skipped


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def target_shape(height: int, width: int, resolution: int, mode: str, patch_size: int) -> tuple[int, int]:
    """Mirror `vggt_omega.utils.load_fn` sizing so training matches inference.

    `resolution <= 0` keeps the native size (rounded down to a patch multiple).
    """
    if resolution <= 0:
        return (height // patch_size) * patch_size, (width // patch_size) * patch_size

    aspect_ratio = height / max(width, 1)
    if mode == "balanced":
        token_number = (resolution // patch_size) ** 2
        w_patches = np.sqrt(token_number / aspect_ratio)
        h_patches = max(1, int(round(token_number / w_patches)))
        w_patches = max(1, int(round(w_patches)))
        return h_patches * patch_size, w_patches * patch_size

    if mode == "max_size":
        if aspect_ratio >= 1.0:
            out_h = resolution
            out_w = max(patch_size, int(round(resolution / aspect_ratio / patch_size)) * patch_size)
        else:
            out_w = resolution
            out_h = max(patch_size, int(round(resolution * aspect_ratio / patch_size)) * patch_size)
        return out_h, out_w

    raise ValueError(f"unknown resize mode {mode!r}")


def colmap_to_opencv_K(K: np.ndarray) -> np.ndarray:
    """COLMAP puts the coordinate origin at the *corner* of the top-left pixel;
    OpenCV (and every resampler we use downstream) puts it at the pixel centre."""
    out = K.copy()
    out[0, 2] -= 0.5
    out[1, 2] -= 0.5
    return out


def centered_pinhole(K: np.ndarray, out_h: int, out_w: int, src_h: int, src_w: int) -> np.ndarray:
    """Rescale K to the output size and pin the principal point to the centre.

    VGGT-Omega's pose encoding only carries (fov_h, fov_w) and `encoding_to_camera`
    decodes cx=W/2, cy=H/2. Rendering the training images through exactly that
    camera makes the model's built-in assumption true by construction, instead of
    leaving a half-pixel bias for it to absorb. DL3DV's COLMAP principal points
    are already dead centre, so the only thing this really does is the shift.
    """
    sx, sy = out_w / src_w, out_h / src_h
    out = np.eye(3, dtype=np.float64)
    out[0, 0] = K[0, 0] * sx
    out[1, 1] = K[1, 1] * sy
    out[0, 2] = out_w / 2.0
    out[1, 2] = out_h / 2.0
    return out


def covisibility_matrix(track_sets: list[set]) -> np.ndarray:
    n = len(track_sets)
    out = np.eye(n, dtype=np.float32)
    for i in range(n):
        si = track_sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = track_sets[j]
            if not sj:
                continue
            inter = len(si & sj)
            if inter:
                out[i, j] = out[j, i] = inter / min(len(si), len(sj))
    return out.astype(np.float16)


# --------------------------------------------------------------------------- #
# per-scene worker
# --------------------------------------------------------------------------- #


def process_scene(task: dict) -> dict:
    subset = task["subset"]
    scene = task["scene"]
    out_dir = Path(task["out_dir"])
    meta_path = out_dir / "meta.npz"

    if meta_path.exists() and not task["overwrite"]:
        with np.load(meta_path, allow_pickle=True) as data:
            return {"status": "cached", "subset": subset, "scene": scene, "num_frames": int(len(data["frame_names"]))}

    cameras = read_cameras_bin(Path(task["sparse_dir"]) / "cameras.bin")
    colmap_images = read_images_bin(Path(task["sparse_dir"]) / "images.bin")
    points = read_points3D_bin(Path(task["sparse_dir"]) / "points3D.bin")

    if len(points.ids) < task["min_points"]:
        return {"status": "skip", "reason": "too_few_points", "subset": subset, "scene": scene}

    image_dir = Path(task["image_dir"])
    on_disk = {p.name: p for p in image_dir.iterdir() if p.suffix in IMAGE_EXTS}
    # COLMAP names may carry a directory prefix; match on the basename.
    matched = []
    for img in colmap_images.values():
        base = Path(img.name).name
        path = on_disk.get(base) or on_disk.get(Path(base).stem + ".png") or on_disk.get(Path(base).stem + ".jpg")
        if path is not None:
            matched.append((base, img, path))
    matched.sort(key=lambda item: item[0])

    if task["frame_stride"] > 1:
        matched = matched[:: task["frame_stride"]]
    if len(matched) < task["min_frames"]:
        return {
            "status": "skip",
            "reason": "too_few_frames",
            "subset": subset,
            "scene": scene,
            "detail": f"{len(matched)} matched of {len(colmap_images)} colmap / {len(on_disk)} on disk",
        }

    # Point filtering: drop badly triangulated points and short tracks up front so
    # the observation arrays stay small.
    keep_point = (points.error <= task["max_point_error"]) & (points.track_length >= task["min_track_length"])
    if keep_point.sum() < task["min_points"]:
        keep_point = points.track_length >= 2  # fall back rather than lose the scene
    point_ids = points.ids[keep_point]
    points_xyz = points.xyz[keep_point]
    points_err = points.error[keep_point]
    pid_to_idx = {int(pid): i for i, pid in enumerate(point_ids)}

    probe = PILImage.open(matched[0][2])
    src_w, src_h = probe.size
    probe.close()
    out_h, out_w = target_shape(src_h, src_w, task["resolution"], task["mode"], task["patch_size"])

    images_out = out_dir / "images"
    if not task["skip_images"]:
        images_out.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    frame_names, extrinsics, intrinsics = [], [], []
    obs_frame, obs_point = [], []
    depth_stats, track_sets = [], []
    undistort_cache: dict[tuple, tuple] = {}

    for frame_idx, (base, img, src_path) in enumerate(matched):
        camera = cameras[img.camera_id]
        cache_key = (camera.id, src_w, src_h)
        if cache_key not in undistort_cache:
            K_full, dist = camera.to_pinhole()
            # COLMAP intrinsics are for the full-resolution originals; images_4 is
            # a plain downsample, so scale K and keep the (scale-free) distortion.
            scale_x, scale_y = src_w / camera.width, src_h / camera.height
            K_src = colmap_to_opencv_K(K_full)
            K_src[0, 0] *= scale_x
            K_src[1, 1] *= scale_y
            K_src[0, 2] = (K_src[0, 2] + 0.5) * scale_x - 0.5
            K_src[1, 2] = (K_src[1, 2] + 0.5) * scale_y - 0.5
            K_out = centered_pinhole(K_src, out_h, out_w, src_h, src_w)

            needs_undistort = bool(np.abs(dist).max() > 1e-8)
            if needs_undistort:
                # Map straight from distorted source pixels to the final resized
                # pinhole image: one resample instead of undistort-then-resize.
                map_x, map_y = cv2.initUndistortRectifyMap(
                    K_src, dist, None, K_out, (out_w, out_h), cv2.CV_32FC1
                )
            else:
                map_x = map_y = None
            undistort_cache[cache_key] = (K_src, dist, K_out, map_x, map_y)

        K_src, dist, K_out, map_x, map_y = undistort_cache[cache_key]

        cam_from_world = img.cam_from_world()
        R, t = cam_from_world[:, :3], cam_from_world[:, 3]

        # Observations: use the triangulated points directly rather than the raw
        # keypoints, so the 2D locations land in the undistorted frame exactly.
        valid = img.point3D_ids >= 0
        local_ids = [pid_to_idx.get(int(p)) for p in img.point3D_ids[valid]]
        local_ids = np.array([i for i in local_ids if i is not None], dtype=np.int64)

        depths = np.zeros(0, dtype=np.float32)
        if local_ids.size:
            cam_xyz = points_xyz[local_ids] @ R.T + t
            z = cam_xyz[:, 2]
            in_front = z > task["min_depth"]
            uv = np.full((local_ids.size, 2), -1.0)
            uv[in_front, 0] = cam_xyz[in_front, 0] / z[in_front] * K_out[0, 0] + K_out[0, 2]
            uv[in_front, 1] = cam_xyz[in_front, 1] / z[in_front] * K_out[1, 1] + K_out[1, 2]
            inside = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < out_w) & (uv[:, 1] >= 0) & (uv[:, 1] < out_h)
            local_ids = local_ids[inside]
            depths = z[inside].astype(np.float32)

        if local_ids.size < task["min_obs_per_frame"]:
            continue

        if local_ids.size > task["max_obs_per_frame"]:
            # Prefer the best-triangulated observations when trimming.
            order = np.argsort(points_err[local_ids])[: task["max_obs_per_frame"]]
            local_ids, depths = local_ids[order], depths[order]

        if not task["skip_images"]:
            dst_path = images_out / (Path(base).stem + ".jpg")
            if task["overwrite"] or not dst_path.exists():
                bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if map_x is not None:
                    bgr = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR)
                elif (bgr.shape[0], bgr.shape[1]) != (out_h, out_w):
                    interp = cv2.INTER_AREA if out_h < bgr.shape[0] else cv2.INTER_CUBIC
                    bgr = cv2.resize(bgr, (out_w, out_h), interpolation=interp)
                cv2.imwrite(str(dst_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, task["jpeg_quality"]])

        idx = len(frame_names)
        frame_names.append(Path(base).stem + ".jpg")
        extrinsics.append(cam_from_world)
        intrinsics.append(K_out)
        obs_frame.append(np.full(local_ids.size, idx, dtype=np.int32))
        obs_point.append(local_ids.astype(np.int32))
        depth_stats.append(np.percentile(depths, [5, 50, 95]))
        track_sets.append(set(local_ids.tolist()))

    if len(frame_names) < task["min_frames"]:
        return {"status": "skip", "reason": "too_few_valid_frames", "subset": subset, "scene": scene}

    obs_frame = np.concatenate(obs_frame)
    obs_point = np.concatenate(obs_point)
    depth_stats = np.stack(depth_stats).astype(np.float32)

    # Keep only points that survived into at least one observation.
    used, remap = np.unique(obs_point, return_inverse=True)
    points_xyz = points_xyz[used]
    points_err = points_err[used]
    obs_point = remap.astype(np.int32)

    scene_scale = float(np.median(depth_stats[:, 1]))
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        return {"status": "skip", "reason": "bad_scale", "subset": subset, "scene": scene}

    np.savez(
        meta_path,
        frame_names=np.array(frame_names),
        extrinsics=np.stack(extrinsics).astype(np.float32),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        image_hw=np.array([out_h, out_w], dtype=np.int32),
        points_xyz=points_xyz.astype(np.float32),
        points_error=points_err.astype(np.float32),
        obs_frame=obs_frame,
        obs_point=obs_point,
        depth_stats=depth_stats,
        covisibility=covisibility_matrix(track_sets),
        scene_scale=np.float32(scene_scale),
        subset=np.array(subset),
        scene_id=np.array(scene),
    )

    return {
        "status": "ok",
        "subset": subset,
        "scene": scene,
        "num_frames": len(frame_names),
        "num_points": int(points_xyz.shape[0]),
        "num_obs": int(obs_frame.size),
        "image_hw": [out_h, out_w],
        "scene_scale": scene_scale,
    }


def _safe_process(task: dict) -> dict:
    try:
        return process_scene(task)
    except Exception as exc:  # a half-downloaded scene shouldn't kill the run
        return {
            "status": "error",
            "subset": task["subset"],
            "scene": task["scene"],
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-root", type=Path, default=Path.home() / "dl3dv-images-960P")
    p.add_argument("--colmap-root", type=Path, default=Path.home() / "dl3dv-cache-960P")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subsets", nargs="*", default=None, help=f"default: any of {SUBSETS} present on disk")
    p.add_argument("--image-dir-name", default="images_4", help="preferred subdirectory of frames")

    p.add_argument("--resolution", type=int, default=512, help="0 keeps the native resolution")
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--skip-images", action="store_true", help="write meta.npz only")

    p.add_argument("--frame-stride", type=int, default=1, help="subsample frames within a scene")
    p.add_argument("--min-frames", type=int, default=24)
    p.add_argument("--min-points", type=int, default=512)
    p.add_argument("--min-obs-per-frame", type=int, default=64)
    p.add_argument("--max-obs-per-frame", type=int, default=4096)
    p.add_argument("--max-point-error", type=float, default=2.0, help="COLMAP reproj. error, full-res px")
    p.add_argument("--min-track-length", type=int, default=3)
    p.add_argument("--min-depth", type=float, default=1e-3)

    p.add_argument("--val-frac", type=float, default=0.02, help="fraction of scenes held out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="process at most N scenes (0 = all)")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list what would be processed and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    subsets = args.subsets or [s for s in SUBSETS if (args.images_root / s).is_dir()]

    scenes, skipped = discover_scenes(args.images_root, args.colmap_root, subsets, args.image_dir_name)
    print(
        f"found {len(scenes)} complete scenes in {subsets} "
        f"(skipped {skipped['no_colmap']} without colmap, {skipped['no_images']} without images)"
    )
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        print("nothing to do", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    is_val = rng.random(len(scenes)) < args.val_frac

    if args.dry_run:
        for (subset, scene, image_dir, sparse_dir), val in list(zip(scenes, is_val))[:20]:
            print(f"  [{'val' if val else 'train'}] {subset}/{scene[:12]}...  {image_dir}  {sparse_dir}")
        print(f"  ... {len(scenes)} scenes total, {int(is_val.sum())} val")
        return 0

    out_root = args.out
    (out_root / "scenes").mkdir(parents=True, exist_ok=True)

    tasks = []
    for (subset, scene, image_dir, sparse_dir), val in zip(scenes, is_val):
        tasks.append(
            {
                "subset": subset,
                "scene": scene,
                "image_dir": str(image_dir),
                "sparse_dir": str(sparse_dir),
                "out_dir": str(out_root / "scenes" / subset / scene),
                "split": "val" if val else "train",
                "resolution": args.resolution,
                "mode": args.mode,
                "patch_size": args.patch_size,
                "jpeg_quality": args.jpeg_quality,
                "skip_images": args.skip_images,
                "frame_stride": max(1, args.frame_stride),
                "min_frames": args.min_frames,
                "min_points": args.min_points,
                "min_obs_per_frame": args.min_obs_per_frame,
                "max_obs_per_frame": args.max_obs_per_frame,
                "max_point_error": args.max_point_error,
                "min_track_length": args.min_track_length,
                "min_depth": args.min_depth,
                "overwrite": args.overwrite,
            }
        )

    entries, failures = [], []
    counts = {"ok": 0, "cached": 0, "skip": 0, "error": 0}
    started = time.time()

    def record(result: dict, task: dict) -> None:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        if result["status"] in ("ok", "cached"):
            entries.append(
                {
                    "subset": task["subset"],
                    "scene": task["scene"],
                    "split": task["split"],
                    "path": f"scenes/{task['subset']}/{task['scene']}",
                    "num_frames": result.get("num_frames"),
                    "num_points": result.get("num_points"),
                    "scene_scale": result.get("scene_scale"),
                }
            )
        else:
            failures.append({k: result.get(k) for k in ("subset", "scene", "status", "reason", "detail")})

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            record(_safe_process(task), task)
            print(f"\r[{i + 1}/{len(tasks)}] {counts}", end="", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_safe_process, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures)):
                record(future.result(), futures[future])
                elapsed = time.time() - started
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(tasks) - i - 1) / max(rate, 1e-6)
                print(
                    f"\r[{i + 1}/{len(tasks)}] {counts} {rate:.2f} scenes/s eta {eta / 60:.1f}m   ",
                    end="",
                    flush=True,
                )
    print()

    entries.sort(key=lambda e: (e["subset"], e["scene"]))
    index = {
        "config": {
            "images_root": str(args.images_root),
            "colmap_root": str(args.colmap_root),
            "subsets": subsets,
            "resolution": args.resolution,
            "mode": args.mode,
            "patch_size": args.patch_size,
            "frame_stride": args.frame_stride,
            "max_point_error": args.max_point_error,
            "min_track_length": args.min_track_length,
            "images_written": not args.skip_images,
        },
        "counts": counts,
        "num_train": sum(e["split"] == "train" for e in entries),
        "num_val": sum(e["split"] == "val" for e in entries),
        "scenes": entries,
    }
    (out_root / "index.json").write_text(json.dumps(index, indent=1))
    if failures:
        (out_root / "failures.json").write_text(json.dumps(failures, indent=1))

    total_frames = sum(e["num_frames"] or 0 for e in entries)
    print(
        f"wrote {out_root / 'index.json'}: {len(entries)} scenes "
        f"({index['num_train']} train / {index['num_val']} val), {total_frames} frames, "
        f"{counts['skip']} skipped, {counts['error']} errors in {(time.time() - started) / 60:.1f}m"
    )
    if failures:
        print(f"see {out_root / 'failures.json'} for the {len(failures)} scenes that were dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
