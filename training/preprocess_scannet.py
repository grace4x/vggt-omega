#!/usr/bin/env python3
"""Turn raw ScanNet v2 `.sens` scans into the same training set format as DL3DV.

Input is one file per scan, as `download-scannet.py --type .sens` leaves it:

    <sens-root>/scans/<scene>/<scene>.sens

Output is byte-for-byte the contract `preprocess_dl3dv.py` writes, so
`DL3DVDataset` loads ScanNet with no changes at all:

    <out>/index.json
    <out>/scenes/scannet/<scene>/meta.npz
    <out>/scenes/scannet/<scene>/images/<stem>.jpg
    <out>/depth/scannet/<scene>/<stem>.png        -- pass as --depth-root
    <out>/depth/scannet/<scene>/meta.json

How this maps onto the DL3DV pipeline, since the two datasets get their geometry
from completely different places:

* DL3DV needs COLMAP because it is raw video with no poses. ScanNet ships
  BundleFusion poses and factory intrinsics inside the `.sens`, so there is no
  reconstruction step, no undistortion (the frames carry no distortion model)
  and no sparse-point stage.
* DL3DV's *sparse* depth branch is COLMAP's triangulated points at ~1% pixel
  coverage; its *dense* branch is DA3 monocular depth scale-aligned to that.
  ScanNet's depth is neither -- it is the metric depth sensor, ~85% coverage,
  and it needs no alignment because it is already in the same metres as the
  poses. So every ScanNet scene takes the loader's dense path, and the sparse
  arrays (`points_xyz`, `obs_frame`, `obs_point`) are written empty. The loader
  already guards on `obs_f.size`, so empty flows straight through to the dense
  branch that overwrites them anyway.
* `covisibility` is the one thing COLMAP was providing that has no ScanNet
  equivalent -- DL3DV computes it as IoU of shared point tracks. Here it is
  computed geometrically instead: reproject a grid of frame i's depth into
  frame j and count what lands in-frustum *and* depth-consistent. That is a
  stricter test than track IoU (it sees occlusion), and it keeps the loader's
  covisibility frame sampler working identically.
* Scale: the loader normalises every scene by its own mean point distance, so
  ScanNet's metric metres get divided out exactly like DL3DV's arbitrary COLMAP
  units. Both datasets therefore arrive at the model in the same unit space,
  which is what makes mixing them coherent -- at the cost of ScanNet's metric
  scale, which the model was never able to observe anyway.

Resolution: ScanNet colour is 1296x968 (97% of scans) or 640x480 (the rest).
Both land on 288x384 at `--resolution 384`, so the whole dataset is one shape
and `DL3DVDataset._select_shape` drops nothing.

Depth is 640x480 for every scan, i.e. a *different* camera sampling than the
1296x968 colour, so it is remapped through the stored pinhole rather than
merely resized -- see `depth_remap`.

Example:

    python training/preprocess_scannet.py \
        --sens-root ~/scannet/scans --out ~/scannet-train \
        --resolution 384 --workers 8

Usually you do not run this directly: `download_scannet.sh` calls it per scan so
each `.sens` can be deleted right after it is consumed.
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
from preprocess_dl3dv import centered_pinhole, target_shape  # noqa: E402
from sens_reader import open_sens  # noqa: E402

SUBSET = "scannet"

# `load_depth` treats a quantised value of 0 as "no ground truth", so the bottom
# rail has to sit strictly below anything the sensor can report or real near
# surfaces would be silently discarded. The Structure sensor's minimum range is
# ~0.4 m; 5 cm is comfortably under it.
DEPTH_QUANT_LO = 0.05


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def depth_remap(K_depth: np.ndarray, K_out: np.ndarray, out_hw, depth_hw):
    """Maps for `cv2.remap`: output pinhole pixel -> source depth pixel.

    ScanNet's depth and colour are the same physical rig (`extrinsic_color` and
    `extrinsic_depth` are both identity in the release, which `process_scene`
    asserts) but they are sampled on different grids -- 640x480 depth against
    1296x968 colour, with intrinsics to match. So going from the stored pinhole
    to a depth pixel is a pure intrinsic change of variables: undo `K_out` to a
    normalised ray, reapply `K_depth`. No pose is involved.

    Sampled with INTER_NEAREST by the caller: bilinear across a depth
    discontinuity averages a foreground and a background hit into a surface that
    exists in neither, and the 0 that marks "no return" would bleed into its
    neighbours as a fractional depth.
    """
    out_h, out_w = out_hw
    u = np.arange(out_w, dtype=np.float32)[None, :]
    v = np.arange(out_h, dtype=np.float32)[:, None]
    x = (u - K_out[0, 2]) / K_out[0, 0]
    y = (v - K_out[1, 2]) / K_out[1, 1]
    map_x = np.broadcast_to(x, (out_h, out_w)) * K_depth[0, 0] + K_depth[0, 2]
    map_y = np.broadcast_to(y, (out_h, out_w)) * K_depth[1, 1] + K_depth[1, 2]
    return (
        np.ascontiguousarray(map_x, dtype=np.float32),
        np.ascontiguousarray(map_y, dtype=np.float32),
    )


def plan_output(
    K_color: np.ndarray,
    src_hw: tuple[int, int],
    *,
    resolution: int,
    mode: str,
    patch_size: int,
    target_hw: tuple[int, int] | None,
    fit: str,
) -> tuple[int, int, np.ndarray, tuple]:
    """Decide the stored shape and the pinhole to render it through.

    Three ways to size the output:

    * `target_hw=None` (default) -- aspect-preserving `--resolution` sizing, the
      same `target_shape` DL3DV uses. ScanNet's two source shapes (1296x968 and
      640x480) both converge on one stored shape at every resolution, so the
      dataset stays a single shape either way.
    * `target_hw` with `fit="crop"` -- scale to *cover* the target, then take a
      centred crop. Focal lengths stay isotropic and the image keeps its natural
      proportions; the cost is field of view, since the excess rows or columns
      are discarded.
    * `target_hw` with `fit="squash"` -- one anisotropic resize onto the target.
      Keeps the entire field of view. Still an exact pinhole, because fx and fy
      scale independently and the 9D pose encoding carries fov_h and fov_w
      separately -- but the stored frames no longer have natural proportions,
      which is a distribution shift for a backbone pretrained on real photos.

    Returns `(out_h, out_w, K_out, plan)`; `plan` tells `process_scene` how to
    resample the colour. Depth needs nothing extra -- `depth_remap` derives its
    maps from `K_out`, which already encodes the scale and the crop.
    """
    src_h, src_w = src_hw
    if target_hw is None:
        out_h, out_w = target_shape(src_h, src_w, resolution, mode, patch_size)
        return out_h, out_w, centered_pinhole(K_color, out_h, out_w, src_h, src_w), ("resize", out_h, out_w, 0, 0)

    out_h, out_w = int(target_hw[0]), int(target_hw[1])
    if fit == "squash":
        return out_h, out_w, centered_pinhole(K_color, out_h, out_w, src_h, src_w), ("resize", out_h, out_w, 0, 0)
    if fit != "crop":
        raise ValueError(f"unknown fit {fit!r}")

    scale = max(out_w / src_w, out_h / src_h)
    inter_w = max(out_w, int(math.ceil(src_w * scale)))
    inter_h = max(out_h, int(math.ceil(src_h * scale)))
    sx, sy = inter_w / src_w, inter_h / src_h
    # Place the crop so the true principal point lands as close to the centre as
    # an integer offset allows, because K_out pins it to the centre exactly (the
    # pose encoding decodes cx=W/2, cy=H/2). ScanNet's pp is already within a
    # quarter pixel of centre, so the residual is far below one pixel.
    x0 = min(max(int(round(K_color[0, 2] * sx - out_w / 2.0)), 0), inter_w - out_w)
    y0 = min(max(int(round(K_color[1, 2] * sy - out_h / 2.0)), 0), inter_h - out_h)

    K_out = np.eye(3, dtype=np.float64)
    K_out[0, 0] = K_color[0, 0] * sx
    K_out[1, 1] = K_color[1, 1] * sy
    K_out[0, 2] = out_w / 2.0
    K_out[1, 2] = out_h / 2.0
    return out_h, out_w, K_out, ("crop", inter_h, inter_w, y0, x0)


def geometric_covisibility(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    grid: tuple[int, int] = (24, 32),
    rel_tol: float = 0.1,
) -> np.ndarray:
    """(N, N) float16 overlap, standing in for DL3DV's shared-track IoU.

    `covis[i, j]` is the fraction of frame i's valid depth samples that reproject
    into frame j's frustum *and* agree with frame j's own depth there to within
    `rel_tol`. The depth-agreement test is what makes this better than counting
    frustum hits: a point on the far side of a wall reprojects inside the image
    but disagrees with the wall's depth, so it is correctly not covisible.

    Symmetrised with the elementwise max -- the more generous of the two
    directions, mirroring DL3DV's `inter / min(|i|, |j|)`.
    """
    n, out_h, out_w = depth.shape
    gy = np.linspace(0, out_h - 1, grid[0]).round().astype(np.int64)
    gx = np.linspace(0, out_w - 1, grid[1]).round().astype(np.int64)
    vv, uu = np.meshgrid(gy.astype(np.float64), gx.astype(np.float64), indexing="ij")
    vv, uu = vv.reshape(-1), uu.reshape(-1)
    sampled = depth[:, gy][:, :, gx].reshape(n, -1)  # (N, M), same flattening

    R = extrinsics[:, :, :3]
    t = extrinsics[:, :, 3]
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    # Frame i's samples in the world frame (extrinsics are world->cam).
    cam = np.stack(
        [(uu - cx[:, None]) / fx[:, None] * sampled, (vv - cy[:, None]) / fy[:, None] * sampled, sampled],
        axis=-1,
    )  # (N, M, 3)
    world = np.einsum("nji,nmj->nmi", R, cam - t[:, None, :])  # R^T (cam - t)

    covis = np.eye(n, dtype=np.float32)
    rows = np.arange(n)
    for i in range(n):
        keep = sampled[i] > 0
        total = int(keep.sum())
        if total == 0:
            continue
        pts = world[i][keep]  # (Mi, 3)
        proj = np.einsum("nij,mj->nmi", R, pts) + t[:, None, :]  # (N, Mi, 3)
        z = proj[..., 2]
        safe = np.where(z > 1e-6, z, 1.0)
        u = proj[..., 0] / safe * fx[:, None] + cx[:, None]
        v = proj[..., 1] / safe * fy[:, None] + cy[:, None]
        ok = (z > 1e-6) & (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
        ui = np.clip(u, 0, out_w - 1).astype(np.int64)
        vi = np.clip(v, 0, out_h - 1).astype(np.int64)
        target = depth[rows[:, None], vi, ui]
        ok &= (target > 0) & (np.abs(z - target) <= rel_tol * safe)
        covis[i] = ok.sum(axis=1) / total

    covis = np.maximum(covis, covis.T)
    np.fill_diagonal(covis, 1.0)
    return covis.astype(np.float16)


def quantise_depth(stack: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Log-quantise metres to uint16, matching `fetch_dl3dv_depth.dequantise`.

    Unlike the DL3DV version this pins `lo` to a constant below the sensor's
    range instead of a low percentile, because ScanNet's invalid pixels are exact
    zeros rather than a noisy near-zero tail: they must land on the q==0 rail
    that `load_depth` reads as "no ground truth", and no *valid* pixel may. `hi`
    is still a high percentile with headroom, so only the far junk tail clips to
    the 65535 rail that `load_depth` also discards.

    16 bits over a log range of ~0.05-10 m is ~0.008% relative error, three
    orders of magnitude below the sensor's own noise.
    """
    valid = stack > 0
    lo = DEPTH_QUANT_LO
    hi = float(np.percentile(stack[valid], 99.99)) * 1.02 if valid.any() else lo * 1000.0
    hi = max(hi, lo * 1.001)
    span = math.log(hi) - math.log(lo)
    q = (np.log(np.clip(stack, lo, hi)) - math.log(lo)) / span
    out = (q * 65535.0).round().astype(np.uint16)
    out[~valid] = 0  # the "no ground truth" rail
    return out, lo, hi


# --------------------------------------------------------------------------- #
# per-scene worker
# --------------------------------------------------------------------------- #


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

    with open_sens(task["sens_path"]) as reader:
        header = reader.header

        # The remap below assumes colour and depth share one camera frame. That
        # holds for every scan in the release, but a scene where it does not
        # would produce plausible-looking, silently misaligned depth -- so refuse
        # it rather than guess at a rig transform we cannot verify.
        for label, extrinsic in (("color", header.extrinsic_color), ("depth", header.extrinsic_depth)):
            if not np.allclose(extrinsic, np.eye(4), atol=1e-5):
                return {"status": "skip", "reason": f"nonidentity_extrinsic_{label}", "scene": scene}

        valid = reader.valid_poses
        indices = np.flatnonzero(valid)
        if indices.size < task["min_frames"]:
            return {
                "status": "skip",
                "reason": "too_few_valid_poses",
                "scene": scene,
                "detail": f"{indices.size} of {header.num_frames}",
            }

        # ScanNet scans are 30 fps video averaging ~1600 frames, so take an even
        # stride across the whole trajectory: it keeps the full spatial extent
        # (which is what the covisibility sampler draws wide baselines from)
        # while capping the per-scene cost.
        stride = max(task["frame_stride"], math.ceil(indices.size / task["max_frames"]))
        indices = indices[::stride]

        src_h, src_w = header.color_hw
        K_color = header.intrinsic_color[:3, :3]
        K_depth = header.intrinsic_depth[:3, :3]
        # ScanNet intrinsics are already OpenCV pixel-centre (the standard
        # unprojection indexes integer pixels), so unlike COLMAP there is no
        # half-pixel shift to undo before rescaling.
        out_h, out_w, K_out, plan = plan_output(
            K_color,
            (src_h, src_w),
            resolution=task["resolution"],
            mode=task["mode"],
            patch_size=task["patch_size"],
            target_hw=task["target_hw"],
            fit=task["fit"],
        )
        map_x, map_y = depth_remap(K_depth, K_out, (out_h, out_w), header.depth_hw)
        _, inter_h, inter_w, crop_y, crop_x = plan

        images_out = out_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        frame_names, extrinsics, depths, depth_stats = [], [], [], []
        for i in indices:
            i = int(i)
            depth_m = reader.read_depth_mm(i).astype(np.float32) / header.depth_shift
            depth_m = cv2.remap(depth_m, map_x, map_y, cv2.INTER_NEAREST, borderValue=0.0)
            good = depth_m > 0
            if good.mean() < task["min_depth_frac"]:
                continue

            bgr = reader.read_color(i)
            if (bgr.shape[0], bgr.shape[1]) != (inter_h, inter_w):
                # INTER_AREA averages the source pixels that fall in each output
                # pixel, which matters here: colour is downsampled 3.4x from
                # 1296x968, and INTER_LINEAR would alias badly at that ratio.
                interp = cv2.INTER_AREA if inter_h < bgr.shape[0] else cv2.INTER_CUBIC
                bgr = cv2.resize(bgr, (inter_w, inter_h), interpolation=interp)
            if (inter_h, inter_w) != (out_h, out_w):
                bgr = bgr[crop_y : crop_y + out_h, crop_x : crop_x + out_w]
            if task["min_sharpness"] > 0:
                grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(grey, cv2.CV_64F).var() < task["min_sharpness"]:
                    continue

            stem = f"frame_{i:06d}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, task["jpeg_quality"]])

            # `poses` is camera-to-world; the contract stores world-to-camera.
            world_from_cam = reader.poses[i]
            cam_from_world = np.linalg.inv(world_from_cam)[:3]

            frame_names.append(f"{stem}.jpg")
            extrinsics.append(cam_from_world)
            depths.append(depth_m)
            depth_stats.append(np.percentile(depth_m[good], [5, 50, 95]))

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

    quantised, lo, hi = quantise_depth(depth_stack)
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
                "source": "scannet_sensor",
            }
        )
    )

    np.savez(
        meta_path,
        frame_names=np.array(frame_names),
        extrinsics=extrinsics.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        image_hw=np.array([out_h, out_w], dtype=np.int32),
        # Empty by design: ScanNet has no triangulated points and never needs
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


def _safe_process(task: dict) -> dict:
    try:
        return process_scene(task)
    except Exception as exc:  # a truncated .sens shouldn't kill the run
        return {
            "status": "error",
            "scene": task["scene"],
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def discover_scans(sens_root: Path) -> list[tuple[str, Path]]:
    """Find `<scene>/<scene>.sens` under `sens_root`, tolerating a flat layout."""
    found = {}
    for path in sorted(sens_root.rglob("*.sens")):
        found.setdefault(path.stem, path)
    return sorted(found.items())


def load_split(splits_dir: Path | None) -> dict[str, str]:
    """ScanNet's official benchmark split, if the txt files are on disk.

    Preferred over a random holdout so val numbers are comparable with published
    ScanNet results and no two scans of the same *space* straddle the split --
    scene0011_00 and scene0011_01 are the same room, so splitting them randomly
    leaks.
    """
    if splits_dir is None:
        return {}
    assignment = {}
    for split, name in (("train", "scannetv2_train.txt"), ("val", "scannetv2_val.txt")):
        path = splits_dir / name
        if path.exists():
            for line in path.read_text().split():
                assignment[line.strip()] = split
    return assignment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sens-root", type=Path, required=True, help="dir containing <scene>/<scene>.sens")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth-out", type=Path, default=None, help="default: <out>/depth")
    p.add_argument("--scenes", nargs="*", default=None, help="process only these scene ids")

    p.add_argument("--resolution", type=int, default=384, help="match your DL3DV set (384 -> 288x384)")
    p.add_argument("--target-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="force an exact stored shape, overriding --resolution. Use 224 384 to make "
                        "ScanNet interchangeable with a DL3DV set built at --resolution 384")
    p.add_argument("--fit", choices=("crop", "squash"), default="crop",
                   help="how --target-hw is reached: 'crop' keeps natural proportions and loses "
                        "field of view; 'squash' keeps the full view but stretches the image")
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--jpeg-quality", type=int, default=95)

    p.add_argument("--frame-stride", type=int, default=1, help="floor on the stride; --max-frames raises it")
    p.add_argument("--max-frames", type=int, default=150, help="cap frames per scene (scans average ~1600)")
    p.add_argument("--min-frames", type=int, default=24)
    p.add_argument("--min-depth-frac", type=float, default=0.2, help="drop frames with less valid depth")
    p.add_argument("--min-sharpness", type=float, default=0.0,
                   help="Laplacian-variance floor; >0 drops motion-blurred frames (try 60)")
    p.add_argument("--covis-grid", type=int, nargs=2, default=(24, 32))
    p.add_argument("--covis-rel-tol", type=float, default=0.1)

    p.add_argument("--splits-dir", type=Path, default=None,
                   help="dir with scannetv2_{train,val}.txt; falls back to --val-frac")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true", help="one line per scene, for the streaming driver")
    p.add_argument("--no-index", action="store_true",
                   help="skip the index write; for parallel single-scan workers")
    p.add_argument("--index-only", action="store_true",
                   help="rebuild index.json from the meta.npz already on disk, then exit")
    p.add_argument("--dry-run", action="store_true")
    return p


def merge_index(out_root: Path, results: list[dict], config: dict) -> dict:
    """Rewrite `index.json`, folding in whatever a previous run already recorded.

    The streaming driver calls this process once per scan, so the index has to
    accumulate rather than be rebuilt from the current batch alone.
    """
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

    The streaming driver runs many single-scan workers in parallel, and a
    read-modify-write of one shared index from N processes loses entries no
    matter how atomic the write is: a worker that read the file before its
    neighbour wrote will overwrite that neighbour. So workers pass `--no-index`
    and write nothing shared, and the index is derived from the per-scene
    artefacts afterwards, which is idempotent and order-independent.
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


def assign_splits(scene_ids, splits_dir: Path | None, val_frac: float, seed: int) -> dict[str, str]:
    official = load_split(splits_dir)
    if official:
        return {s: official.get(s, "train") for s in scene_ids}
    rng = np.random.default_rng(seed)
    # Bucket by *space* id, not scan id: scene0011_00 and scene0011_01 are the
    # same room, so splitting them independently leaks val geometry into train.
    spaces = sorted({s.rsplit("_", 1)[0] for s in scene_ids})
    held = {sp for sp, r in zip(spaces, rng.random(len(spaces))) if r < val_frac}
    return {s: ("val" if s.rsplit("_", 1)[0] in held else "train") for s in scene_ids}


def run_config(args, depth_out: Path, splits_dir_used: bool) -> dict:
    return {
        "sens_root": str(args.sens_root),
        "depth_root": str(depth_out),
        "resolution": args.resolution,
        "target_hw": list(args.target_hw) if args.target_hw else None,
        "fit": args.fit,
        "mode": args.mode,
        "patch_size": args.patch_size,
        "max_frames": args.max_frames,
        "min_depth_frac": args.min_depth_frac,
        "min_sharpness": args.min_sharpness,
        "splits": "official" if splits_dir_used else f"random({args.val_frac})",
    }


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


def main() -> int:
    args = build_parser().parse_args()
    depth_out = args.depth_out or (args.out / "depth")

    if args.index_only:
        # The .sens files are long gone by now, so take the scene universe from
        # the output tree rather than from --sens-root.
        found = sorted(d.name for d in (args.out / "scenes" / SUBSET).glob("*") if (d / "meta.npz").exists())
        if not found:
            print(f"no processed scenes under {args.out}", file=sys.stderr)
            return 1
        splits = assign_splits(found, args.splits_dir, args.val_frac, args.seed)
        index = rebuild_index(args.out, splits, run_config(args, depth_out, bool(load_split(args.splits_dir))))
        consolidate_failures(args.out)
        total = sum(e["num_frames"] or 0 for e in index["scenes"])
        print(
            f"wrote {args.out / 'index.json'}: {len(index['scenes'])} scenes "
            f"({index['num_train']} train / {index['num_val']} val), {total} frames"
        )
        return 0

    scans = discover_scans(args.sens_root)
    if args.scenes:
        wanted = set(args.scenes)
        scans = [(s, p) for s, p in scans if s in wanted]
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        print(f"no .sens files under {args.sens_root}", file=sys.stderr)
        return 1

    assignment = load_split(args.splits_dir)
    splits = assign_splits([s for s, _ in scans], args.splits_dir, args.val_frac, args.seed)

    if args.dry_run:
        for scene, path in scans[:20]:
            print(f"  [{splits[scene]}] {scene}  {path}  ({path.stat().st_size / 1e9:.2f} GB)")
        print(f"  ... {len(scans)} scans, {sum(v == 'val' for v in splits.values())} val")
        return 0

    tasks = [
        {
            "scene": scene,
            "sens_path": str(path),
            "out_dir": str(args.out / "scenes" / SUBSET / scene),
            "depth_dir": str(depth_out / SUBSET / scene),
            "resolution": args.resolution,
            "target_hw": tuple(args.target_hw) if args.target_hw else None,
            "fit": args.fit,
            "mode": args.mode,
            "patch_size": args.patch_size,
            "jpeg_quality": args.jpeg_quality,
            "frame_stride": max(1, args.frame_stride),
            "max_frames": max(1, args.max_frames),
            "min_frames": args.min_frames,
            "min_depth_frac": args.min_depth_frac,
            "min_sharpness": args.min_sharpness,
            "covis_grid": list(args.covis_grid),
            "covis_rel_tol": args.covis_rel_tol,
            "overwrite": args.overwrite,
        }
        for scene, path in scans
    ]

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
                    # Always true: ScanNet's depth is the sensor's, so every scene
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
            record(_safe_process(task))
            if not args.quiet:
                print(f"\r[{i + 1}/{len(tasks)}] {counts}", end="", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_safe_process, task) for task in tasks]
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

    # Under --no-index every shared file is off limits, because N single-scan
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
