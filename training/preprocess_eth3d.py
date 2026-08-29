#!/usr/bin/env python3
"""Turn the ETH3D high-res multi-view set into the same eval set format as DL3DV.

Input is the two per-scene archives from https://www.eth3d.net/datasets, extracted
side by side under one root:

    <raw-root>/<scene>/dslr_calibration_jpg/{cameras,images,points3D}.txt
    <raw-root>/<scene>/images/dslr_images/<name>.JPG
    <raw-root>/<scene>/ground_truth_depth/dslr_images/<name>.JPG

Output is the contract `preprocess_dl3dv.py` writes, so `DL3DVDataset` and
`training/evaluate.py` load ETH3D with no changes at all:

    <out>/index.json
    <out>/scenes/eth3d/<scene>/meta.npz
    <out>/scenes/eth3d/<scene>/images/<stem>.jpg
    <out>/depth/eth3d/<scene>/<stem>.png        -- pass as --depth-root
    <out>/depth/eth3d/<scene>/meta.json

Why the *distorted* archive and not `_dslr_undistorted`. ETH3D's ground truth
depth maps "match the original (distorted) versions of the images, not the
pre-undistorted ones", so using the shipped undistorted JPGs would need the
distorted calibration anyway (it is the only thing that maps an undistorted ray
back to a depth pixel) -- and that lives in `_dslr_jpg.7z`. Reading the distorted
pair instead means one archive fewer per scene, and image and depth go through
exactly the same resampling map, so they cannot drift apart.

Three things about the data that this file exists to handle:

* **Camera model.** The distorted cameras are COLMAP `THIN_PRISM_FISHEYE`, which
  `colmap_io.Camera.to_pinhole()` cannot express and `cv2.undistort` cannot
  either. It does not need inverting though: `undistort_map` walks *output*
  pinhole pixels, turns each into a ray, and pushes the ray through the forward
  model to find its source pixel -- which is exactly the direction `cv2.remap`
  consumes. `--check` reports the model's reprojection error against
  `points3D.txt`; it should land near ETH3D's own ~0.6 px.

* **Depth is Z, in the poses' own units, with `inf` for "no ground truth".** Raw
  little-endian float32, row-major, at the distorted image's resolution. Verified
  against the sparse tracks: depth / z_cam has a median of 0.9999, against 0.903
  for depth / ||xyz||, so it is depth along the optical axis and not ray length.

* **It is sparse at full resolution** -- ~9% of pixels on `pipes`, because the
  laser scan is nowhere near 24 MP dense. A single `INTER_NEAREST` sample per
  output pixel would keep it at 9% and throw away the other 91% of the ground
  truth. So the map is built at `--supersample` times the output resolution and
  each output pixel takes the *median* of the valid samples that land in it,
  which turns ~9% at 6048x4032 into ~90%+ at 256x384. The median rather than the
  min because a block straddling a depth edge should report whichever surface
  fills most of it, not always the nearer one.

Resolution: every DSLR scene is 6048x4032, so `--resolution 384` gives one shape,
256x384, for the whole set and `DL3DVDataset._select_shape` drops nothing. That is
*not* the 224x384 a DL3DV set built at `--resolution 384` is stored at. It does not
have to be -- an eval run loads one dataset -- but to force the match:

    --target-hw 224 384 --fit crop

Example:

    python training/preprocess_eth3d.py \
        --raw-root ~/eth3d-raw --out ~/eth3d-eval \
        --resolution 384 --workers 4

Usually you do not run this directly: `download_eth3d.sh` calls it per scene so
each archive can be deleted right after it is consumed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_dl3dv import centered_pinhole, colmap_to_opencv_K, target_shape  # noqa: E402
from preprocess_scannet import geometric_covisibility, plan_output  # noqa: E402
from training.colmap_io import Camera, qvec_to_rotmat  # noqa: E402

SUBSET = "eth3d"

# The 13 scenes of the high-res multi-view *training* split -- the ones with
# ground truth. The test split has no scans and no depth, so it is not scoreable
# offline and is deliberately not listed here.
SCENES = (
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
)


# --------------------------------------------------------------------------- #
# COLMAP text model
# --------------------------------------------------------------------------- #

# (name, num_params) by model name, for the text format where the model is spelled
# out. Only the two ETH3D actually uses are needed, but an unknown name should say
# so rather than silently mis-slice `params`.
TEXT_CAMERA_PARAMS = {"PINHOLE": 4, "SIMPLE_PINHOLE": 3, "THIN_PRISM_FISHEYE": 12}


def read_cameras_txt(path: Path) -> dict[int, Camera]:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        model = fields[1]
        expected = TEXT_CAMERA_PARAMS.get(model)
        params = np.array([float(x) for x in fields[4:]], dtype=np.float64)
        if expected is not None and params.size != expected:
            raise ValueError(f"{path}: {model} wants {expected} params, got {params.size}")
        cameras[int(fields[0])] = Camera(int(fields[0]), model, int(fields[2]), int(fields[3]), params)
    return cameras


def read_images_txt(path: Path) -> list[dict]:
    """One dict per image, in the file's order, with the 2D observations kept.

    The keypoints are needed twice over: as the sparse depth fallback the loader
    projects when no `--depth-root` is given, and as the `--check` self-test for
    the distortion model.
    """
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    out = []
    for i in range(0, len(rows) - 1, 2):
        head = rows[i].split()
        xys = np.array([float(x) for x in rows[i + 1].split()], dtype=np.float64).reshape(-1, 3)
        out.append(
            {
                "id": int(head[0]),
                "qvec": np.array([float(x) for x in head[1:5]], dtype=np.float64),
                "tvec": np.array([float(x) for x in head[5:8]], dtype=np.float64),
                "camera_id": int(head[8]),
                "name": head[9],
                "xys": xys[:, :2],
                "point3D_ids": xys[:, 2].astype(np.int64),
            }
        )
    return out


def read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """(xyz (m, 3), error (m,), point3D_id -> row) for the scene's tracked points."""
    ids, xyz, error = [], [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        ids.append(int(f[0]))
        xyz.append((float(f[1]), float(f[2]), float(f[3])))
        error.append(float(f[7]))
    return (
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(error, dtype=np.float64),
        {pid: i for i, pid in enumerate(ids)},
    )


# --------------------------------------------------------------------------- #
# THIN_PRISM_FISHEYE
# --------------------------------------------------------------------------- #


def project_thin_prism(camera: Camera, xy: np.ndarray) -> np.ndarray:
    """Normalised rays (..., 2) -> pixels (..., 2), in COLMAP's corner-origin frame.

    A transcription of COLMAP's `ThinPrismFisheyeCameraModel::WorldToImage`: the
    equidistant fisheye step first (theta = atan(r), radius rescaled to theta),
    then the thin-prism radial-tangential polynomial on the *fisheye* coordinates,
    not on the pinhole ones. Params after (fx, fy, cx, cy) are ordered
    (k1, k2, p1, p2, k3, k4, sx1, sy1) -- note p1/p2 sit between k2 and k3, which
    is not the OpenCV layout, and that swapping them costs ~0.1 px of reprojection
    error on ETH3D, i.e. enough to notice and not enough to catch by eye.

    `PINHOLE` is accepted too so the undistorted archive could be fed in as well;
    it is the same code path with every distortion coefficient zero.
    """
    p = camera.params
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = p
        return np.stack([fx * xy[..., 0] + cx, fy * xy[..., 1] + cy], axis=-1)
    if camera.model == "SIMPLE_PINHOLE":
        return np.stack([p[0] * xy[..., 0] + p[1], p[0] * xy[..., 1] + p[2]], axis=-1)
    if camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError(f"unsupported ETH3D camera model {camera.model!r}")

    fx, fy, cx, cy = p[:4]
    k1, k2, p1, p2, k3, k4, sx1, sy1 = p[4:]

    x, y = xy[..., 0], xy[..., 1]
    r = np.hypot(x, y)
    # theta/r as r -> 0 is 1, and the ratio is what the model actually uses, so
    # take the limit rather than dividing by a clamped zero.
    scale = np.where(r > 1e-12, np.arctan(r) / np.where(r > 1e-12, r, 1.0), 1.0)
    u, v = x * scale, y * scale

    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4
    du = u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2
    dv = v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2
    return np.stack([fx * (u + du) + cx, fy * (v + dv) + cy], axis=-1)


def check_camera(camera: Camera, images: list[dict], xyz: np.ndarray, lookup: dict[int, int]) -> float:
    """Median reprojection error in source pixels, for the distortion self-test."""
    errors = []
    for image in images:
        keep = image["point3D_ids"] > 0
        rows = [lookup.get(int(pid), -1) for pid in image["point3D_ids"][keep]]
        keep_rows = np.array([r for r in rows if r >= 0], dtype=np.int64)
        if keep_rows.size == 0:
            continue
        observed = image["xys"][keep][np.array([r >= 0 for r in rows])]
        cam = xyz[keep_rows] @ qvec_to_rotmat(image["qvec"]).T + image["tvec"]
        ahead = cam[:, 2] > 1e-6
        if not ahead.any():
            continue
        projected = project_thin_prism(camera, cam[ahead, :2] / cam[ahead, 2:3])
        errors.append(np.linalg.norm(projected - observed[ahead], axis=1))
    return float(np.median(np.concatenate(errors))) if errors else float("nan")


def undistort_map(camera: Camera, K_out: np.ndarray, out_hw: tuple[int, int]):
    """Maps for `cv2.remap`: output pinhole pixel -> source distorted pixel.

    Both ends are in OpenCV's pixel-centre convention; the half-pixel hop into and
    out of COLMAP's corner-origin convention is done here so callers never see it.
    Returns `(map_x, map_y, oob_fraction)`; `oob_fraction` is how much of the
    output grid falls outside the source image, which is the cost of rendering the
    barrel-corrected view onto the source's own canvas instead of the larger one
    ETH3D's undistorted JPGs use. `cv2.remap` fills those with the border value.
    """
    out_h, out_w = out_hw
    u = (np.arange(out_w, dtype=np.float64) - K_out[0, 2]) / K_out[0, 0]
    v = (np.arange(out_h, dtype=np.float64) - K_out[1, 2]) / K_out[1, 1]
    rays = np.stack(np.broadcast_arrays(u[None, :], v[:, None]), axis=-1)

    pixels = project_thin_prism(camera, rays) - 0.5  # COLMAP corner -> OpenCV centre
    inside = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] <= camera.width - 1)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] <= camera.height - 1)
    )
    return (
        np.ascontiguousarray(pixels[..., 0], dtype=np.float32),
        np.ascontiguousarray(pixels[..., 1], dtype=np.float32),
        float(1.0 - inside.mean()),
    )


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #


def read_eth3d_depth(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """ETH3D's raw float32 dump as (H, W) metres, with 0 where there is no ground truth.

    Invalid pixels are stored as `inf`; negatives and NaN should not occur but are
    folded into the same 0 so nothing downstream has to test for them again.
    """
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != hw[0] * hw[1]:
        raise ValueError(f"{path}: {raw.size} floats, expected {hw[0]}x{hw[1]}={hw[0] * hw[1]}")
    depth = raw.reshape(hw)
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)


def pool_valid(supersampled: np.ndarray, out_h: int, out_w: int, ss: int, min_samples: int) -> np.ndarray:
    """(out_h*ss, out_w*ss) depth -> (out_h, out_w), median of the valid samples.

    This is the step that recovers ETH3D's ground truth from its sparsity: at
    `ss=8` each output pixel pools 64 source samples, so a ~9%-dense source comes
    out ~90%+ dense. Blocks with fewer than `min_samples` valid entries stay 0.
    """
    blocks = supersampled.reshape(out_h, ss, out_w, ss).transpose(0, 2, 1, 3).reshape(out_h, out_w, ss * ss)
    valid = blocks > 0
    counts = valid.sum(axis=-1)
    pooled = np.zeros((out_h, out_w), dtype=np.float32)
    enough = counts >= min_samples
    if enough.any():
        # nanmedian over the whole block array allocates a second copy of it; do
        # the masking in place on a float32 view to keep that at ~25 MB.
        masked = np.where(valid, blocks, np.nan)
        with warnings.catch_warnings():
            # An all-invalid block is the normal case for sky, not a problem; its
            # nan is discarded by the `enough` mask on the next line anyway.
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(masked, axis=-1)
        pooled[enough] = median[enough].astype(np.float32)
    return pooled


def quantise_depth(stack: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Log-quantise to uint16, matching `fetch_dl3dv_depth.dequantise`.

    `lo` and `hi` bracket the *observed* range with 2% of headroom rather than
    sitting at percentiles, because ETH3D's depth is a registered laser scan and
    has no junk tail to clip away -- every valid value is worth keeping. The two
    rails stay reserved: `load_depth` reads q==0 and q==65535 as "no ground truth",
    so no real measurement may land on either.
    """
    valid = stack > 0
    if not valid.any():
        return np.zeros_like(stack, dtype=np.uint16), 0.1, 100.0
    lo = max(float(stack[valid].min()) * 0.98, 1e-4)
    hi = max(float(stack[valid].max()) * 1.02, lo * 1.001)
    span = math.log(hi) - math.log(lo)
    q = (np.log(np.clip(stack, lo, hi)) - math.log(lo)) / span
    out = (q * 65535.0).round().astype(np.uint16)
    out[~valid] = 0
    return out, lo, hi


# --------------------------------------------------------------------------- #
# per-scene worker
# --------------------------------------------------------------------------- #


def process_scene(task: dict) -> dict:
    scene = task["scene"]
    raw_dir = Path(task["raw_dir"])
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

    cal_dir = raw_dir / "dslr_calibration_jpg"
    image_root = raw_dir / "images"
    depth_root = raw_dir / "ground_truth_depth"
    for required in (cal_dir / "cameras.txt", cal_dir / "images.txt", cal_dir / "points3D.txt"):
        if not required.exists():
            return {"status": "skip", "reason": "missing_calibration", "scene": scene, "detail": str(required)}
    if not depth_root.exists() and not task["allow_no_depth"]:
        return {"status": "skip", "reason": "missing_depth", "scene": scene, "detail": str(depth_root)}

    cameras = read_cameras_txt(cal_dir / "cameras.txt")
    images = read_images_txt(cal_dir / "images.txt")
    xyz_world, point_error, point_lookup = read_points3d_txt(cal_dir / "points3D.txt")
    # Sort by filename: ETH3D names frames by shutter order (DSC_0634, DSC_0635,
    # ...), so this makes `--sampling contiguous` mean "a real sub-trajectory" and
    # makes the whole run reproducible, which the file order does not.
    images.sort(key=lambda im: im["name"])

    # One camera per scene in every ETH3D release, but the format allows more and a
    # mixed-resolution scene cannot be stored at a single shape. Keep the majority.
    shapes = {}
    for image in images:
        cam = cameras[image["camera_id"]]
        shapes.setdefault((cam.height, cam.width), []).append(image)
    src_hw = max(shapes, key=lambda hw: len(shapes[hw]))
    dropped_shape = len(images) - len(shapes[src_hw])
    images = shapes[src_hw]
    camera = cameras[images[0]["camera_id"]]

    # `plan_output` wants a pinhole to rescale. The fisheye's own (fx, fy) is the
    # right one to hand it: ETH3D's official undistorted cameras keep exactly these
    # focal lengths and merely widen the canvas to fit the corrected corners, so
    # rendering onto the source canvas instead reproduces their view minus a few
    # percent of the edges -- and gains a guarantee that no output pixel is blank.
    K_src = colmap_to_opencv_K(
        np.array([[camera.params[0], 0.0, camera.params[2]], [0.0, camera.params[1], camera.params[3]], [0, 0, 1]])
    )
    out_h, out_w, K_out, _ = plan_output(
        K_src,
        src_hw,
        resolution=task["resolution"],
        mode=task["mode"],
        patch_size=task["patch_size"],
        target_hw=task["target_hw"],
        fit=task["fit"],
    )

    ss = int(task["supersample"])
    K_ss = K_out.copy()
    K_ss[0, 0] *= ss
    K_ss[1, 1] *= ss
    # Output pixel u maps to supersampled pixel (u + 0.5) * ss - 0.5; carrying that
    # through the principal point is what keeps the ss grid concentric with the
    # output grid rather than half a block off.
    K_ss[0, 2] = (K_out[0, 2] + 0.5) * ss - 0.5
    K_ss[1, 2] = (K_out[1, 2] + 0.5) * ss - 0.5
    map_x, map_y, oob = undistort_map(camera, K_ss, (out_h * ss, out_w * ss))

    # The ss grid is still a downsample of the 24 MP source (about 2x at ss=8), and
    # `remap` point-samples, so pre-filter to that scale or the frames alias. The
    # distortion's Jacobian is within a few percent of identity here, which is what
    # makes one global sigma a good enough stand-in for a per-pixel one.
    sigma = 0.5 * max(src_hw[1] / (out_w * ss), src_hw[0] / (out_h * ss))
    sigma = sigma if sigma > 0.6 else 0.0

    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    frame_names, extrinsics, kept, depths, depth_stats = [], [], [], [], []
    for image in images:
        stem = Path(image["name"]).stem
        # images.txt names are relative to images/, i.e. "dslr_images/DSC_0634.JPG".
        src_image = image_root / image["name"]
        if not src_image.exists():
            src_image = image_root / Path(image["name"]).name
        if not src_image.exists():
            continue

        depth_path = depth_root / image["name"]
        if not depth_path.exists():
            depth_path = depth_root / "dslr_images" / Path(image["name"]).name
        if not depth_path.exists():
            if not task["allow_no_depth"]:
                continue
            pooled = np.zeros((out_h, out_w), dtype=np.float32)
        else:
            source = read_eth3d_depth(depth_path, src_hw)
            fine = cv2.remap(source, map_x, map_y, cv2.INTER_NEAREST, borderValue=0.0)
            pooled = pool_valid(fine, out_h, out_w, ss, task["min_samples"])

        coverage = float((pooled > 0).mean())
        if coverage < task["min_depth_frac"]:
            continue

        bgr = cv2.imread(str(src_image), cv2.IMREAD_COLOR)
        if bgr is None or (bgr.shape[0], bgr.shape[1]) != src_hw:
            continue
        if sigma > 0:
            bgr = cv2.GaussianBlur(bgr, (0, 0), sigma)
        fine_bgr = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        bgr = cv2.resize(fine_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(images_out / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, task["jpeg_quality"]])

        frame_names.append(f"{stem}.jpg")
        extrinsics.append(np.concatenate([qvec_to_rotmat(image["qvec"]), image["tvec"][:, None]], axis=1))
        kept.append(image)
        depths.append(pooled)
        good = pooled > 0
        depth_stats.append(np.percentile(pooled[good], [5, 50, 95]) if good.any() else np.zeros(3))

    if len(frame_names) < task["min_frames"]:
        return {
            "status": "skip",
            "reason": "too_few_usable_frames",
            "scene": scene,
            "detail": f"{len(frame_names)} kept of {len(images)}",
        }

    extrinsics = np.stack(extrinsics)
    intrinsics = np.repeat(K_out[None], len(frame_names), axis=0)
    depth_stack = np.stack(depths)

    # Sparse tracks, in the same layout the loader's no-`--depth-root` path wants.
    # ETH3D genuinely has these (unlike ScanNet), and keeping them means the set is
    # still usable, at ~0.2% coverage, without the depth archives.
    obs_frame, obs_point = [], []
    for f, image in enumerate(kept):
        for pid in image["point3D_ids"]:
            row = point_lookup.get(int(pid), -1) if pid > 0 else -1
            if row >= 0:
                obs_frame.append(f)
                obs_point.append(row)

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
                "source": "eth3d_laser_scan",
            }
        )
    )

    np.savez(
        meta_path,
        frame_names=np.array(frame_names),
        extrinsics=extrinsics.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        image_hw=np.array([out_h, out_w], dtype=np.int32),
        points_xyz=xyz_world.astype(np.float32),
        points_error=point_error.astype(np.float32),
        obs_frame=np.array(obs_frame, dtype=np.int32),
        obs_point=np.array(obs_point, dtype=np.int32),
        depth_stats=np.stack(depth_stats).astype(np.float32),
        covisibility=covisibility,
        subset=np.array(SUBSET),
        scene_id=np.array(scene),
    )

    result = {
        "status": "ok",
        "scene": scene,
        "num_frames": len(frame_names),
        "num_points": int(xyz_world.shape[0]),
        "image_hw": [out_h, out_w],
        "src_hw": list(src_hw),
        "camera_model": camera.model,
        "depth_valid": float((depth_stack > 0).mean()),
        "median_depth": float(np.median(depth_stats, axis=0)[1]),
        "oob_frac": oob,
        "dropped_shape": dropped_shape,
    }
    if task["check"]:
        result["reproj_px"] = check_camera(camera, kept, xyz_world, point_lookup)
    return result


def _safe_process(task: dict) -> dict:
    try:
        return process_scene(task)
    except Exception as exc:
        return {
            "status": "error",
            "scene": task["scene"],
            "reason": type(exc).__name__,
            "detail": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #


def scene_entry(scene: str, num_frames: int, image_hw: list[int], num_points: int, split: str) -> dict:
    return {
        "subset": SUBSET,
        "scene": scene,
        "split": split,
        "path": f"scenes/{SUBSET}/{scene}",
        "num_frames": num_frames,
        "num_points": num_points,
        "image_hw": image_hw,
        "has_depth": True,
    }


def rebuild_index(out_root: Path, config: dict, split: str) -> dict:
    """Assemble `index.json` from the `meta.npz` files already on disk.

    Derived rather than accumulated, for the same reason as ScanNet's: the shell
    driver runs one process per scene, and a read-modify-write of a shared index
    from N processes loses whichever entry lost the race. Deriving it afterwards is
    idempotent and order-independent.
    """
    entries = []
    for meta_path in sorted((out_root / "scenes" / SUBSET).glob("*/meta.npz")):
        try:
            with np.load(meta_path, allow_pickle=True) as data:
                entries.append(
                    scene_entry(
                        meta_path.parent.name,
                        int(len(data["frame_names"])),
                        [int(v) for v in data["image_hw"]],
                        int(data["points_xyz"].shape[0]),
                        split,
                    )
                )
        except Exception:
            continue  # a worker killed mid-write; the next run redoes it
    index = {
        "config": config,
        "num_train": sum(e["split"] == "train" for e in entries),
        "num_val": sum(e["split"] == "val" for e in entries),
        "scenes": entries,
    }
    tmp = (out_root / "index.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=1))
    tmp.replace(out_root / "index.json")  # atomic, so a reader never sees half a file
    return index


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, required=True, help="dir holding the extracted <scene>/ trees")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth-out", type=Path, default=None, help="default <out>/depth")
    p.add_argument("--scenes", nargs="*", default=None, help="default: every scene found under --raw-root")
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--resolution", type=int, default=384, help="0 keeps the native resolution")
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--target-hw", type=int, nargs=2, default=None,
                   help="force a stored shape, e.g. 224 384 to match a DL3DV set")
    p.add_argument("--fit", choices=("crop", "squash"), default="crop", help="how --target-hw is reached")
    p.add_argument("--jpeg-quality", type=int, default=95)

    p.add_argument("--supersample", type=int, default=8,
                   help="depth samples per output pixel per axis; 1 disables pooling")
    p.add_argument("--min-samples", type=int, default=1, help="valid samples an output depth pixel needs")
    p.add_argument("--min-depth-frac", type=float, default=0.05, help="drop frames with less depth coverage")
    p.add_argument("--min-frames", type=int, default=8, help="drop scenes left with fewer usable frames")
    p.add_argument("--allow-no-depth", action="store_true", help="keep frames with no ground truth depth file")

    p.add_argument("--covis-grid", type=int, nargs=2, default=(24, 32))
    p.add_argument("--covis-rel-tol", type=float, default=0.1)

    p.add_argument("--split", choices=("val", "train"), default="val",
                   help="what to record in index.json; ETH3D is a held-out set, so val")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check", action="store_true", help="also report the distortion model's reprojection error")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-index", action="store_true", help="write per-scene files only; for the shell driver")
    p.add_argument("--index-only", action="store_true", help="rebuild index.json from what is on disk, then exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    depth_out = args.depth_out or (args.out / "depth")
    config = {
        "dataset": "eth3d_high_res_multi_view_training",
        "resolution": args.resolution,
        "mode": args.mode,
        "patch_size": args.patch_size,
        "target_hw": list(args.target_hw) if args.target_hw else None,
        "fit": args.fit,
        "supersample": args.supersample,
        "min_samples": args.min_samples,
        "min_depth_frac": args.min_depth_frac,
        "depth_source": "eth3d_laser_scan",
    }

    if args.index_only:
        index = rebuild_index(args.out, config, args.split)
        total = sum(e["num_frames"] or 0 for e in index["scenes"])
        print(f"wrote {args.out / 'index.json'}: {len(index['scenes'])} scenes, {total} frames")
        return 0

    scenes = args.scenes
    if not scenes:
        found = sorted(d.name for d in args.raw_root.glob("*") if (d / "dslr_calibration_jpg").is_dir())
        scenes = [s for s in SCENES if s in found] + [s for s in found if s not in SCENES]
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        raise SystemExit(
            f"no scene under {args.raw_root} has a dslr_calibration_jpg/ directory.\n"
            "Extract <scene>_dslr_jpg.7z and <scene>_dslr_depth.7z there, or use "
            "training/download_eth3d.sh."
        )

    tasks = [
        {
            "scene": scene,
            "raw_dir": str(args.raw_root / scene),
            "out_dir": str(args.out / "scenes" / SUBSET / scene),
            "depth_dir": str(depth_out / SUBSET / scene),
            "resolution": args.resolution,
            "mode": args.mode,
            "patch_size": args.patch_size,
            "target_hw": tuple(args.target_hw) if args.target_hw else None,
            "fit": args.fit,
            "jpeg_quality": args.jpeg_quality,
            "supersample": args.supersample,
            "min_samples": args.min_samples,
            "min_depth_frac": args.min_depth_frac,
            "min_frames": args.min_frames,
            "allow_no_depth": args.allow_no_depth,
            "covis_grid": tuple(args.covis_grid),
            "covis_rel_tol": args.covis_rel_tol,
            "overwrite": args.overwrite,
            "check": args.check,
        }
        for scene in scenes
    ]

    if not args.quiet:
        shape = f"{args.target_hw[0]}x{args.target_hw[1]} ({args.fit})" if args.target_hw else f"--resolution {args.resolution}"
        print(f"ETH3D -> {args.out}   {len(tasks)} scene(s), {shape}, depth ss={args.supersample}")

    started = time.time()
    results = []
    if args.workers <= 1:
        for task in tasks:
            results.append(_safe_process(task))
            if not args.quiet:
                print(_line(results[-1]), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_safe_process, t): t["scene"] for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())
                if not args.quiet:
                    print(_line(results[-1]), flush=True)
    if args.quiet:  # the shell driver parses this
        for result in results:
            print(_line(result), flush=True)

    if not args.no_index:
        index = rebuild_index(args.out, config, args.split)
        frames = sum(e["num_frames"] or 0 for e in index["scenes"])
        ok = sum(r["status"] in ("ok", "cached") for r in results)
        print(
            f"\n{ok}/{len(results)} scenes in {time.time() - started:.0f}s. "
            f"wrote {args.out / 'index.json'}: {len(index['scenes'])} scenes, {frames} frames"
        )
        print(f"\nevaluate with:\n  --data-root {args.out} --depth-root {depth_out} --split all")
    return 0 if all(r["status"] != "error" for r in results) else 1


def _line(r: dict) -> str:
    bits = [r["scene"], r["status"]]
    if r["status"] in ("ok", "cached"):
        bits.append(f"frames={r['num_frames']}")
        if "image_hw" in r:
            bits.append(f"hw={r['image_hw'][0]}x{r['image_hw'][1]}")
        if "depth_valid" in r:
            bits.append(f"depth={r['depth_valid']:.1%}")
        if "median_depth" in r:
            bits.append(f"med={r['median_depth']:.2f}")
        if "oob_frac" in r and r["oob_frac"] > 0.001:
            bits.append(f"oob={r['oob_frac']:.2%}")
        if "reproj_px" in r:
            bits.append(f"reproj={r['reproj_px']:.2f}px")
    else:
        bits.append(r.get("reason", ""))
        bits.append(r.get("detail", ""))
    return " ".join(str(b) for b in bits if b != "")


if __name__ == "__main__":
    raise SystemExit(main())
