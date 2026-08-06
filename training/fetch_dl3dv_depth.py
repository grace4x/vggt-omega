#!/usr/bin/env python3
"""Fetch DA3-aligned dense depth for the set produced by `preprocess_dl3dv.py`.

Pulls one zip per scene from `KangLiao/DL3DV-Depth-DA3-Aligned`, warps every
depth map through the same undistort+resize the RGB went through, and stores it
as 16-bit log-quantised PNG:

    <depth-root>/<subset>/<hash>/frame_XXXXX.png
    <depth-root>/<subset>/<hash>/meta.json    -- lo/hi for dequant

`--data-root` is only read (for `index.json` and each scene's `meta.npz`); every
byte written goes under `--depth-root`, including the transient zips.

The upstream zips carry depth for *every* video frame, while DL3DV's 960P image
release only ships the odd-numbered ones -- so most scenes get ~2x more depth
maps than they have images. We keep them all: COLMAP has poses for those frames
too, so they become usable the moment the matching RGB is downloaded.

Depth is already in the same scale as the scene's COLMAP reconstruction
(median(da3 / sparse_depth) measured at 0.992-1.000 across frames), so
`scene_scale` in `meta.npz` stays valid and nothing needs rescaling.

Sizes: source maps are ~2.07 MB each (e.g. 539x961 float32). Resized to the
stored 224x384 and quantised they cost ~40 MB per scene, ~180 GB for all 4382
covered scenes. A worker holds one zip (~540 MB) plus one scene's stack
(~110 MB), so peak disk tracks --workers rather than the ~2.4 TB downloaded.

Roughly 12% of scenes have no upstream depth; those are recorded as
`has_depth: false` in `index.json` so the loader can skip them.

Usage:

    python training/fetch_dl3dv_depth.py --depth-root ~/dl3dv-depth --workers 6
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras_bin  # noqa: E402
from preprocess_dl3dv import colmap_to_opencv_K, sparse_dir_for  # noqa: E402

REPO = "KangLiao/DL3DV-Depth-DA3-Aligned"


# --------------------------------------------------------------------------- #
# quantisation
# --------------------------------------------------------------------------- #


def quantise(stack: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Log-quantise to uint16. 16 bits over a log range is ~0.003% median error,
    far below DA3's own accuracy, and log spacing keeps near surfaces precise.

    `hi` is the 99.99th percentile rather than the max: DA3 emits occasional
    degenerate sky pixels (5.7e6 in one frame I checked) that would otherwise
    eat the whole range -- and which would also overflow float16 storage.
    """
    lo = max(float(np.percentile(stack, 0.1)), 1e-3)
    hi = max(float(np.percentile(stack, 99.99)), lo * 1.001)
    clipped = float(((stack < lo) | (stack > hi)).mean())
    q = (np.log(np.clip(stack, lo, hi)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (q * 65535.0).round().astype(np.uint16), lo, hi, clipped


def dequantise(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = math.log(hi) - math.log(lo)
    return np.exp(math.log(lo) + q.astype(np.float32) / 65535.0 * span)


def load_depth(depth_dir: Path, stems) -> np.ndarray:
    """Read depth for `stems` (frame names without extension) as (S, H, W) float32.

    Zero means "no ground truth here", matching `dl3dv_dataset`. Frames with no
    stored depth are all-zero, and so are pixels that hit either end of the
    quantisation range: the ceiling is DA3's degenerate sky, and the floor is its
    near-zero output (0.18% of pixels in one scene I measured), both of which are
    junk that would otherwise dominate a log-depth loss.
    """
    with open(depth_dir / "meta.json") as f:
        meta = json.load(f)
    out_h, out_w = meta["out_hw"]
    out = np.zeros((len(stems), out_h, out_w), dtype=np.float32)
    for i, stem in enumerate(stems):
        path = depth_dir / f"{stem}.png"
        if path.exists():
            q = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            plane = dequantise(q, meta["lo"], meta["hi"])
            plane[(q == 0) | (q == 65535)] = 0.0
            out[i] = plane
    return out


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def pinhole_map(camera, K_out: np.ndarray, out_hw, depth_hw):
    """Map output pinhole pixels -> source depth-map pixels, for `cv2.remap`.

    Mirrors `preprocess_dl3dv.py`: COLMAP's intrinsics describe the 3840x2160
    originals, so scale them to whatever resolution the depth map happens to be
    (it is close to but not exactly the image size -- 539x961 vs 540x960) and let
    one resample do undistortion and resizing together. `K_out` is read back from
    `meta.npz`, so the depth lands in exactly the camera the images were rendered
    through, including the principal point pinned to the centre.
    """
    K_full, dist = camera.to_pinhole()
    dh, dw = depth_hw
    sx, sy = dw / camera.width, dh / camera.height
    K_src = colmap_to_opencv_K(K_full)
    K_src[0, 0] *= sx
    K_src[1, 1] *= sy
    K_src[0, 2] = (K_src[0, 2] + 0.5) * sx - 0.5
    K_src[1, 2] = (K_src[1, 2] + 0.5) * sy - 0.5
    out_h, out_w = out_hw
    return cv2.initUndistortRectifyMap(K_src, dist, None, K_out, (out_w, out_h), cv2.CV_32FC1)


# --------------------------------------------------------------------------- #
# per-scene work
# --------------------------------------------------------------------------- #


def process_scene(task: dict) -> dict:
    subset, scene = task["subset"], task["scene"]
    scene_dir = Path(task["scene_dir"])
    result = {"subset": subset, "scene": scene}
    depth_dir = Path(task["depth_dir"])
    marker = depth_dir / "meta.json"

    if marker.exists() and not task["overwrite"]:
        with open(marker) as f:
            return {**result, "status": "cached", "num_depth": len(json.load(f)["frames"])}

    sparse_dir = sparse_dir_for(Path(task["colmap_root"]), subset, scene)
    if sparse_dir is None:
        return {**result, "status": "error", "reason": "no_colmap"}
    cameras = read_cameras_bin(sparse_dir / "cameras.bin")
    if len(cameras) != 1:
        return {**result, "status": "error", "reason": f"{len(cameras)}_cameras"}
    camera = next(iter(cameras.values()))

    meta = np.load(scene_dir / "meta.npz", allow_pickle=True)
    out_h, out_w = (int(v) for v in meta["image_hw"])
    K = meta["intrinsics"]
    if not np.allclose(K, K[0], atol=1e-4):
        return {**result, "status": "error", "reason": "varying_intrinsics"}
    K_out = K[0].astype(np.float64)

    tmp_dir = Path(task["tmp_root"]) / subset / scene
    try:
        zip_path = hf_hub_download(
            REPO, f"{subset}/{scene}.zip", repo_type="dataset", local_dir=str(tmp_dir)
        )
    except EntryNotFoundError:
        return {**result, "status": "missing"}
    except Exception as exc:  # network flake: leave it for the next run
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {**result, "status": "error", "reason": type(exc).__name__}

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = sorted(n for n in zf.namelist() if n.endswith(".npy"))
            if not members:
                return {**result, "status": "error", "reason": "empty_zip"}

            first = np.load(io.BytesIO(zf.read(members[0])))
            map_x, map_y = pinhole_map(camera, K_out, (out_h, out_w), first.shape)

            stack = np.empty((len(members), out_h, out_w), dtype=np.float32)
            for i, name in enumerate(members):
                src = first if i == 0 else np.load(io.BytesIO(zf.read(name)))
                if src.shape != first.shape:
                    return {**result, "status": "error", "reason": "ragged_depth_shapes"}
                # NEAREST, not AREA/LINEAR: averaging across a depth discontinuity
                # invents surfaces that are in front of both neighbours.
                stack[i] = cv2.remap(src, map_x, map_y, cv2.INTER_NEAREST)

        q, lo, hi, clipped = quantise(stack)
        depth_dir.mkdir(parents=True, exist_ok=True)
        stems = [Path(n).stem for n in members]
        for stem, plane in zip(stems, q):
            cv2.imwrite(str(depth_dir / f"{stem}.png"), plane)

        payload = {
            "lo": lo,
            "hi": hi,
            "clipped_frac": clipped,
            "out_hw": [out_h, out_w],
            "source_hw": list(first.shape),
            "frames": stems,
        }
        with open(marker, "w") as f:
            json.dump(payload, f)
        return {**result, "status": "ok", "num_depth": len(stems)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path.home() / "dl3dv-train", help="read-only: index.json + meta.npz")
    p.add_argument("--depth-root", type=Path, default=Path.home() / "dl3dv-depth", help="where depth is written")
    p.add_argument("--colmap-root", type=Path, default=Path.home() / "dl3dv-cache-960P")
    p.add_argument("--tmp-root", type=Path, default=None, help="scratch for zips (default <depth-root>/.tmp)")
    p.add_argument("--subsets", nargs="*", default=None, help="restrict to these buckets")
    p.add_argument("--limit", type=int, default=0, help="stop after N scenes (for a trial run)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    index_path = args.data_root / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    tmp_root = args.tmp_root or args.depth_root / ".tmp"
    scenes = index["scenes"]
    if args.subsets:
        scenes = [s for s in scenes if s["subset"] in args.subsets]
    if args.limit:
        scenes = scenes[: args.limit]

    tasks = [
        {
            "subset": s["subset"],
            "scene": s["scene"],
            "scene_dir": str(args.data_root / s["path"]),
            "depth_dir": str(args.depth_root / s["subset"] / s["scene"]),
            "colmap_root": str(args.colmap_root),
            "tmp_root": str(tmp_root),
            "overwrite": args.overwrite,
        }
        for s in scenes
    ]
    print(f"[depth] {len(tasks)} scenes -> {args.depth_root}, {args.workers} workers, scratch {tmp_root}")

    counts: dict[str, int] = {}
    depth_counts: dict[tuple[str, str], int] = {}
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_scene, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            r = future.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if "num_depth" in r:
                depth_counts[(r["subset"], r["scene"])] = r["num_depth"]
            if r["status"] == "error":
                print(f"[depth] error {r['subset']}/{r['scene'][:8]}: {r['reason']}")
            if done % 25 == 0 or done == len(futures):
                rate = done / max(time.time() - start, 1e-6)
                eta = (len(futures) - done) / max(rate, 1e-9) / 3600
                print(f"[depth] {done}/{len(futures)}  {rate * 3600:.0f}/h  eta {eta:.1f} h  {counts}")

    # Only touch scenes this run actually attempted, so partial runs stay additive.
    attempted = {(t["subset"], t["scene"]) for t in tasks}
    for s in index["scenes"]:
        key = (s["subset"], s["scene"])
        if key in attempted:
            n = depth_counts.get(key)
            s["has_depth"] = n is not None
            s["num_depth"] = n or 0
    index["depth"] = {"repo": REPO, "root": str(args.depth_root), "counts": counts}
    tmp_index = index_path.with_suffix(".json.tmp")
    with open(tmp_index, "w") as f:
        json.dump(index, f)
    tmp_index.replace(index_path)

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"[depth] done in {(time.time() - start) / 3600:.2f} h: {counts}")
    return 0 if not counts.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
