#!/usr/bin/env python3
"""Stream Hypersim into a VGGT-Omega training set, one camera at a time.

Hypersim is published as one zip per scene (`ai_001_001.zip`, ...) at
`docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/`, each
~4 GB and containing far more than training needs: raw HDR `color.hdf5`,
`_geometry_preview`, `_pre_preview`, surface normals, and so on. Nothing here
downloads a whole zip. Each zip's central directory is read over HTTP range
requests (`RangeReader` below, the same trick `download_megasynth.py` uses over
`HfFileSystem`, just with plain `Range:` headers instead), and only the members
one camera actually needs are pulled out, converted, and the staged copies
deleted -- so peak disk is a few tens of MB per in-flight camera, not 4 GB per
scene, and the transfer is only the frames actually wanted.

Per camera this fetches:

    <camera>_final_preview/frame.<id>.tonemap.jpg          tonemapped RGB
    <camera>_geometry_hdf5/frame.<id>.depth_meters.hdf5    Euclidean ray distance
    _detail/<camera>/camera_keyframe_positions.hdf5        (N,3) asset units
    _detail/<camera>/camera_keyframe_orientations.hdf5     (N,3,3) cam-to-world, OpenGL

plus `_detail/metadata_scene.csv` once per scene, for `meters_per_asset_unit`.
Member names are matched by suffix rather than an assumed exact prefix, because
the top-level folder name inside the zip is not guaranteed (mirroring
`download_megasynth.py`'s `split_dir_prefix`, which had to do the same after the
Hub zips turned out to be packed two different ways).

The scene list and official train/test split come from a CSV checked into the
ml-hypersim GitHub repo (fetched once, cached under `<out>/.split.csv`):

    https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv

Its `(camera_name, frame_id)` rows are also the only reliable source of which
frames actually exist and are usable (`included_in_public_release == True`) --
frame counts are not fixed across cameras, so this is read rather than assumed.

Usage:

    training/download_hypersim.py --limit 50            # 50 <scene>/<camera>
    training/download_hypersim.py --limit 200 --jobs 4
    training/download_hypersim.py --limit 50 --frames-per-scene 48

Resumable and idempotent: a `<scene>_<camera>` with a marker under `<out>/.done`
is skipped, so re-running after an interruption picks up where it left off.
Cameras that failed on a transient network error get no marker and are retried
next run; cameras the preprocessor legitimately rejected are marked so they are
not downloaded again.

Sizing. `--frames-per-scene` is the main cost knob: each kept frame pulls one
~0.3-1 MB tonemap JPEG and one ~2-4 MB depth HDF5 (depth is float32 at
1024x768, uncompressed in most releases), so the default 40 of a typical
~100-frame camera trajectory is ~150 MB per camera. By default frames are
stored at 224x384, matching a DL3DV set built at `--resolution 384` and a
ScanNet/MegaSynth set built with `--target-hw 224 384`, so all four stack in
one batch.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import struct
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.preprocess_hypersim import (  # noqa: E402
    consolidate_failures,
    make_task,
    rebuild_index,
    run_config,
    safe_process,
)

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"
SPLIT_CSV_URL = (
    "https://raw.githubusercontent.com/apple/ml-hypersim/main/"
    "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
)

# A local file header is 30 bytes plus the file name plus an extra field whose
# length is only recorded locally. Over-reading by this much lets one range
# request usually cover the header and the compressed payload together;
# `read_member` falls back to a second request if it is not enough.
LOCAL_HEADER_SLACK = 128


# --------------------------------------------------------------------------- #
# remote zip access
# --------------------------------------------------------------------------- #


class RangeReader(io.RawIOBase):
    """A file-like object over HTTP `Range` requests, for `zipfile.ZipFile`.

    `zipfile` only ever seeks and reads small spans (the central directory,
    then one member at a time), so nothing here needs a read-ahead cache the
    way MegaSynth's `HfFileSystem` handle does -- each `.read()` becomes exactly
    one ranged GET for exactly the bytes zipfile asked for.
    """

    def __init__(self, url: str):
        self._url = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            self._size = int(resp.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            end = self._size - 1
        else:
            end = min(self._pos + size, self._size) - 1
        if end < self._pos:
            return b""
        req = urllib.request.Request(self._url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        self._pos += len(data)
        return data


def open_scene_zip(scene: str) -> tuple[RangeReader, zipfile.ZipFile]:
    handle = RangeReader(f"{BASE_URL}/{scene}.zip")
    return handle, zipfile.ZipFile(handle)


def read_member(handle: RangeReader, info: zipfile.ZipInfo) -> bytes:
    """One member's decompressed bytes, in one HTTP range request (usually)."""
    want = 30 + len(info.filename.encode()) + LOCAL_HEADER_SLACK + info.compress_size
    handle.seek(info.header_offset)
    blob = handle.read(want)
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    if start + info.compress_size > len(blob):  # an unexpectedly long extra field
        handle.seek(info.header_offset + start)
        raw = handle.read(info.compress_size)
    else:
        raw = blob[start : start + info.compress_size]
    if len(raw) != info.compress_size:
        raise IOError(f"short read for {info.filename}: {len(raw)}/{info.compress_size}")
    return zlib.decompress(raw, -15) if info.compress_type == zipfile.ZIP_DEFLATED else raw


class SceneCache:
    """One open scene zip per thread, mirroring `download_megasynth.py`'s SplitCache."""

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self, scene: str):
        state = getattr(self._local, "open", None)
        if state is not None and state[0] == scene:
            return state[1], state[2]
        self.close()
        handle, zf = open_scene_zip(scene)
        self._local.open = (scene, handle, zf)
        return handle, zf

    def close(self) -> None:
        state = getattr(self._local, "open", None)
        if state is not None:
            try:
                state[2].close()
            except Exception:
                pass
            self._local.open = None


def find_member(names: list[str], *suffixes: str) -> str | None:
    """First name ending in any of `suffixes`; defensive against an unknown
    top-level folder name inside the zip (Hypersim's own is not documented to be
    stable across scenes, unlike MegaSynth's two known-fixed layouts)."""
    for name in names:
        if any(name.endswith(suffix) for suffix in suffixes):
            return name
    return None


def find_frame_member(names: list[str], dir_suffix: str, frame_id: str, file_suffix: str) -> str | None:
    """Like `find_member`, but for a per-frame file whose numeric width is not
    fixed by the split CSV.

    The zip stores frame files zero-padded (`frame.0000.tonemap.jpg`), but
    `metadata_images_split_scene_v1.csv`'s `frame_id` column is a plain integer
    (`0`, `1`, ...), so a literal `frame.{frame_id}.` suffix never matches. This
    matches on the integer value instead of a specific width, so it is correct
    regardless of how many digits any given release actually pads to.
    """
    pattern = re.compile(rf"{re.escape(dir_suffix)}/frame\.0*{int(frame_id)}\.{re.escape(file_suffix)}$")
    for name in names:
        if pattern.search(name):
            return name
    return None


# --------------------------------------------------------------------------- #
# split / frame-list metadata
# --------------------------------------------------------------------------- #


def fetch_split_csv(cache_path: Path) -> Path:
    if not cache_path.exists():
        print("fetching the official train/test split...", flush=True)
        urllib.request.urlretrieve(SPLIT_CSV_URL, cache_path)
    return cache_path


def load_worklist(split_csv: Path) -> list[tuple[str, str, list[str]]]:
    """(scene, camera, frame_ids) for every public, usable camera, from the CSV.

    Frame ids are kept as the zero-padded strings the CSV writes them as (the
    exact width is discovered here, per row, rather than assumed -- see the
    module docstring's note on this).
    """
    by_camera: dict[tuple[str, str], list[str]] = {}
    with split_csv.open() as f:
        for row in csv.DictReader(f):
            if row["included_in_public_release"].strip().lower() != "true":
                continue
            key = (row["scene_name"], row["camera_name"])
            by_camera.setdefault(key, []).append(row["frame_id"].strip())
    return [(scene, camera, sorted(set(ids))) for (scene, camera), ids in sorted(by_camera.items())]


def evenly_spaced(total: int, keep: int) -> list[int]:
    """`keep` indices spread across `range(total)`, endpoints included."""
    if keep <= 0 or keep >= total:
        return list(range(total))
    if keep == 1:
        return [0]
    return sorted({round(i * (total - 1) / (keep - 1)) for i in range(keep)})


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #


def stage_camera(
    handle: RangeReader, zf: zipfile.ZipFile, scene: str, camera: str, frame_ids: list[str],
    stage_root: Path, frames: int,
) -> Path:
    """Pull one camera's chosen frames + poses out of the scene zip."""
    names = zf.namelist()

    scene_csv_name = find_member(names, "_detail/metadata_scene.csv")
    if scene_csv_name is None:
        raise FileNotFoundError(f"{scene}: no _detail/metadata_scene.csv in zip")
    scene_csv = read_member(handle, zf.getinfo(scene_csv_name)).decode("utf-8")
    meters_per_unit = None
    for row in csv.DictReader(io.StringIO(scene_csv)):
        if row["parameter_name"] == "meters_per_asset_unit":
            meters_per_unit = float(row["parameter_value"])
            break
    if meters_per_unit is None:
        raise ValueError(f"{scene}: meters_per_asset_unit not found in metadata_scene.csv")

    positions_name = find_member(names, f"_detail/{camera}/camera_keyframe_positions.hdf5")
    orientations_name = find_member(names, f"_detail/{camera}/camera_keyframe_orientations.hdf5")
    if positions_name is None or orientations_name is None:
        raise FileNotFoundError(f"{scene}/{camera}: no camera_keyframe_{{positions,orientations}}.hdf5")
    positions_raw = read_member(handle, zf.getinfo(positions_name))
    orientations_raw = read_member(handle, zf.getinfo(orientations_name))
    with h5py.File(io.BytesIO(positions_raw), "r") as h:
        positions_all = np.asarray(h["dataset"]) * meters_per_unit
    with h5py.File(io.BytesIO(orientations_raw), "r") as h:
        orientations_all = np.asarray(h["dataset"])

    keep = evenly_spaced(len(frame_ids), frames)
    kept_ids = [frame_ids[i] for i in keep]

    dest = stage_root / scene / camera
    dest.mkdir(parents=True, exist_ok=True)

    positions, orientations, kept_final = [], [], []
    for frame_id in kept_ids:
        idx = int(frame_id)
        if idx >= len(positions_all) or idx >= len(orientations_all):
            continue  # the keyframe track is shorter than the frame_id range claims

        jpg_name = find_frame_member(names, f"{camera}_final_preview", frame_id, "tonemap.jpg")
        depth_name = find_frame_member(names, f"{camera}_geometry_hdf5", frame_id, "depth_meters.hdf5")
        if jpg_name is None or depth_name is None:
            continue

        (dest / f"frame.{frame_id}.tonemap.jpg").write_bytes(read_member(handle, zf.getinfo(jpg_name)))
        depth_raw = read_member(handle, zf.getinfo(depth_name))
        with h5py.File(io.BytesIO(depth_raw), "r") as h:
            distance = np.asarray(h["dataset"], dtype=np.float32)
        np.save(dest / f"frame.{frame_id}.depth_meters.npy", distance)

        positions.append(positions_all[idx])
        orientations.append(orientations_all[idx])
        kept_final.append(frame_id)

    np.save(dest / "positions_m.npy", np.stack(positions) if positions else np.zeros((0, 3)))
    np.save(dest / "orientations.npy", np.stack(orientations) if orientations else np.zeros((0, 3, 3)))
    (dest / "frame_ids.json").write_text(json.dumps(kept_final))
    return dest


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def build_queue(args, worklist: list[tuple[str, str, list[str]]]) -> list[tuple[str, str, list[str]]]:
    if args.scenes:
        wanted = set(args.scenes)
        worklist = [w for w in worklist if w[0] in wanted]
    if args.limit:
        worklist = worklist[: args.limit]
    return worklist


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path.home() / "hypersim-train")
    p.add_argument("--stage", type=Path, default=None, help="scratch dir; default <out>/.stage")
    p.add_argument("--limit", type=int, default=0, help="number of <scene>/<camera> (0 = all)")
    p.add_argument("--scenes", nargs="*", default=None, help="restrict to these scene_name values")
    p.add_argument("--frames-per-scene", type=int, default=40,
                   help="views to fetch per camera trajectory; the main transfer knob")
    p.add_argument("--jobs", type=int, default=8,
                   help="cameras in flight. HTTP-latency bound, so this can exceed the core count")

    p.add_argument("--target-hw", type=int, nargs=2, default=(224, 384), metavar=("H", "W"),
                   help="stored shape; the default matches DL3DV at --resolution 384")
    p.add_argument("--fit", choices=("crop", "squash"), default="crop")
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--min-depth-frac", type=float, default=0.2)
    p.add_argument("--covis-grid", type=int, nargs=2, default=(24, 32))
    p.add_argument("--covis-rel-tol", type=float, default=0.1)

    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--keep-raw", action="store_true", help="do not delete the staged frames")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--index-only", action="store_true", help="just rebuild index.json and exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.max_frames = max(args.max_frames, args.frames_per_scene)
    depth_out = args.out / "depth"
    stage_root = args.stage or (args.out / ".stage")
    done_dir = args.out / ".done"
    args.src_root = stage_root  # only read back to label index.json's config

    args.out.mkdir(parents=True, exist_ok=True)
    split_csv = fetch_split_csv(args.out / ".split.csv")

    if args.index_only:
        splits = _splits_from_csv(split_csv)
        index = rebuild_index(args.out, splits, run_config(args, depth_out, True))
        consolidate_failures(args.out)
        print(f"{len(index['scenes'])} scenes ({index['num_train']} train / {index['num_val']} val)")
        return 0

    done_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    worklist = build_queue(args, load_worklist(split_csv))
    marker_ids = [f"{scene}_{camera}" for scene, camera, _ in worklist]
    already = sum((done_dir / mid).exists() for mid in marker_ids)

    print(f"Hypersim -> {args.out}")
    print(f"  {len(worklist)} <scene>/<camera> queued, {already} already done, {args.jobs} in flight")
    print(f"  {args.frames_per_scene} views per camera "
          f"(~{args.frames_per_scene * 3.3:.0f} MB transferred each, "
          f"~{(len(worklist) - already) * args.frames_per_scene * 3.3 / 1024:.0f} GB total)")
    shape = "x".join(str(v) for v in args.target_hw) if args.target_hw else f"native@{args.resolution}"
    print(f"  storing {shape}, staging in {stage_root} (deleted after each camera)")
    print()
    if args.dry_run:
        for scene, camera, frame_ids in worklist[:10]:
            print(f"  {scene}/{camera}  ({len(frame_ids)} frames available)")
        print(f"  ... {len(worklist)} <scene>/<camera>")
        return 0

    scenes = SceneCache()
    counts = {"ok": 0, "cached": 0, "skip": 0, "error": 0, "done": 0, "fail": 0}
    lock = threading.Lock()
    started = time.time()

    def run_camera(job: tuple[str, str, list[str]]) -> None:
        scene, camera, frame_ids = job
        marker_id = f"{scene}_{camera}"
        marker = done_dir / marker_id
        if marker.exists() and not args.overwrite:
            with lock:
                counts["done"] += 1
            return

        staged = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                remote, zf = scenes.get(scene)
                staged = stage_camera(remote, zf, scene, camera, frame_ids, stage_root, args.frames_per_scene)
                break
            except Exception as exc:
                # A dropped connection leaves the handle unusable, so throw it
                # away rather than retrying through it.
                scenes.close()
                shutil.rmtree(stage_root / scene / camera, ignore_errors=True)
                if attempt >= max(1, args.retries):
                    with lock:
                        counts["fail"] += 1
                    print(f"[{'fail':<6}] {marker_id}: {type(exc).__name__}: {exc}", flush=True)
                    return
                time.sleep(attempt * 5)

        try:
            result = safe_process(make_task(args, marker_id, staged, depth_out))
        finally:
            if not args.keep_raw:
                shutil.rmtree(stage_root / scene / camera, ignore_errors=True)
                if not any((stage_root / scene).iterdir()) if (stage_root / scene).is_dir() else False:
                    shutil.rmtree(stage_root / scene, ignore_errors=True)

        status = result["status"]
        with lock:
            counts[status] = counts.get(status, 0) + 1
            n = sum(counts[k] for k in ("ok", "cached", "skip", "error", "done", "fail"))
        if status in ("ok", "cached", "skip"):
            # Marked even when skipped: a camera the preprocessor rejected will
            # be rejected again, so there is no point paying for its bytes twice.
            marker.touch()
        else:
            (args.out / "failures").mkdir(parents=True, exist_ok=True)
            (args.out / "failures" / f"{marker_id}.json").write_text(json.dumps(result))

        rate = n / max(time.time() - started, 1e-6)
        detail = (
            f"frames={result['num_frames']} valid={100 * result.get('depth_valid', 0):.0f}%"
            if status in ("ok", "cached")
            else f"{result.get('reason')} ({result.get('detail')})"
        )
        print(
            f"[{status:<6}] {marker_id} {detail} "
            f"[{n}/{len(worklist)} {rate * 3600:.0f}/h eta {(len(worklist) - n) / max(rate, 1e-9) / 3600:.1f}h]",
            flush=True,
        )

    def guarded(job: tuple[str, str, list[str]]) -> None:
        # One camera must not take the pool down with it: the whole point of the
        # .done markers is that a run gets as far as it can and resumes.
        try:
            run_camera(job)
        except Exception as exc:
            with lock:
                counts["fail"] += 1
            print(f"[{'fail':<6}] {job[0]}_{job[1]}: {type(exc).__name__}: {exc}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(guarded, worklist))
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume", flush=True)

    print("\nrebuilding index...")
    splits = _splits_from_csv(split_csv)
    index = rebuild_index(args.out, splits, run_config(args, depth_out, True))
    consolidate_failures(args.out)
    frames = sum(e["num_frames"] or 0 for e in index["scenes"])
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e9
    print(
        f"{len(index['scenes'])} scenes ({index['num_train']} train / {index['num_val']} val), "
        f"{frames} frames, {size:.1f} GB in {args.out}; this run {counts} "
        f"in {(time.time() - started) / 60:.0f}m"
    )
    print()
    print("train with:")
    print(f"  --hypersim-root {args.out}")
    return 0


def _splits_from_csv(split_csv: Path) -> dict[str, str]:
    """`<scene>_<camera> -> train/val`, majority vote per camera (see
    `preprocess_hypersim.load_split`'s docstring for why a vote at all)."""
    votes: dict[str, dict[str, int]] = {}
    with split_csv.open() as f:
        for row in csv.DictReader(f):
            if row["included_in_public_release"].strip().lower() != "true":
                continue
            key = f"{row['scene_name']}_{row['camera_name']}"
            partition = "val" if row["split_partition_name"].strip().lower() == "test" else "train"
            votes.setdefault(key, {"train": 0, "val": 0})[partition] += 1
    return {key: max(counts, key=counts.get) for key, counts in votes.items()}


if __name__ == "__main__":
    raise SystemExit(main())
