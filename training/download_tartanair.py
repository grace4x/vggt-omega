#!/usr/bin/env python3
"""Stream TartanAir into a VGGT-Omega training set, one trajectory at a time.

TartanAir (Wang et al., 2020) is ~1000 stereo trajectories across ~30
photorealistic AirSim environments, published as one zip per
`<env>/<difficulty>/<modality>_<camera>` on the AirLab Ceph RGW mirror --
`image_left.zip` for an env/difficulty holds *every* trajectory's RGB frames
plus both `pose_left.txt`/`pose_right.txt` files, `depth_left.zip` holds every
trajectory's depth, and each can be several GB. Nothing here downloads a whole
zip: `RangeReader` opens one over ranged `GetObject` calls the same way
`download_hypersim.py`'s `RangeReader` opens a Hypersim scene zip over ranged
HTTP GETs, so only the central directory and the members actually wanted are
ever transferred.

Per trajectory this fetches (from `image_left.zip`, once per env/difficulty):

    <env>/<difficulty>/<traj>/image_left/<frame>_left.png    640x480 RGB
    <env>/<difficulty>/<traj>/pose_left.txt                  one 7-float line/frame

and, from `depth_left.zip`:

    <env>/<difficulty>/<traj>/depth_left/<frame>_left_depth.npy   640x480 f32 planar z

The env/difficulty/trajectory universe and each trajectory's frame count come
from `download_training_zipfiles.txt` (castacks/tartanair_tools; cached under
`<out>/.manifest.txt`) plus one directory listing of each env/difficulty's
`image_left.zip` (its central directory, not its contents).

Usage:

    training/download_tartanair.py --limit 50            # 50 <env>/<difficulty>/<traj>
    training/download_tartanair.py --limit 200 --jobs 4
    training/download_tartanair.py --limit 50 --frames-per-traj 48

Resumable and idempotent: a `<env>_<difficulty>_<traj>` with a marker under
`<out>/.done` is skipped, so re-running after an interruption picks up where it
left off. Trajectories that failed on a transient network error get no marker
and are retried next run; trajectories the preprocessor legitimately rejected
are marked so they are not downloaded again.

Train/val split: TartanAir has no official held-out set (see
`preprocess_tartanair.py`'s docstring), so whole *environments* are held out --
never individual trajectories from an otherwise-trained environment, since
consecutive frames within one env are highly correlated.
"""

from __future__ import annotations

import argparse
import hashlib
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.preprocess_tartanair import (  # noqa: E402
    consolidate_failures,
    make_task,
    rebuild_index,
    run_config,
    safe_process,
)

MANIFEST_URL = (
    "https://raw.githubusercontent.com/castacks/tartanair_tools/master/download_training_zipfiles.txt"
)
ENDPOINT_URL = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
BUCKET = "tartanair"

# See `download_hypersim.py`'s `RangeReader` for why this slack exists.
LOCAL_HEADER_SLACK = 128


# --------------------------------------------------------------------------- #
# remote zip access
# --------------------------------------------------------------------------- #


class RangeReader(io.RawIOBase):
    """A file-like object over ranged `GetObject` calls, for `zipfile.ZipFile`.

    Same role as `download_hypersim.py`'s `RangeReader`, just against the
    AirLab Ceph RGW S3 endpoint (anonymous, `boto3` + `UNSIGNED`) instead of
    plain HTTPS -- the bucket is not reachable with an unsigned `urllib` GET.
    """

    def __init__(self, client, bucket: str, key: str):
        self._client = client
        self._bucket = bucket
        self._key = key
        self._pos = 0
        self._size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]

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
        resp = self._client.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={self._pos}-{end}"
        )
        data = resp["Body"].read()
        self._pos += len(data)
        return data


def make_s3_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as exc:
        raise ImportError("boto3 is required: pip install boto3") from exc
    return boto3.client("s3", endpoint_url=ENDPOINT_URL, config=Config(signature_version=UNSIGNED))


def open_zip(client, key: str) -> tuple[RangeReader, zipfile.ZipFile]:
    handle = RangeReader(client, BUCKET, key)
    return handle, zipfile.ZipFile(handle)


def read_member(handle: RangeReader, info: zipfile.ZipInfo) -> bytes:
    """One member's decompressed bytes, in one ranged `GetObject` (usually)."""
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


class EnvCache:
    """One open (image_left.zip, depth_left.zip) pair per thread, per env/difficulty."""

    def __init__(self, client) -> None:
        self._client = client
        self._local = threading.local()

    def get(self, env: str, difficulty: str):
        key = (env, difficulty)
        state = getattr(self._local, "open", None)
        if state is not None and state[0] == key:
            return state[1]
        self.close()
        image_h, image_zf = open_zip(self._client, f"{env}/{difficulty}/image_left.zip")
        depth_h, depth_zf = open_zip(self._client, f"{env}/{difficulty}/depth_left.zip")
        bundle = (image_h, image_zf, depth_h, depth_zf)
        self._local.open = (key, bundle)
        return bundle

    def close(self) -> None:
        state = getattr(self._local, "open", None)
        if state is not None:
            for handle_or_zf in state[1]:
                try:
                    handle_or_zf.close()
                except Exception:
                    pass
            self._local.open = None


FRAME_RE = re.compile(r"(?P<traj>P\d+)/image_left/(?P<frame>\d+)_left\.png$")


def list_trajectories(image_zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """`{trajectory: sorted frame ids}` from an `image_left.zip`'s namelist."""
    by_traj: dict[str, list[str]] = {}
    for name in image_zf.namelist():
        m = FRAME_RE.search(name)
        if m:
            by_traj.setdefault(m.group("traj"), []).append(m.group("frame"))
    return {t: sorted(ids) for t, ids in by_traj.items()}


def evenly_spaced(total: int, keep: int) -> list[int]:
    if keep <= 0 or keep >= total:
        return list(range(total))
    if keep == 1:
        return [0]
    return sorted({round(i * (total - 1) / (keep - 1)) for i in range(keep)})


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def fetch_manifest(cache_path: Path) -> Path:
    if not cache_path.exists():
        print("fetching the scene/difficulty/modality manifest...", flush=True)
        urllib.request.urlretrieve(MANIFEST_URL, cache_path)
    return cache_path


def load_env_difficulty_pairs(manifest_path: Path) -> list[tuple[str, str]]:
    """`(env, difficulty)` for every pair that has both `image_left.zip` and
    `depth_left.zip` -- everything this preprocessor needs; `seg`/`flow` zips
    the manifest may also list are irrelevant here."""
    have_image, have_depth = set(), set()
    for line in manifest_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or not parts[0].endswith(".zip"):
            continue
        env, difficulty, filename = parts[0].split("/")
        if filename == "image_left.zip":
            have_image.add((env, difficulty))
        elif filename == "depth_left.zip":
            have_depth.add((env, difficulty))
    return sorted(have_image & have_depth)


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #


def stage_trajectory(
    image_h: RangeReader, image_zf: zipfile.ZipFile, depth_h: RangeReader, depth_zf: zipfile.ZipFile,
    env: str, difficulty: str, traj: str, frame_ids: list[str], stage_root: Path, frames: int,
) -> Path:
    """Pull one trajectory's chosen frames + poses out of the env/difficulty zips."""
    prefix = f"{env}/{difficulty}/{traj}"
    pose_name = f"{prefix}/pose_left.txt"
    pose_lines = read_member(image_h, image_zf.getinfo(pose_name)).decode("utf-8").splitlines()
    poses_all = np.asarray([[float(v) for v in line.split()] for line in pose_lines if line.strip()])
    if len(poses_all) != len(frame_ids):
        raise ValueError(f"{prefix}: {len(poses_all)} poses for {len(frame_ids)} frames")

    keep = evenly_spaced(len(frame_ids), frames)
    kept_ids = [frame_ids[i] for i in keep]

    dest = stage_root / env / difficulty / traj
    dest.mkdir(parents=True, exist_ok=True)

    poses, kept_final = [], []
    for i, frame_id in zip(keep, kept_ids):
        img_name = f"{prefix}/image_left/{frame_id}_left.png"
        depth_name = f"{prefix}/depth_left/{frame_id}_left_depth.npy"
        try:
            img_info = image_zf.getinfo(img_name)
            depth_info = depth_zf.getinfo(depth_name)
        except KeyError:
            continue  # a frame image_left.zip lists that depth_left.zip does not (rare, seen on a few trajs)

        (dest / f"{frame_id}_left.png").write_bytes(read_member(image_h, img_info))
        depth_raw = read_member(depth_h, depth_info)
        depth = np.load(io.BytesIO(depth_raw)).astype(np.float32)
        np.save(dest / f"{frame_id}_left_depth.npy", depth)

        poses.append(poses_all[i])
        kept_final.append(frame_id)

    np.save(dest / "poses.npy", np.stack(poses) if poses else np.zeros((0, 7)))
    (dest / "frame_ids.json").write_text(json.dumps(kept_final))
    return dest


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def assign_splits(envs: list[str], val_frac: float, seed: int) -> dict[str, str]:
    """Whole environments held out, never individual trajectories -- see the
    module docstring for why a trajectory-level split is the wrong unit here."""
    out = {}
    for env in envs:
        digest = hashlib.blake2b(f"{seed}:{env}".encode(), digest_size=8).digest()
        out[env] = "val" if int.from_bytes(digest, "big") / 2.0**64 < val_frac else "train"
    return out


def build_queue(args, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if args.envs:
        wanted = set(args.envs)
        pairs = [p for p in pairs if p[0] in wanted]
    if args.only_easy:
        pairs = [p for p in pairs if p[1] == "Easy"]
    if args.only_hard:
        pairs = [p for p in pairs if p[1] == "Hard"]
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path.home() / "tartanair-train")
    p.add_argument("--stage", type=Path, default=None, help="scratch dir; default <out>/.stage")
    p.add_argument("--limit", type=int, default=0, help="number of <env>/<difficulty>/<traj> (0 = all)")
    p.add_argument("--envs", nargs="*", default=None, help="restrict to these environment names")
    p.add_argument("--only-easy", action="store_true")
    p.add_argument("--only-hard", action="store_true")
    p.add_argument("--frames-per-traj", type=int, default=40,
                   help="views to fetch per trajectory; the main transfer knob")
    p.add_argument("--jobs", type=int, default=8,
                   help="trajectories in flight. HTTP-latency bound, so this can exceed the core count")

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

    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of *environments* held out, not trajectories -- see module docstring")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--keep-raw", action="store_true", help="do not delete the staged frames")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--index-only", action="store_true", help="just rebuild index.json and exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.only_easy and args.only_hard:
        raise SystemExit("--only-easy and --only-hard are mutually exclusive")

    args.max_frames = max(args.max_frames, args.frames_per_traj)
    depth_out = args.out / "depth"
    stage_root = args.stage or (args.out / ".stage")
    done_dir = args.out / ".done"
    args.src_root = stage_root  # only read back to label index.json's config

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest(args.out / ".manifest.txt")
    pairs = build_queue(args, load_env_difficulty_pairs(manifest))

    if args.index_only:
        splits = _traj_splits(args.out, pairs, args.val_frac, args.seed)
        index = rebuild_index(args.out, splits, run_config(args, depth_out))
        consolidate_failures(args.out)
        print(f"{len(index['scenes'])} scenes ({index['num_train']} train / {index['num_val']} val)")
        return 0

    done_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    print(f"TartanAir -> {args.out}")
    print(f"  discovering trajectories in {len(pairs)} <env>/<difficulty> zip pairs...")
    client = make_s3_client()
    worklist: list[tuple[str, str, str, list[str]]] = []
    for env, difficulty in pairs:
        try:
            _, image_zf = open_zip(client, f"{env}/{difficulty}/image_left.zip")
            for traj, frame_ids in sorted(list_trajectories(image_zf).items()):
                worklist.append((env, difficulty, traj, frame_ids))
            image_zf.close()
        except Exception as exc:
            print(f"  [skip] {env}/{difficulty}: {type(exc).__name__}: {exc}")

    if args.limit:
        worklist = worklist[: args.limit]

    marker_ids = [f"{env}_{diff}_{traj}" for env, diff, traj, _ in worklist]
    already = sum((done_dir / mid).exists() for mid in marker_ids)

    print(f"  {len(worklist)} <env>/<difficulty>/<traj> queued, {already} already done, {args.jobs} in flight")
    print(f"  {args.frames_per_traj} views per trajectory")
    shape = "x".join(str(v) for v in args.target_hw) if args.target_hw else f"native@{args.resolution}"
    print(f"  storing {shape}, staging in {stage_root} (deleted after each trajectory)")
    print()
    if args.dry_run:
        for env, difficulty, traj, frame_ids in worklist[:10]:
            print(f"  {env}/{difficulty}/{traj}  ({len(frame_ids)} frames available)")
        print(f"  ... {len(worklist)} <env>/<difficulty>/<traj>")
        return 0

    envs = EnvCache(client)
    counts = {"ok": 0, "cached": 0, "skip": 0, "error": 0, "done": 0, "fail": 0}
    lock = threading.Lock()
    started = time.time()

    def run_traj(job: tuple[str, str, str, list[str]]) -> None:
        env, difficulty, traj, frame_ids = job
        marker_id = f"{env}_{difficulty}_{traj}"
        marker = done_dir / marker_id
        if marker.exists() and not args.overwrite:
            with lock:
                counts["done"] += 1
            return

        staged = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                image_h, image_zf, depth_h, depth_zf = envs.get(env, difficulty)
                staged = stage_trajectory(
                    image_h, image_zf, depth_h, depth_zf, env, difficulty, traj, frame_ids,
                    stage_root, args.frames_per_traj,
                )
                break
            except Exception as exc:
                envs.close()  # a dropped connection leaves the handle unusable
                shutil.rmtree(stage_root / env / difficulty / traj, ignore_errors=True)
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
                shutil.rmtree(stage_root / env / difficulty / traj, ignore_errors=True)

        status = result["status"]
        with lock:
            counts[status] = counts.get(status, 0) + 1
            n = sum(counts[k] for k in ("ok", "cached", "skip", "error", "done", "fail"))
        if status in ("ok", "cached", "skip"):
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

    def guarded(job: tuple[str, str, str, list[str]]) -> None:
        try:
            run_traj(job)
        except Exception as exc:
            with lock:
                counts["fail"] += 1
            print(f"[{'fail':<6}] {job[0]}_{job[1]}_{job[2]}: {type(exc).__name__}: {exc}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(guarded, worklist))
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume", flush=True)

    print("\nrebuilding index...")
    splits = _traj_splits(args.out, pairs, args.val_frac, args.seed)
    index = rebuild_index(args.out, splits, run_config(args, depth_out))
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
    print(f"  --tartanair-root {args.out}")
    return 0


def _traj_splits(out_root: Path, pairs: list[tuple[str, str]], val_frac: float, seed: int) -> dict[str, str]:
    """`<env>_<difficulty>_<traj> -> train/val`, one draw per environment (see
    `assign_splits`) applied to every trajectory found on disk under it."""
    envs = sorted({env for env, _ in pairs})
    per_env = assign_splits(envs, val_frac, seed)
    splits = {}
    for meta_path in (out_root / "scenes" / "tartanair").glob("*/meta.npz"):
        scene = meta_path.parent.name
        # `<env>_<difficulty>_<traj>`; `env` itself may contain underscores
        # (e.g. `abandonedfactory_night`), so split from the right instead.
        env = scene.rsplit("_", 2)[0]
        splits[scene] = per_env.get(env, "train")
    return splits


if __name__ == "__main__":
    raise SystemExit(main())
