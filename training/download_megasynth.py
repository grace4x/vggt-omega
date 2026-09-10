#!/usr/bin/env python3
"""Stream MegaSynth into a VGGT-Omega training set, one scene at a time.

MegaSynth is published as 100 zips of 672 procedurally generated scenes each
(`hf.co/datasets/hwjiang/MegaSynth`, ~40 GB per zip, ~4 TB in total). Nothing
here downloads a whole zip. Each zip's central directory is read over HTTP range
requests, and then each scene's members are pulled out individually, preprocessed
into the DL3DV contract, and deleted -- so peak disk is (jobs x ~40 MB) of
staging plus the output, not 40 GB, and the transfer is only the scenes and
frames actually wanted.

Usage:

    training/download_megasynth.py --limit 5000        # 5000 scenes
    training/download_megasynth.py --limit 200 --jobs 4
    training/download_megasynth.py --limit 5000 --frames-per-scene 48

Resumable and idempotent: a scene with a marker under `<out>/.done` is skipped,
so re-running after an interruption picks up where it left off. Scenes that
failed on a transient network error get no marker and are retried next run;
scenes the preprocessor legitimately rejected are marked so they are not
downloaded again.

The Hub zips are packed two ways (`split_0/<uuid>/...` vs `data/split_N/<uuid>/...`).
Scene listing and member reads both search for `split_N/` rather than assuming it
is the first path component, and a `.zipindex` that is not a list of UUIDs is
rebuilt instead of reused.

Sizing. `--frames-per-scene` is the main cost knob, because the transfer is
almost entirely the renders: each of the 48 views is a ~0.5 MB RGBA PNG plus a
~0.8 MB float EXR, so a full scene is ~61 MB and the default 24 views is ~31 MB.
The default is a deliberate halving -- `train.py --num-frames 16` draws a
covisibility walk from whatever is stored, and 24 candidate views is ample for
that while 48 doubles a 150 GB transfer for little extra spatial coverage.
Output is ~2.5 MB per scene at the default shape, so 5000 scenes is ~12 GB on
disk against ~150 GB pulled over the wire.

By default frames are stored at 224x384, matching a DL3DV set built at
`--resolution 384` and a ScanNet set built with `--target-hw 224 384`, so all
three stack in one batch. The renders are square, so this crops ~42% of the
vertical field of view; `--fit squash` keeps all of it and stretches instead,
and `--target-hw` with no value at all stores the native square shape (which
then only trains on its own, or with `--batch-size 1`).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfFileSystem  # noqa: E402

from training.preprocess_megasynth import (  # noqa: E402
    consolidate_failures,
    make_task,
    rebuild_index,
    run_config,
    safe_process,
)

REPO = "datasets/hwjiang/MegaSynth"
NUM_SPLITS = 100
SCENE_DIR = "hanwen"
# Scene folders are UUID names. Used to reject a poisoned `.zipindex` (the old
# parser stored a single line `data` for every zip after split_0).
SCENE_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# A local file header is 30 bytes plus the file name plus an extra field whose
# length is only recorded locally. Over-reading by this much lets one range
# request cover the header and the compressed payload together; 128 bytes is far
# more than the zip64/timestamp extras these archives actually carry, and
# `read_member` falls back to a second request if it ever is not.
LOCAL_HEADER_SLACK = 128


# --------------------------------------------------------------------------- #
# remote zip access
# --------------------------------------------------------------------------- #


def open_split(index: int):
    """(handle, ZipFile) for one remote split, reading exact byte ranges.

    `cache_type="none"` is the whole trick. fsspec's default read-ahead cache
    would round every access up to a 5 MB block, and since the ~96 members of one
    scene are spread over a contiguous ~61 MB span, fetching a subset of the
    frames would end up pulling the entire span anyway. With no cache each read
    is exactly the range asked for, so skipping a frame really does skip its
    bytes. `zipfile` still parses the central directory, which is the part worth
    not reimplementing (these are zip64 archives).
    """
    handle = HfFileSystem().open(f"{REPO}/split_{index}.zip", "rb", cache_type="none")
    return handle, zipfile.ZipFile(handle)


def read_member(handle, info: zipfile.ZipInfo) -> bytes:
    """One member's decompressed bytes, in one HTTP range request."""
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


class SplitCache:
    """One open split per thread.

    Opening a split costs an 11 MB read of its central directory, so a handle is
    kept for reuse -- but only one at a time, since the queue is walked in split
    order and a stale handle is just an idle socket.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self, index: int):
        state = getattr(self._local, "open", None)
        if state is not None and state[0] == index:
            return state[1], state[2]
        self.close()
        handle, zf = open_split(index)
        self._local.open = (index, handle, zf)
        return handle, zf

    def close(self) -> None:
        state = getattr(self._local, "open", None)
        if state is not None:
            try:
                state[2].close()
                state[1].close()
            except Exception:
                pass
            self._local.open = None


def split_dir_prefix(names: list[str], index: int) -> str:
    """Directory inside the zip that holds scene folders, including the slash.

    The Hub zips were not packed consistently. `split_0.zip` members look like
    `split_0/<uuid>/hanwen/...`; every later split is `data/split_N/<uuid>/...`.
    The old parser took the first path component, so splits 1-99 collapsed to a
    single fake scene named `data` and `--limit 5000` only ever queued split_0.
    """
    token = f"split_{index}/"
    for name in names:
        pos = name.find(token)
        if pos == 0 or (pos > 0 and name[pos - 1] == "/"):
            return name[: pos + len(token)]
    raise ValueError(f"split_{index}.zip has no members under {token!r}")


def split_dir_prefix_of(zf: zipfile.ZipFile, index: int) -> str:
    cached = getattr(zf, "_megasynth_root", None)
    if cached is not None:
        return cached
    root = split_dir_prefix(zf.namelist(), index)
    zf._megasynth_root = root
    return root


def scenes_from_names(names: list[str], index: int) -> list[str]:
    """Scene UUIDs in one split, sorted so `--limit` is a stable prefix."""
    root = split_dir_prefix(names, index)
    seen: set[str] = set()
    for name in names:
        if not name.startswith(root):
            continue
        scene, sep, tail = name[len(root) :].partition("/")
        if sep and tail and SCENE_ID.fullmatch(scene):
            seen.add(scene)
    if not seen:
        raise ValueError(f"split_{index}.zip: no scene UUIDs under {root!r}")
    return sorted(seen)


def scenes_in(zf: zipfile.ZipFile, index: int) -> list[str]:
    """The scene ids in one split, in a fixed order.

    Sorted, so the same `--limit` always takes the same scenes. The ids are
    random uuids and the scenes are independent procedural samples, so a prefix
    of the sorted order is as uniform a sample as a shuffle would be, and it does
    not depend on a seed.
    """
    return scenes_from_names(zf.namelist(), index)


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #


def evenly_spaced(total: int, keep: int) -> list[int]:
    """`keep` indices spread across `range(total)`, endpoints included."""
    if keep <= 0 or keep >= total:
        return list(range(total))
    if keep == 1:
        return [0]
    return sorted({round(i * (total - 1) / (keep - 1)) for i in range(keep)})


def stage_scene(handle, zf, index: int, scene: str, stage_root: Path, frames: int) -> Path:
    """Pull one scene's chosen views out of the zip. Returns its staged root."""
    prefix = f"{split_dir_prefix_of(zf, index)}{scene}/{SCENE_DIR}/"
    cameras = json.loads(read_member(handle, zf.getinfo(prefix + "opencv_cameras.json")))
    keep = evenly_spaced(len(cameras["frames"]), frames)
    cameras["frames"] = [cameras["frames"][i] for i in keep]

    dest = stage_root / scene / SCENE_DIR
    (dest / "renderings").mkdir(parents=True, exist_ok=True)
    (dest / "opencv_cameras.json").write_text(json.dumps(cameras))

    wanted = []
    for frame in cameras["frames"]:
        stem = Path(frame["file_path"]).name[: -len("_rgba.png")]
        wanted += [f"renderings/{stem}_rgba.png", f"renderings/{stem}_depth.exr"]
    # Ascending offset, so the requests walk the scene's span forwards rather
    # than seeking back and forth across 60 MB.
    infos = sorted((zf.getinfo(prefix + name) for name in wanted), key=lambda i: i.header_offset)
    for info in infos:
        (dest / "renderings" / Path(info.filename).name).write_bytes(read_member(handle, info))
    return stage_root / scene


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def parse_splits(spec: str) -> list[int]:
    """"0-7", "3", "0-3,90-93" -> a list of split indices."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
        else:
            lo = hi = int(part)
        if not (0 <= lo <= hi < NUM_SPLITS):
            raise SystemExit(f"--splits out of range: {part!r} (0-{NUM_SPLITS - 1})")
        out += list(range(lo, hi + 1))
    if not out:
        raise SystemExit("--splits selected nothing")
    return out


def build_queue(args, splits: SplitCache) -> list[tuple[int, str]]:
    """(split, scene) pairs up to --limit, opening only the splits needed.

    Each split's scene list is cached under `<out>/.zipindex`, so a rerun does
    not spend an 11 MB central-directory read per split just to work out what it
    already did last time.
    """
    cache_dir = args.out / ".zipindex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    queue: list[tuple[int, str]] = []
    for index in parse_splits(args.splits):
        if args.limit and len(queue) >= args.limit:
            break
        cached = cache_dir / f"split_{index}.txt"
        names: list[str] | None = None
        if cached.exists():
            names = [n for n in cached.read_text().split() if SCENE_ID.fullmatch(n)]
            if not names:
                # Pre-fix caches for split_1+ are a single line `data`. Drop them
                # rather than queueing 99 fake scenes and stopping far short of
                # --limit.
                print(f"reading the index of split_{index} (replacing a stale zipindex)...", flush=True)
                names = None
        if names is None:
            if not cached.exists():
                print(f"reading the index of split_{index}...", flush=True)
            _, zf = splits.get(index)
            names = scenes_in(zf, index)
            cached.write_text("\n".join(names) + "\n")
        room = args.limit - len(queue) if args.limit else len(names)
        queue += [(index, scene) for scene in names[:room]]
    return queue


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path.home() / "megasynth-train")
    p.add_argument("--stage", type=Path, default=None, help="scratch dir; default <out>/.stage")
    p.add_argument("--limit", type=int, default=0, help="number of scenes (0 = all 67200)")
    p.add_argument("--splits", default=f"0-{NUM_SPLITS - 1}", help="which split_N.zip to draw from")
    p.add_argument("--frames-per-scene", type=int, default=24,
                   help="views to fetch of the 48 rendered; the main transfer knob")
    p.add_argument("--jobs", type=int, default=12,
                   help="scenes in flight. The work is HTTP-latency bound, so this can exceed the core count")

    p.add_argument("--target-hw", type=int, nargs=2, default=(224, 384), metavar=("H", "W"),
                   help="stored shape; the default matches DL3DV at --resolution 384")
    p.add_argument("--fit", choices=("crop", "squash"), default="crop")
    p.add_argument("--native-shape", action="store_true",
                   help="ignore --target-hw and store the square render at --resolution")
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--mode", choices=("max_size", "balanced"), default="max_size")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-frames", type=int, default=48)
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--min-depth-frac", type=float, default=0.2)
    p.add_argument("--covis-grid", type=int, nargs=2, default=(24, 32))
    p.add_argument("--covis-rel-tol", type=float, default=0.1)

    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--keep-raw", action="store_true", help="do not delete the staged renders")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--index-only", action="store_true", help="just rebuild index.json and exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.native_shape:
        args.target_hw = None
    depth_out = args.out / "depth"
    stage_root = args.stage or (args.out / ".stage")
    done_dir = args.out / ".done"
    args.src_root = stage_root  # only read back to label index.json's config

    if args.index_only:
        index = rebuild_index(args.out, args.val_frac, args.seed, run_config(args, depth_out))
        consolidate_failures(args.out)
        print(f"{len(index['scenes'])} scenes ({index['num_train']} train / {index['num_val']} val)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    splits = SplitCache()
    queue = build_queue(args, splits)
    already = sum((done_dir / scene).exists() for _, scene in queue)

    shape = "x".join(str(v) for v in args.target_hw) if args.target_hw else f"native@{args.resolution}"
    print(f"MegaSynth -> {args.out}")
    print(f"  {len(queue)} scenes queued, {already} already done, {args.jobs} in flight")
    print(f"  {args.frames_per_scene}/48 views per scene (~{args.frames_per_scene * 1.28:.0f} MB "
          f"transferred each, ~{(len(queue) - already) * args.frames_per_scene * 1.28 / 1024:.0f} GB total)")
    print(f"  storing {shape}, staging in {stage_root} (deleted after each scene)")
    print()
    if args.dry_run:
        for index, scene in queue[:10]:
            print(f"  split_{index}/{scene}")
        print(f"  ... {len(queue)} scenes")
        return 0

    counts = {"ok": 0, "cached": 0, "skip": 0, "error": 0, "done": 0, "fail": 0}
    lock = threading.Lock()
    started = time.time()

    def run_scene(job: tuple[int, str]) -> None:
        index, scene = job
        marker = done_dir / scene
        if marker.exists() and not args.overwrite:
            with lock:
                counts["done"] += 1
            return

        staged = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                remote, zf = splits.get(index)
                staged = stage_scene(remote, zf, index, scene, stage_root, args.frames_per_scene)
                break
            except Exception as exc:
                # A dropped connection leaves the handle unusable, so throw it
                # away rather than retrying through it.
                splits.close()
                shutil.rmtree(stage_root / scene, ignore_errors=True)
                if attempt >= max(1, args.retries):
                    with lock:
                        counts["fail"] += 1
                    print(f"[{'fail':<6}] {scene}: {type(exc).__name__}: {exc}", flush=True)
                    return
                time.sleep(attempt * 5)

        try:
            result = safe_process(make_task(args, scene, staged, depth_out))
        finally:
            if not args.keep_raw:
                shutil.rmtree(stage_root / scene, ignore_errors=True)

        status = result["status"]
        with lock:
            counts[status] = counts.get(status, 0) + 1
            n = sum(counts[k] for k in ("ok", "cached", "skip", "error", "done", "fail"))
        if status in ("ok", "cached", "skip"):
            # Marked even when skipped: a scene the preprocessor rejected will be
            # rejected again, so there is no point paying for its bytes twice.
            marker.touch()
        else:
            # An error is worth retrying -- unlike a skip it usually means a
            # truncated member rather than a scene that cannot work.
            (args.out / "failures").mkdir(parents=True, exist_ok=True)
            (args.out / "failures" / f"{scene}.json").write_text(json.dumps(result))

        rate = n / max(time.time() - started, 1e-6)
        detail = (
            f"frames={result['num_frames']} valid={100 * result.get('depth_valid', 0):.0f}%"
            if status in ("ok", "cached")
            else f"{result.get('reason')} ({result.get('detail')})"
        )
        print(
            f"[{status:<6}] {scene} {detail} "
            f"[{n}/{len(queue)} {rate * 3600:.0f}/h eta {(len(queue) - n) / max(rate, 1e-9) / 3600:.1f}h]",
            flush=True,
        )

    def guarded(job: tuple[int, str]) -> None:
        # One scene must not take the pool down with it: `pool.map` abandons the
        # rest of the queue on the first exception, and the whole point of the
        # .done markers is that a run gets as far as it can and resumes.
        try:
            run_scene(job)
        except Exception as exc:
            with lock:
                counts["fail"] += 1
            print(f"[{'fail':<6}] {job[1]}: {type(exc).__name__}: {exc}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(guarded, queue))
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume", flush=True)

    print("\nrebuilding index...")
    index = rebuild_index(args.out, args.val_frac, args.seed, run_config(args, depth_out))
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
    print(f"  --megasynth-root {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
