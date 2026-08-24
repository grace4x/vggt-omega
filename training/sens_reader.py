"""Reader for ScanNet's `.sens` (`SensorData`) container.

One `.sens` per scan holds everything VGGT-Omega needs to train -- RGB, metric
depth, per-frame pose and both intrinsics -- so it is the only file type worth
downloading. Verified against `scene0366_00.sens` (version 4, StructureSensor).

Layout, little-endian throughout:

    u32          version                     -- 4; anything else is unsupported
    u64 + bytes  sensor_name
    f32[16]      intrinsic_color   (4x4, row major)
    f32[16]      extrinsic_color
    f32[16]      intrinsic_depth
    f32[16]      extrinsic_depth
    u32          color_compression           -- 0 raw, 1 png, 2 jpeg
    u32          depth_compression           -- 0 raw, 1 zlib, 2 occi
    u32 u32      color_width, color_height
    u32 u32      depth_width, depth_height
    f32          depth_shift                 -- divide the u16 depth by this for metres
    u64          num_frames
    per frame:
        f32[16]  camera_to_world
        u64 u64  timestamp_color, timestamp_depth
        u64 u64  color_size_bytes, depth_size_bytes
        bytes    color_data
        bytes    depth_data
    u64 + blocks IMU measurements               -- trailing, ignored

Note the two `*_compression` enums are *different*: 1 means PNG for colour but
zlib for depth. Getting them confused decodes silently-wrong data rather than
raising, so both are looked up by name below.

`open_sens` does a seek-only index pass first. Frames are variable length, so
there is no way to random-access frame `i` without walking the per-frame
headers -- but walking them is cheap (no decode, no pixel reads) and it hands
back every pose up front. That matters because ScanNet leaves BundleFusion's
tracking failures in the file as all-`-inf` poses (38 of 1037 in scene0366_00),
and those frames have to be dropped *before* deciding which ones to decode.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COLOR_COMPRESSION = {0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}

_HEADER = struct.Struct("<4I f Q")  # compressions, color wh, depth wh handled separately
_FRAME_HEADER = struct.Struct("<64s 4Q")  # cam2world bytes, 2 timestamps, 2 sizes


@dataclass
class SensHeader:
    sensor_name: str
    intrinsic_color: np.ndarray  # (4, 4)
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    color_compression: str
    depth_compression: str
    color_hw: tuple[int, int]
    depth_hw: tuple[int, int]
    depth_shift: float
    num_frames: int


class SensReader:
    """Random-access reader over one `.sens`. Use as a context manager."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        try:
            self.header = self._read_header()
            self.poses, self._offsets = self._index()
        except Exception:
            self._fh.close()
            raise

    # -- lifecycle ---------------------------------------------------------- #

    def __enter__(self) -> "SensReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    # -- header / index ----------------------------------------------------- #

    def _read_header(self) -> SensHeader:
        fh = self._fh
        (version,) = struct.unpack("<I", _read_exactly(fh, 4))
        if version != 4:
            raise ValueError(f"{self.path.name}: unsupported .sens version {version}")
        (strlen,) = struct.unpack("<Q", _read_exactly(fh, 8))
        if strlen > 1 << 16:
            raise ValueError(f"{self.path.name}: implausible sensor_name length {strlen}")
        name = _read_exactly(fh, strlen).decode("utf8", "replace").rstrip("\x00")

        mats = [np.frombuffer(_read_exactly(fh, 64), "<f4").reshape(4, 4).copy() for _ in range(4)]
        ccomp, dcomp, cw, ch, dw, dh = struct.unpack("<6I", _read_exactly(fh, 24))
        (depth_shift,) = struct.unpack("<f", _read_exactly(fh, 4))
        (num_frames,) = struct.unpack("<Q", _read_exactly(fh, 8))

        if ccomp not in COLOR_COMPRESSION:
            raise ValueError(f"{self.path.name}: unknown colour compression {ccomp}")
        if dcomp not in DEPTH_COMPRESSION:
            raise ValueError(f"{self.path.name}: unknown depth compression {dcomp}")
        if not (0 < cw <= 1 << 15 and 0 < ch <= 1 << 15 and 0 < dw <= 1 << 15 and 0 < dh <= 1 << 15):
            raise ValueError(f"{self.path.name}: implausible frame size {cw}x{ch} / {dw}x{dh}")
        if not np.isfinite(depth_shift) or depth_shift <= 0:
            raise ValueError(f"{self.path.name}: bad depth_shift {depth_shift}")

        return SensHeader(
            sensor_name=name,
            intrinsic_color=mats[0].astype(np.float64),
            extrinsic_color=mats[1].astype(np.float64),
            intrinsic_depth=mats[2].astype(np.float64),
            extrinsic_depth=mats[3].astype(np.float64),
            color_compression=COLOR_COMPRESSION[ccomp],
            depth_compression=DEPTH_COMPRESSION[dcomp],
            color_hw=(ch, cw),
            depth_hw=(dh, dw),
            depth_shift=float(depth_shift),
            num_frames=int(num_frames),
        )

    def _index(self) -> tuple[np.ndarray, np.ndarray]:
        """Walk the per-frame headers, collecting poses and payload offsets.

        Seek-only: the colour/depth payloads are skipped, not read. Returns
        `poses` as (N, 4, 4) camera-to-world and `offsets` as (N, 3) int64 rows
        of (color_offset, color_bytes, depth_bytes) -- depth immediately follows
        colour, so its offset is implied.
        """
        fh = self._fh
        n = self.header.num_frames
        poses = np.empty((n, 4, 4), dtype=np.float64)
        offsets = np.empty((n, 3), dtype=np.int64)
        for i in range(n):
            raw = fh.read(_FRAME_HEADER.size)
            if len(raw) < _FRAME_HEADER.size:
                # A truncated download leaves a partial tail; keep what parsed.
                poses, offsets = poses[:i], offsets[:i]
                self.header.num_frames = i
                break
            pose_bytes, _tc, _td, csz, dsz = _FRAME_HEADER.unpack(raw)
            poses[i] = np.frombuffer(pose_bytes, "<f4").reshape(4, 4)
            offsets[i] = (fh.tell(), csz, dsz)
            fh.seek(csz + dsz, 1)
        return poses, offsets

    # -- frame access ------------------------------------------------------- #

    @property
    def valid_poses(self) -> np.ndarray:
        """(N,) bool. False where BundleFusion lost tracking and wrote -inf.

        Also rejects a non-rigid rotation block: `scene_scale` and the
        relative-pose transform downstream both assume R is orthonormal.
        """
        ok = np.isfinite(self.poses).all(axis=(1, 2))
        R = self.poses[:, :3, :3]
        gram = np.einsum("nij,nkj->nik", R, R)
        ok &= np.abs(gram - np.eye(3)) .max(axis=(1, 2)) < 1e-3
        ok &= np.abs(np.linalg.det(np.where(ok[:, None, None], R, np.eye(3))) - 1.0) < 1e-3
        return ok

    def read_color(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8 BGR, at the header's `color_hw`."""
        payload = self._payload(i, depth=False)
        if self.header.color_compression == "raw":
            h, w = self.header.color_hw
            return np.frombuffer(payload, np.uint8).reshape(h, w, 3).copy()
        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"{self.path.name}: frame {i} colour failed to decode")
        return bgr

    def read_depth_mm(self, i: int) -> np.ndarray:
        """(H, W) uint16 raw sensor units, at the header's `depth_hw`.

        Divide by `header.depth_shift` for metres. 0 means "no return".
        """
        payload = self._payload(i, depth=True)
        mode = self.header.depth_compression
        if mode == "zlib_ushort":
            payload = zlib.decompress(payload)
        elif mode != "raw_ushort":
            raise NotImplementedError(f"{self.path.name}: depth compression {mode!r}")
        h, w = self.header.depth_hw
        expected = h * w * 2
        if len(payload) != expected:
            raise ValueError(f"{self.path.name}: frame {i} depth is {len(payload)}B, want {expected}B")
        return np.frombuffer(payload, "<u2").reshape(h, w).copy()

    def _payload(self, i: int, depth: bool) -> bytes:
        offset, csz, dsz = self._offsets[i]
        self._fh.seek(offset + csz if depth else offset)
        return _read_exactly(self._fh, int(dsz if depth else csz))


def _read_exactly(fh, n: int) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise EOFError(f"wanted {n} bytes, got {len(buf)}")
    return buf


def open_sens(path: str | Path) -> SensReader:
    return SensReader(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump a .sens header and sanity-check its frames.")
    parser.add_argument("path")
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    with open_sens(args.path) as reader:
        h = reader.header
        ok = reader.valid_poses
        print(f"{Path(args.path).name}: {h.sensor_name}  {h.num_frames} frames  {ok.sum()} with valid poses")
        print(f"  colour {h.color_hw[1]}x{h.color_hw[0]} {h.color_compression}   "
              f"depth {h.depth_hw[1]}x{h.depth_hw[0]} {h.depth_compression}  shift={h.depth_shift}")
        print(f"  K_color fx={h.intrinsic_color[0, 0]:.2f} fy={h.intrinsic_color[1, 1]:.2f} "
              f"cx={h.intrinsic_color[0, 2]:.2f} cy={h.intrinsic_color[1, 2]:.2f}")
        print(f"  K_depth fx={h.intrinsic_depth[0, 0]:.2f} fy={h.intrinsic_depth[1, 1]:.2f} "
              f"cx={h.intrinsic_depth[0, 2]:.2f} cy={h.intrinsic_depth[1, 2]:.2f}")
        print(f"  extrinsic_color identity={np.allclose(h.extrinsic_color, np.eye(4))}  "
              f"extrinsic_depth identity={np.allclose(h.extrinsic_depth, np.eye(4))}")
        for i in np.flatnonzero(ok)[: args.frames]:
            colour = reader.read_color(int(i))
            depth = reader.read_depth_mm(int(i)) / h.depth_shift
            valid = depth > 0
            print(f"  frame {i}: colour {colour.shape}  depth valid {100 * valid.mean():5.1f}%  "
                  f"range {depth[valid].min():.2f}-{depth[valid].max():.2f} m")
