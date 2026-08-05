# Minimal, dependency-free readers for COLMAP's binary sparse model format.
#
# Only the fields VGGT-Omega training needs are kept: intrinsics, camera-from-world
# poses, 3D points and the image<->point track associations.

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


# (model_id, model_name, num_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def to_pinhole(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (K, dist) with dist in OpenCV's (k1, k2, p1, p2[, k3]) layout."""
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
            dist = np.zeros(5)
        elif self.model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist = np.zeros(5)
        elif self.model == "SIMPLE_RADIAL":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
            dist = np.array([p[3], 0.0, 0.0, 0.0, 0.0])
        elif self.model == "RADIAL":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
            dist = np.array([p[3], p[4], 0.0, 0.0, 0.0])
        elif self.model == "OPENCV":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist = np.array([p[4], p[5], p[6], p[7], 0.0])
        elif self.model == "FULL_OPENCV":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist = np.array([p[4], p[5], p[6], p[7], p[8]])
        else:
            raise ValueError(f"Unsupported COLMAP camera model for pinhole conversion: {self.model}")

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, dist.astype(np.float64)


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w, x, y, z), world -> camera
    tvec: np.ndarray  # world -> camera
    camera_id: int
    name: str
    xys: np.ndarray  # (n, 2) keypoints in the *distorted* full-resolution image
    point3D_ids: np.ndarray  # (n,), -1 where the keypoint has no triangulated point

    def cam_from_world(self) -> np.ndarray:
        """3x4 camera-from-world matrix in OpenCV convention (+x right, +y down, +z forward)."""
        R = qvec_to_rotmat(self.qvec)
        return np.concatenate([R, self.tvec[:, None]], axis=1)


@dataclass
class Points3D:
    ids: np.ndarray  # (m,)
    xyz: np.ndarray  # (m, 3)
    rgb: np.ndarray  # (m, 3) uint8
    error: np.ndarray  # (m,) mean reprojection error in full-res pixels
    track_length: np.ndarray  # (m,)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _read(fid, fmt: str):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, fid.read(size))


def read_cameras_bin(path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as fid:
        (num_cameras,) = _read(fid, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(fid, "iiQQ")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.array(_read(fid, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model_name, int(width), int(height), params)
    return cameras


def read_images_bin(path, load_keypoints: bool = True) -> dict[int, Image]:
    images = {}
    with open(path, "rb") as fid:
        (num_images,) = _read(fid, "Q")
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(fid, "idddddddi")
            name_chars = []
            while True:
                c = fid.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")

            (num_points2D,) = _read(fid, "Q")
            blob = fid.read(24 * num_points2D)
            if load_keypoints and num_points2D:
                raw = np.frombuffer(blob, dtype=np.dtype([("xy", "<f8", 2), ("pid", "<i8")]), count=num_points2D)
                xys = np.array(raw["xy"], dtype=np.float64)
                pids = np.array(raw["pid"], dtype=np.int64)
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                pids = np.zeros((0,), dtype=np.int64)

            images[image_id] = Image(
                id=int(image_id),
                qvec=np.array([qw, qx, qy, qz], dtype=np.float64),
                tvec=np.array([tx, ty, tz], dtype=np.float64),
                camera_id=int(camera_id),
                name=name,
                xys=xys,
                point3D_ids=pids,
            )
    return images


def read_points3D_bin(path) -> Points3D:
    ids, xyz, rgb, error, track_length = [], [], [], [], []
    with open(path, "rb") as fid:
        (num_points,) = _read(fid, "Q")
        for _ in range(num_points):
            pid, x, y, z, r, g, b, err = _read(fid, "QdddBBBd")
            (track_len,) = _read(fid, "Q")
            fid.read(8 * track_len)  # (image_id, point2D_idx) pairs, recoverable from images.bin
            ids.append(pid)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            error.append(err)
            track_length.append(track_len)

    return Points3D(
        ids=np.array(ids, dtype=np.int64),
        xyz=np.array(xyz, dtype=np.float64).reshape(-1, 3),
        rgb=np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        error=np.array(error, dtype=np.float64),
        track_length=np.array(track_length, dtype=np.int32),
    )


def read_model(sparse_dir) -> tuple[dict[int, Camera], dict[int, Image], Points3D]:
    from pathlib import Path

    sparse_dir = Path(sparse_dir)
    return (
        read_cameras_bin(sparse_dir / "cameras.bin"),
        read_images_bin(sparse_dir / "images.bin"),
        read_points3D_bin(sparse_dir / "points3D.bin"),
    )
