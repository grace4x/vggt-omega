"""Reference loader for the preprocessed DL3DV set produced by `preprocess_dl3dv.py`.

This file is the executable spec of the on-disk format. A batch is one scene's
worth of frames, shaped the way `VGGTOmega.forward` consumes them:

    images        (S, 3, H, W)  float32 in [0, 1] -- the model applies ImageNet norm
    extrinsics    (S, 3, 4)     camera-from-world, OpenCV, first camera = identity
    intrinsics    (S, 3, 3)     pinhole for (H, W)
    pose_enc      (S, 9)        target for `predictions["pose_enc"]`
    depth         (S, H, W)     sparse GT depth, 0 where unknown
    depth_mask    (S, H, W)     bool, True where `depth` is valid
    point_map     (S, H, W, 3)  the same points in the first camera's frame
    scene_scale   ()            divisor applied to translations and depths

Sanity check:

    python training/dl3dv_dataset.py --root ~/dl3dv-train --num-frames 8
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


class DL3DVDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_frames: int = 8,
        resolution: int | None = None,
        patch_size: int = 16,
        sampling: str = "covisibility",
        min_covisibility: float = 0.1,
        augment: bool = True,
        seed: int | None = None,
        image_hw: tuple[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        index = json.loads((self.root / "index.json").read_text())
        self.scenes = [e for e in index["scenes"] if e["split"] == split]
        if not self.scenes:
            raise ValueError(f"no {split!r} scenes under {self.root}")

        # Batching requires a uniform frame count, so drop scenes that cannot
        # supply `num_frames` rather than silently returning a short sample.
        short = [e for e in self.scenes if (e.get("num_frames") or 0) < num_frames]
        if short:
            self.scenes = [e for e in self.scenes if (e.get("num_frames") or 0) >= num_frames]
            if not self.scenes:
                raise ValueError(f"no {split!r} scene has {num_frames} frames")

        # It also requires a uniform image size. `max_size` preprocessing keeps
        # the source aspect ratio, so a portrait DL3DV scene is stored (W, H)
        # swapped relative to the landscape majority and cannot stack with it.
        # Resizing does not reconcile them either -- `_resize_shape` preserves
        # aspect ratio too. Keep one shape and drop the rest.
        self.image_hw = self._select_shape(image_hw, split)

        self.num_frames = num_frames
        self.resolution = resolution
        self.patch_size = patch_size
        self.sampling = sampling
        self.min_covisibility = min_covisibility
        self.augment = augment and split == "train"
        self.seed = seed

    def __len__(self) -> int:
        return len(self.scenes)

    # -- shape filtering ---------------------------------------------------- #

    def _select_shape(self, image_hw: tuple[int, int] | None, split: str) -> tuple[int, int]:
        """Restrict `self.scenes` to a single stored (H, W) and return it."""
        by_shape: dict[tuple[int, int], list[dict]] = {}
        for entry in self.scenes:
            hw = entry.get("image_hw")
            if hw is None:  # older index.json; npz load is lazy, so this is cheap
                with np.load(self.root / entry["path"] / "meta.npz", allow_pickle=True) as meta:
                    hw = [int(v) for v in meta["image_hw"]]
                entry["image_hw"] = hw
            by_shape.setdefault((hw[0], hw[1]), []).append(entry)

        if image_hw is not None:
            target = (int(image_hw[0]), int(image_hw[1]))
            if target not in by_shape:
                raise ValueError(
                    f"no {split!r} scene stored at {target}; available: {sorted(by_shape)}"
                )
        else:
            target = max(by_shape, key=lambda hw: len(by_shape[hw]))

        if len(by_shape) > 1:
            dropped = sorted((hw, len(v)) for hw, v in by_shape.items() if hw != target)
            print(
                f"[dl3dv] {split}: keeping {len(by_shape[target])} scenes at {target}; "
                f"dropped {sum(n for _, n in dropped)} at mismatched sizes {dropped}"
            )
            self.scenes = by_shape[target]  # insertion order == original index order
        return target

    # -- frame selection ---------------------------------------------------- #

    def _sample_frames(self, covisibility: np.ndarray, rng: random.Random) -> np.ndarray:
        n = covisibility.shape[0]
        if n <= self.num_frames:
            return np.arange(n)

        if self.sampling == "random":
            return np.array(sorted(rng.sample(range(n), self.num_frames)))

        if self.sampling == "contiguous":
            # DL3DV scenes are videos: a window gives smooth, high-overlap trajectories.
            start = rng.randrange(n - self.num_frames + 1)
            return np.arange(start, start + self.num_frames)

        # Covisibility walk: seed at a random frame, then repeatedly add a frame
        # that still overlaps the set. Produces harder, wider-baseline tuples
        # than a sliding window without drifting into disjoint sub-scenes.
        chosen = [rng.randrange(n)]
        covis = covisibility.astype(np.float32)
        while len(chosen) < self.num_frames:
            score = covis[chosen].max(axis=0)
            score[chosen] = -1.0
            candidates = np.flatnonzero(score >= self.min_covisibility)
            if candidates.size == 0:
                remaining = [i for i in range(n) if i not in chosen]
                chosen.extend(rng.sample(remaining, self.num_frames - len(chosen)))
                break
            weights = score[candidates]
            chosen.append(int(rng.choices(candidates.tolist(), weights=weights.tolist(), k=1)[0]))
        return np.array(sorted(chosen))

    # -- main --------------------------------------------------------------- #

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.scenes[idx]
        scene_dir = self.root / entry["path"]
        rng = random.Random(self.seed + idx if self.seed is not None else None)

        with np.load(scene_dir / "meta.npz", allow_pickle=True) as meta:
            frame_names = [str(x) for x in meta["frame_names"]]
            extrinsics = meta["extrinsics"].astype(np.float64)
            intrinsics = meta["intrinsics"].astype(np.float64)
            src_h, src_w = (int(v) for v in meta["image_hw"])
            points_xyz = meta["points_xyz"].astype(np.float64)
            obs_frame = meta["obs_frame"]
            obs_point = meta["obs_point"]
            covisibility = meta["covisibility"]

        sel = self._sample_frames(covisibility, rng)
        if self.augment and rng.random() < 0.5:
            sel = sel[::-1].copy()  # reversed traversal order

        extrinsics = extrinsics[sel]
        intrinsics = intrinsics[sel]

        # Re-express everything relative to the first selected camera. This is the
        # frame the model predicts in: pose_enc[0] is (near-)identity by construction.
        first = _to_4x4(extrinsics[0])
        first_inv = np.linalg.inv(first)
        extrinsics = np.stack([(_to_4x4(e) @ first_inv)[:3] for e in extrinsics])
        points_first = points_xyz @ first[:3, :3].T + first[:3, 3]

        images, scales = [], []
        for i in sel:
            image = Image.open(scene_dir / "images" / frame_names[i]).convert("RGB")
            if self.resolution is not None:
                out_h, out_w = _resize_shape(image.size[1], image.size[0], self.resolution, self.patch_size)
                if (image.size[1], image.size[0]) != (out_h, out_w):
                    image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
            images.append(torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0))
            scales.append((image.size[1] / src_h, image.size[0] / src_w))

        out_h, out_w = images[0].shape[-2:]
        sy, sx = scales[0]
        intrinsics[:, 0, :] *= sx
        intrinsics[:, 1, :] *= sy

        # Sparse depth: project the tracked points into each selected frame.
        frame_lookup = {int(f): k for k, f in enumerate(sel)}
        keep = np.isin(obs_frame, sel)
        obs_f = np.array([frame_lookup[int(f)] for f in obs_frame[keep]], dtype=np.int64)
        obs_p = obs_point[keep].astype(np.int64)

        depth = np.zeros((len(sel), out_h, out_w), dtype=np.float32)
        point_map = np.zeros((len(sel), out_h, out_w, 3), dtype=np.float32)
        mask = np.zeros((len(sel), out_h, out_w), dtype=bool)

        if obs_f.size:
            R = extrinsics[obs_f, :, :3]
            t = extrinsics[obs_f, :, 3]
            xyz_first = points_first[obs_p]
            cam = np.einsum("nij,nj->ni", R, xyz_first) + t
            z = cam[:, 2]
            valid = z > 1e-6
            fx = intrinsics[obs_f, 0, 0]
            fy = intrinsics[obs_f, 1, 1]
            cx = intrinsics[obs_f, 0, 2]
            cy = intrinsics[obs_f, 1, 2]
            u = np.rint(cam[:, 0] / np.where(valid, z, 1.0) * fx + cx).astype(np.int64)
            v = np.rint(cam[:, 1] / np.where(valid, z, 1.0) * fy + cy).astype(np.int64)
            valid &= (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)

            f_i, v_i, u_i = obs_f[valid], v[valid], u[valid]
            # Nearest surface wins where two points land on the same pixel.
            order = np.argsort(-z[valid])
            depth[f_i[order], v_i[order], u_i[order]] = z[valid][order]
            point_map[f_i[order], v_i[order], u_i[order]] = xyz_first[valid][order]
            mask[f_i, v_i, u_i] = True

        # Scale normalisation: COLMAP scenes have arbitrary units, so put the
        # median observed depth at 1. Do this *after* the relative transform.
        observed = depth[mask]
        scene_scale = float(np.median(observed)) if observed.size else 1.0
        if not np.isfinite(scene_scale) or scene_scale <= 1e-8:
            scene_scale = 1.0
        depth /= scene_scale
        point_map /= scene_scale
        points_first = points_first / scene_scale
        extrinsics[:, :, 3] /= scene_scale

        extrinsics_t = torch.from_numpy(extrinsics).float()
        intrinsics_t = torch.from_numpy(intrinsics).float()
        pose_enc = extri_intri_to_pose_encoding(
            extrinsics_t[None], intrinsics_t[None], (out_h, out_w)
        )[0]

        return {
            "images": torch.stack(images),
            "extrinsics": extrinsics_t,
            "intrinsics": intrinsics_t,
            "pose_enc": pose_enc,
            "depth": torch.from_numpy(depth),
            "depth_mask": torch.from_numpy(mask),
            "point_map": torch.from_numpy(point_map),
            "scene_scale": torch.tensor(scene_scale),
            "scene_id": entry["scene"],
        }


def _to_4x4(extrinsic: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3] = extrinsic
    return out


def _resize_shape(height: int, width: int, resolution: int, patch_size: int) -> tuple[int, int]:
    """`max_size` sizing, matching `vggt_omega.utils.load_fn`."""
    aspect_ratio = height / max(width, 1)
    if aspect_ratio >= 1.0:
        out_h = resolution
        out_w = max(patch_size, int(round(resolution / aspect_ratio / patch_size)) * patch_size)
    else:
        out_w = resolution
        out_h = max(patch_size, int(round(resolution * aspect_ratio / patch_size)) * patch_size)
    return out_h, out_w


def collate_scenes(batch: list[dict]) -> dict:
    """Stack scenes into (B, S, ...). Requires a uniform S and image size."""
    shapes = {tuple(b["images"].shape) for b in batch}
    if len(shapes) > 1:
        ids = {tuple(b["images"].shape): b["scene_id"] for b in batch}
        raise ValueError(
            "cannot batch scenes with different image shapes: "
            + ", ".join(f"{s} ({ids[s][:12]})" for s in sorted(shapes))
            + " -- pass image_hw= to DL3DVDataset, or use batch_size=1"
        )

    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    dataset = DL3DVDataset(
        args.root, split=args.split, num_frames=args.num_frames, resolution=args.resolution, seed=0
    )
    print(f"{len(dataset)} {args.split} scenes")
    for i in range(min(args.n, len(dataset))):
        sample = dataset[i]
        coverage = sample["depth_mask"].float().mean().item()
        print(
            f"{sample['scene_id'][:12]}  images={tuple(sample['images'].shape)}  "
            f"depth_valid={coverage * 100:.2f}%  "
            f"depth[median]={sample['depth'][sample['depth_mask']].median():.3f}  "
            f"fov_deg={torch.rad2deg(sample['pose_enc'][0, 7:]).tolist()}  "
            f"|t| max={sample['extrinsics'][:, :, 3].norm(dim=-1).max():.2f}"
        )
