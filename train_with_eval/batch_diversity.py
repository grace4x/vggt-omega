"""Cross-scene geometric diversity via DINOv3 patch-token descriptors.

One descriptor per scene (mean of per-frame patch-token means at layer 30, matching
`patch_avg_clustering/extract_features.py`). Pairwise cosine similarity is computed
**across scenes in the batch only** -- not across multi-view frames within a scene.

By default each selected layer's tokens go through the model's final layernorm
before the patch mean -- `hidden_states[l]` is otherwise pre-norm, and only
`last_hidden_state` has it applied. That matches how DINOv3 intermediate
features are normally consumed (`apply_layernorm=True`). Layernorm is per token
and not linear, so it has to happen at extract/forward time, not on a stored
patch mean.

Prefer `BatchDiversityTracker.from_npz` (see `extract_features.py`) over a live
ViT-7B forward. Requires batch_size >= 2; use --batch-size 8 (or similar).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
DEFAULT_LAYERS = (30,)
DEFAULT_FINAL_LN = True


def scene_key(dataset: str, subset: str, scene: str) -> str:
    """Lookup key written by `extract_features.py` and read from a training batch."""
    return f"{dataset}/{subset}/{scene}"


def _empty_diversity(n_scenes: int) -> dict[str, float]:
    return {
        "n_scenes": float(n_scenes),
        "n_pairs": 0.0,
        "avg_cos_sim": float("nan"),
        "diversity": float("nan"),
        "cos_sim_std": float("nan"),
    }


def _pairwise_diversity(scene_desc: torch.Tensor) -> dict[str, float]:
    b = scene_desc.shape[0]
    if b < 2:
        return _empty_diversity(b)
    pairs = [scene_desc[i].dot(scene_desc[j]).item() for i, j in itertools.combinations(range(b), 2)]
    arr = np.array(pairs, dtype=np.float64)
    avg = float(arr.mean())
    return {
        "n_scenes": float(b),
        "n_pairs": float(len(pairs)),
        "avg_cos_sim": avg,
        "diversity": 1.0 - avg,
        "cos_sim_std": float(arr.std()) if len(pairs) > 1 else 0.0,
    }


def load_feature_cache(paths: Sequence[str | Path]) -> tuple[dict[str, np.ndarray], dict]:
    """Merge one or more extract_features.py npz files into key -> unit scene vector."""
    cache: dict[str, np.ndarray] = {}
    meta: dict | None = None
    for path in paths:
        path = Path(path)
        with np.load(path, allow_pickle=True) as z:
            recorded = {
                "model": str(z["model"]),
                "layers": tuple(int(v) for v in z["patch_layers"]),
                "n_images": int(z["n_images"]),
                "final_ln": bool(z["final_ln"]),
            }
            if meta is None:
                meta = {**recorded, "paths": [str(path)]}
            elif {k: recorded[k] for k in ("model", "layers", "n_images", "final_ln")} != {
                k: meta[k] for k in ("model", "layers", "n_images", "final_ln")
            }:
                raise ValueError(f"{path} recipe {recorded} does not match {meta}")
            else:
                meta["paths"].append(str(path))

            if "scene" in z.files:
                vecs = z["scene"].astype(np.float32)
            else:
                patch = z["patch"].astype(np.float32)
                vecs = patch.mean(axis=1)
                norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
                vecs = vecs / np.clip(norms, 1e-12, None)

            keys = z["keys"] if "keys" in z.files else [
                scene_key(str(d), str(s), str(sc))
                for d, s, sc in zip(z["datasets"], z["subsets"], z["scenes"])
            ]
            for key, vec in zip(keys, vecs):
                cache[str(key)] = vec
    if meta is None:
        raise ValueError("no feature npz files given")
    meta["n_scenes"] = len(cache)
    return cache, meta


class BatchDiversityTracker:
    """Frozen DINOv3, or a precomputed npz cache, for cross-scene diversity."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        layers: Sequence[int] = DEFAULT_LAYERS,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        max_frames: int = 4,
        final_ln: bool = DEFAULT_FINAL_LN,
        cache: dict[str, np.ndarray] | None = None,
        cache_meta: dict | None = None,
    ) -> None:
        self.model_id = model_id
        self.layers = tuple(layers)
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_frames = max_frames
        self.final_ln = final_ln
        self._cache = cache
        self.cache_meta = cache_meta
        self._processor = None
        self._model = None
        self._skip = None
        self._missing: set[str] = set()

    @classmethod
    def from_npz(
        cls,
        paths: Sequence[str | Path],
        *,
        layers: Sequence[int] = DEFAULT_LAYERS,
    ) -> "BatchDiversityTracker":
        cache, meta = load_feature_cache(paths)
        if tuple(layers) != meta["layers"]:
            raise ValueError(
                f"requested layers {tuple(layers)} but npz has {meta['layers']}"
            )
        return cls(
            model_id=meta["model"],
            layers=meta["layers"],
            cache=cache,
            cache_meta=meta,
            final_ln=bool(meta["final_ln"]),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()
        self._skip = 1 + self._model.config.num_register_tokens

    @torch.inference_mode()
    def frame_descriptors(self, images: torch.Tensor) -> torch.Tensor:
        """Patch-mean descriptors for frames in [0, 1], shape (N, D).

        `images` is (N, 3, H, W) float32 in [0, 1].
        """
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None and self._skip is not None

        n_layers = self._model.config.num_hidden_layers
        for layer in self.layers:
            if not (-n_layers <= layer <= n_layers):
                raise ValueError(f"layer {layer} out of range for {self.model_id} (depth {n_layers})")

        mean = torch.tensor(self._processor.image_mean, device=images.device, dtype=torch.float32)
        std = torch.tensor(self._processor.image_std, device=images.device, dtype=torch.float32)
        pixel_values = ((images - mean[:, None, None]) / std[:, None, None]).to(
            device=self.device, dtype=self.dtype
        )

        out_ = self._model(pixel_values=pixel_values, output_hidden_states=True)
        vecs = []
        for layer_idx in self.layers:
            hidden = out_.hidden_states[layer_idx]  # (N, 1+reg+patches, dim)
            if self.final_ln:
                hidden = self._model.norm(hidden)
            patch_mean = hidden[:, self._skip :, :].mean(dim=1).float()
            patch_mean = F.normalize(patch_mean, dim=-1)
            vecs.append(patch_mean)
        return torch.cat(vecs, dim=-1)  # (N, len(layers)*dim)

    def _scene_descriptor(self, frames: torch.Tensor) -> torch.Tensor:
        """One L2-normalized descriptor for a scene's frames, shape (D,)."""
        n = frames.shape[0]
        if n > self.max_frames:
            idx = torch.linspace(0, n - 1, self.max_frames).round().long()
            frames = frames[idx]
        frame_desc = self.frame_descriptors(frames.cpu())
        scene_desc = F.normalize(frame_desc.mean(dim=0), dim=0)
        return scene_desc

    @torch.inference_mode()
    def from_batch_images(self, images: torch.Tensor) -> dict[str, float]:
        """Cross-scene diversity from the batch's actual frames (live DINOv3).

        `images` is (B, S, 3, H, W). Builds one descriptor per scene, then mean
        pairwise cosine similarity across the B scenes (not within-scene views).
        """
        b = images.shape[0]
        if b < 2:
            return _empty_diversity(b)
        scene_desc = torch.stack([self._scene_descriptor(images[i]) for i in range(b)])
        return _pairwise_diversity(scene_desc)

    def from_batch_ids(
        self,
        datasets: Sequence[str],
        subsets: Sequence[str],
        scene_ids: Sequence[str],
    ) -> dict[str, float]:
        """Cross-scene diversity from precomputed scene descriptors."""
        if self._cache is None:
            raise RuntimeError("from_batch_ids requires a feature cache (use from_npz)")
        vecs = []
        for dataset, subset, scene in zip(datasets, subsets, scene_ids):
            key = scene_key(dataset, subset, scene)
            vec = self._cache.get(key)
            if vec is None:
                if key not in self._missing:
                    print(f"[diversity] no cached descriptor for {key}", flush=True)
                    self._missing.add(key)
                continue
            vecs.append(torch.from_numpy(vec))
        record = _pairwise_diversity(torch.stack(vecs) if vecs else torch.zeros(0, 1))
        record["n_missing"] = float(len(datasets) - len(vecs))
        return record

    def from_batch(self, batch: dict) -> dict[str, float]:
        """Dispatch to the npz cache when present, otherwise a live DINOv3 forward."""
        if self._cache is not None:
            return self.from_batch_ids(batch["dataset"], batch["subset"], batch["scene_id"])
        return self.from_batch_images(batch["images"])


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r between two equal-length finite sequences."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
