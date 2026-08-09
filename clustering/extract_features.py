"""N DINOv3 embeddings per training scene, for clustering.

Same shape as the clustering-imgs script it is adapted from: take random frames of
every scene, push them through DINOv3, keep the CLS tokens. The input here is the
preprocessed training set (`preprocess_dl3dv.py`'s output) rather than raw DL3DV,
so the scene list is exactly what `DL3DVDataset(split="train", dense_only=True)`
iterates over and the frames are already at the resolution the model trains on.

    ~/clustering-imgs/.venv/bin/python clustering/extract_features.py

(that venv, not the repo's: it is the one with `transformers` installed.)
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

# ---- inputs ----
HERE = Path(__file__).parent
DATA = Path("~/dl3dv-train").expanduser()  # preprocess_dl3dv.py output
SPLIT = "train"
MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
OUT = HERE / "dinov3_features.npz"
H, W = 224, 384  # frames are stored 384x224 by preprocess_dl3dv.py; already a multiple of 16
# Frames embedded per scene: cluster.py averages their pairwise cosine similarities into
# a single edge weight per scene pair, so a bigger N_IMAGES gives smoother/more robust
# edges at the cost of ~N_IMAGES x more DINOv3 forward passes (already the dominant cost
# of this script). Sweeping this constant means re-running the extraction pass.
N_IMAGES = 4
SEED = 0  # for reproducible frame sampling
# Per-patch tokens are 336 * 4096 * 2 B = 2.75 MB per frame, i.e. ~13 GB * N_IMAGES over
# the training set, and nothing downstream reads them -- cluster.py uses `cls` alone.
SAVE_PATCHES = False
# Match `train.py --dense-only`: the ~12% of scenes with no DA3 map supply ~1%-coverage
# COLMAP depth, so clustering them would put scenes training never loads into the pool
# a sampler draws from.
DENSE_ONLY = True

processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
patch = model.config.patch_size
skip = 1 + model.config.num_register_tokens  # cls + register tokens precede the patches

index = json.loads((DATA / "index.json").read_text())
scenes = [e for e in index["scenes"] if e["split"] == SPLIT]
if DENSE_ONLY:
    dense = [e for e in scenes if e.get("has_depth")]
    print(f"{len(dense)}/{len(scenes)} {SPLIT} scenes have dense depth; dropped the rest")
    scenes = dense

# N_IMAGES random frames per scene, seeded for reproducibility. On the rare scene with
# fewer than N_IMAGES frames, fall back to every frame it has and pad by resampling with
# replacement, so every scene still contributes the same count downstream.
rng = random.Random(SEED)
frames, short = [], 0
for e in scenes:
    all_frames = sorted((DATA / e["path"] / "images").glob("*.jpg"))
    if len(all_frames) >= N_IMAGES:
        picked = rng.sample(all_frames, N_IMAGES)
    else:
        short += 1
        picked = all_frames + [rng.choice(all_frames) for _ in range(N_IMAGES - len(all_frames))]
    frames.append(picked)
if short:
    print(f"{short}/{len(scenes)} scenes have fewer than N_IMAGES={N_IMAGES} frames; padded by resampling")

# ---- inference ----
cls, mean_patch, patches, sizes = [], [], [], []
for i, scene_frames in enumerate(frames):
    scene_cls, scene_mean_patch, scene_patches = [], [], []
    for frame in scene_frames:
        image = processor(load_image(str(frame)), size={"height": H, "width": W},
                          do_center_crop=False, return_tensors="pt")
        with torch.inference_mode():
            tokens = model(**image.to("cuda", torch.float16)).last_hidden_state[0]
        grid = tokens[skip:].reshape(H // patch, W // patch, -1)

        scene_cls.append(tokens[0].float().cpu().numpy())
        scene_mean_patch.append(grid.mean((0, 1)).float().cpu().numpy())
        if SAVE_PATCHES:
            scene_patches.append(grid.cpu().numpy())

    cls.append(np.stack(scene_cls))                # (N_IMAGES, dim)
    mean_patch.append(np.stack(scene_mean_patch))  # (N_IMAGES, dim)
    if SAVE_PATCHES:
        patches.append(np.stack(scene_patches))
    # Stored size, not H/W: the portrait scenes are the ones DL3DVDataset drops for
    # being unbatchable, so recording it lets cluster.py exclude them too.
    with np.load(DATA / scenes[i]["path"] / "meta.npz", allow_pickle=True) as meta:
        sizes.append([int(v) for v in meta["image_hw"]])
    print(f"{i + 1}/{len(frames)} {scenes[i]['subset']}/{scenes[i]['scene']}", flush=True)

# ---- outputs ----
np.savez(
    OUT,
    cls=np.stack(cls).astype(np.float16),                # (N, N_IMAGES, dim)
    mean_patch=np.stack(mean_patch).astype(np.float16),  # (N, N_IMAGES, dim)
    scenes=[e["scene"] for e in scenes],
    subsets=[e["subset"] for e in scenes],
    frames=[[str(f) for f in fs] for fs in frames],      # (N, N_IMAGES)
    image_hw=np.array(sizes),                            # (N, 2), stored (H, W)
    n_images=np.array(N_IMAGES),
    # Every scene gets an embedding regardless; cluster.py decides whether to use
    # the sparse-depth ones, so flipping that choice costs a re-cluster, not a re-extract.
    has_depth=np.array([bool(e.get("has_depth")) for e in scenes]),
    **({"patches": np.stack(patches).astype(np.float16)} if SAVE_PATCHES else {}),
)
print(f"wrote {OUT} ({len(scenes)} {SPLIT} scenes, {N_IMAGES} frames/scene)")
