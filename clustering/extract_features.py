"""N DINOv3 embeddings per training scene, for clustering.

Same shape as the clustering-imgs script it is adapted from: take random frames of
every scene, push them through DINOv3, keep the CLS tokens -- concatenated across a
few layers (see CLS_LAYERS) rather than the final layer alone. The input here is the
preprocessed training set (`preprocess_dl3dv.py`'s output) rather than raw DL3DV,
so the scene list is exactly what `DL3DVDataset(split="train", dense_only=True)`
iterates over and the frames are already at the resolution the model trains on.

    ~/clustering-imgs/.venv/bin/python clustering/extract_features.py [--layers 10 20 30 40]

(that venv, not the repo's: it is the one with `transformers` installed.)
"""

import argparse
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
# This directory is a held-out set preprocessed on its own, so every scene in it
# is eval data regardless of the train/val label `preprocess_dl3dv.py` assigned
# (see the matching comment in `DL3DVDataset`) -- "all" here means "everything in
# this index", not the training split.
SPLIT = "train"
MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
OUT = HERE / "6k_features_layers.npz"
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
# `cls` is built from these hidden-state indices (1 = after the first transformer block,
# model.config.num_hidden_layers = after the last -- 40 for vit7b16) instead of just the
# final layer: DINOv3's earlier/middle layers carry more low-level appearance, later
# layers more semantic content, so concatenating a few spans a broader notion of "similar"
# for cluster.py's cosine similarity than the final layer alone. Each layer's CLS vector
# is L2-normalized before concatenating -- raw CLS norms grow sharply with depth (single
# digits by layer ~20, tens by layer 40), so without that the deepest layer would dominate
# the combined vector and cosine similarity would collapse back to ~last-layer-only.
# Spaced evenly across the full 40-layer depth (not just the first quarters of it -- (4,
# 11, 17, 23) is the quarter/half/3-quarter/last spacing of a 24-layer ViT, e.g. ViT-L/14,
# not this model's 40). Override with --layers, e.g. `--layers -1` for last-layer-only.
CLS_LAYERS = (10, 20, 30, 40)
parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+", default=list(CLS_LAYERS), metavar="L",
                     help="hidden-state indices (1..num_hidden_layers, negative counts from "
                          "the end) whose CLS tokens get concatenated (default: %(default)s)")
# `out.hidden_states[l]` is the block-l output *before* the model's final layernorm; only
# `last_hidden_state` has it applied. --final-ln applies that layernorm to every selected
# layer, which is how DINOv3 intermediate features are normally consumed (HF's
# DINOv3ViTBackbone norms every requested stage -- `apply_layernorm` defaults True -- as does
# DINOv2's `get_intermediate_layers(norm=True)`). All-or-nothing on purpose: norming only
# some selected layers would make a layer's vector depend on what else was selected, so
# `--layers -1` would stop being a clean ablation of layer 40's slice of a multi-layer run.
parser.add_argument("--final-ln", action="store_true",
                     help="apply the model's final layernorm to every selected layer's CLS "
                          "before L2-normalizing it (default: raw hidden states)")
args = parser.parse_args()
CLS_LAYERS = tuple(args.layers)
FINAL_LN = args.final_ln
# Match `train.py --dense-only`: the ~12% of scenes with no DA3 map supply ~1%-coverage
# COLMAP depth, so clustering them would put scenes training never loads into the pool
# a sampler draws from.
DENSE_ONLY = True

processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
patch = model.config.patch_size
skip = 1 + model.config.num_register_tokens  # cls + register tokens precede the patches
n_layers = model.config.num_hidden_layers
assert all(-n_layers <= l <= n_layers for l in CLS_LAYERS), f"--layers must be in [-{n_layers}, {n_layers}]"
print(f"aggregating CLS from layers {CLS_LAYERS} of {n_layers} total"
      + (" (post-final-layernorm)" if FINAL_LN else ""))

index = json.loads((DATA / "index.json").read_text())
scenes = [e for e in index["scenes"] if SPLIT == "all" or e["split"] == SPLIT]
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
            out = model(**image.to("cuda", torch.float16), output_hidden_states=True)
        tokens = out.last_hidden_state[0]  # final layer, post-final-layernorm; patches unaffected by CLS_LAYERS
        grid = tokens[skip:].reshape(H // patch, W // patch, -1)

        # out.hidden_states[l][0, 0] is the CLS token as it stood after transformer block l,
        # pre-final-layernorm; under --final-ln each one is pushed through `model.norm` (for
        # the last layer that reproduces `tokens[0]` exactly). L2-normalize each before
        # concatenating so every chosen layer contributes equally regardless of its raw
        # activation scale.
        with torch.inference_mode():
            layer_cls = [out.hidden_states[l][0, 0] for l in CLS_LAYERS]
            if FINAL_LN:
                layer_cls = [model.norm(v) for v in layer_cls]
        layer_cls = [v.float().cpu().numpy() for v in layer_cls]
        scene_cls.append(np.concatenate([v / np.linalg.norm(v) for v in layer_cls]))
        scene_mean_patch.append(grid.mean((0, 1)).float().cpu().numpy())
        if SAVE_PATCHES:
            scene_patches.append(grid.cpu().numpy())

    cls.append(np.stack(scene_cls))                # (N_IMAGES, len(CLS_LAYERS) * dim)
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
    cls=np.stack(cls).astype(np.float16),                # (N, N_IMAGES, len(CLS_LAYERS) * dim)
    mean_patch=np.stack(mean_patch).astype(np.float16),  # (N, N_IMAGES, dim)
    scenes=[e["scene"] for e in scenes],
    subsets=[e["subset"] for e in scenes],
    frames=[[str(f) for f in fs] for fs in frames],      # (N, N_IMAGES)
    image_hw=np.array(sizes),                            # (N, 2), stored (H, W)
    n_images=np.array(N_IMAGES),
    cls_layers=np.array(CLS_LAYERS),
    # cls_layers alone no longer identifies the features, so record the flag too.
    final_ln=np.array(FINAL_LN),
    # The final layernorm's learned affine, so cluster.py can apply the layernorm to an
    # extraction run made *without* --final-ln instead of costing a re-extract. That works
    # because LN's per-token centering/rescaling is recoverable from the L2-normalized CLS
    # this script saves: z = (u - u.mean()) / u.std() for the stored unit vector u equals
    # (x - x.mean()) / x.std() for the raw token x (L2-normalizing is a positive scalar
    # multiple, which cancels), so ln_weight * z + ln_bias reconstructs `model.norm(x)` to
    # ~1e-7 cosine -- the only loss is float16 storage and the ignored eps=1e-5 term.
    ln_weight=model.norm.weight.detach().float().cpu().numpy(),
    ln_bias=model.norm.bias.detach().float().cpu().numpy(),
    # Every scene gets an embedding regardless; cluster.py decides whether to use
    # the sparse-depth ones, so flipping that choice costs a re-cluster, not a re-extract.
    has_depth=np.array([bool(e.get("has_depth")) for e in scenes]),
    **({"patches": np.stack(patches).astype(np.float16)} if SAVE_PATCHES else {}),
)
print(f"wrote {OUT} ({len(scenes)} {SPLIT} scenes, {N_IMAGES} frames/scene, "
      f"layers {CLS_LAYERS}{', final-ln' if FINAL_LN else ''})")
