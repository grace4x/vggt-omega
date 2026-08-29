"""DINOv3 embeddings per training scene, for one dataset at a time.

Same pipeline and same npz schema as `multi_clustering/extract_features.py`, with one
difference: a layer's scene descriptor is the **mean of that layer's patch tokens**
instead of its CLS token. CLS is trained to summarize an image into a single global
vector, so it leans on whatever DINOv3 found most discriminative and can ignore the
rest of the frame; averaging the patch grid weights every spatial location equally, so
the descriptor tracks "what is in this room / on this street, and how much of it" more
than "what is this image about". For clustering scenes -- where the thing being matched
is the overall content of a capture, not a salient object -- that is usually the more
faithful notion of similarity, and it is the reason this folder exists next to
`multi_clustering/`.

Everything else is deliberately identical, so the two runs are comparable: the same
model, the same per-scene-seeded frame sampling, the same per-layer L2-normalize before
concatenation, and the same refusal (in `cluster.py`) to pool npz files whose
(model, layers, final_ln, n_images) disagree.

    .venv/bin/python patch_avg_clustering/extract_features.py --dataset scannet
    .venv/bin/python patch_avg_clustering/extract_features.py --dataset dl3dv

Unlike `multi_clustering/`, *both* datasets need extracting here: the npz that
`clustering/` already wrote holds CLS features, so there is no cache to reuse.

The `cls` array is written too -- it is free, the forward pass is the whole cost -- so a
consumer can compare the two descriptors on identical frames without re-running this.
`cluster.py` reads `patch`.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, however this is invoked

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

from scene_sets import DEFAULTS, add_source_args, key, resolve, sources_from_args

MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
# Frames embedded per scene: cluster.py averages their pairwise cosine similarities
# into a single edge weight per scene pair, so a bigger N_IMAGES gives smoother
# edges at ~N_IMAGES x the DINOv3 forward passes (the dominant cost here). Must
# match the other npz files being pooled.
N_IMAGES = 4
SEED = 0
# Hidden-state indices (1 = after the first transformer block, 40 = after the last
# for vit7b16) whose patch-token means get concatenated: DINOv3's earlier layers carry
# more low-level appearance, later ones more semantic content, so a few spread evenly
# over the depth span a broader notion of "similar" than the final layer alone. Each
# layer's mean is L2-normalized before concatenation because raw token norms grow
# sharply with depth, and without that the deepest layer would dominate the combined
# vector.
PATCH_LAYERS = (10, 20, 30, 40)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", choices=tuple(DEFAULTS), required=True,
               help="which preprocessed source to embed")
p.add_argument("--out", type=Path, default=None, metavar="NPZ",
               help="output npz (default: the source's --<name>-features path)")
p.add_argument("--model", default=MODEL)
p.add_argument("--n-images", type=int, default=N_IMAGES, metavar="N",
               help="frames embedded per scene (default: %(default)s)")
p.add_argument("--layers", type=int, nargs="+", default=list(PATCH_LAYERS), metavar="L",
               help="hidden-state indices whose patch-token means are concatenated "
                    "(negative counts from the end; default: %(default)s)")
# `out.hidden_states[l]` is block l's output *before* the model's final layernorm; only
# `last_hidden_state` has it applied. --final-ln applies that layernorm to every selected
# layer, which is how DINOv3 intermediate features are normally consumed. It is applied
# per token, before the mean over the grid -- layernorm is not linear, so averaging first
# would not give the same vector. All-or-nothing on purpose, so a layer's vector never
# depends on what else was selected.
p.add_argument("--final-ln", action="store_true",
               help="apply the final layernorm to every selected layer's patch tokens "
                    "before averaging them")
p.add_argument("--seed", type=int, default=SEED, help="frame-sampling seed")
p.add_argument("--overwrite", action="store_true",
               help="allow replacing an existing npz (refused otherwise)")
add_source_args(p)
args = p.parse_args()

(source,) = sources_from_args(args, names=[args.dataset])
out = args.out or source.features
if out.exists() and not args.overwrite:
    raise SystemExit(f"{out} exists; pass --overwrite to replace it")

scenes, (H, W) = resolve(source, args.split, args.num_frames)
if H % 16 or W % 16:
    raise SystemExit(f"stored shape {(H, W)} is not a multiple of the patch size")

# ---- frame choice: N_IMAGES per scene, seeded by scene so it is cap-independent ----
# A scene with fewer than N_IMAGES frames takes all of them and pads by resampling with
# replacement, so every scene contributes the same count downstream. (`--num-frames 16`
# already drops the short scenes, so this is a safety net rather than a common path.)
frames, short = [], 0
for entry in scenes:
    rng = random.Random(f"{args.seed}:{key(entry)}")
    available = sorted((source.root / entry["path"] / "images").glob("*.jpg"))
    if not available:
        raise SystemExit(f"no frames under {source.root / entry['path'] / 'images'}")
    if len(available) >= args.n_images:
        frames.append(rng.sample(available, args.n_images))
    else:
        short += 1
        frames.append(available + [rng.choice(available) for _ in range(args.n_images - len(available))])
if short:
    print(f"{short}/{len(scenes)} scenes have fewer than {args.n_images} frames; padded by resampling")

# ---- model ----
processor = AutoImageProcessor.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model, dtype=torch.float16).cuda().eval()
skip = 1 + model.config.num_register_tokens  # cls + register tokens precede the patches
n_layers = model.config.num_hidden_layers
layers = tuple(args.layers)
if not all(-n_layers <= l <= n_layers for l in layers):
    raise SystemExit(f"--layers must be in [-{n_layers}, {n_layers}]")
print(f"[{source.name}] {len(scenes)} scenes x {args.n_images} frames at {(H, W)}, "
      f"mean patch tokens from layers {layers} of {n_layers}"
      + (" (post-final-layernorm)" if args.final_ln else ""))

# ---- inference ----
patch_avg, cls = [], []
for i, scene_frames in enumerate(frames):
    scene_patch, scene_cls = [], []
    for frame in scene_frames:
        image = processor(load_image(str(frame)), size={"height": H, "width": W},
                          do_center_crop=False, return_tensors="pt")
        with torch.inference_mode():
            out_ = model(**image.to("cuda", torch.float16), output_hidden_states=True)
            hidden = [out_.hidden_states[l][0] for l in layers]  # (1 + registers + patches, dim)
            if args.final_ln:
                hidden = [model.norm(h) for h in hidden]
            # The mean over the patch grid, register tokens excluded: they hold no image
            # content and would drag every scene's descriptor toward a shared direction.
            layer_vec = [h[skip:].mean(0).float().cpu().numpy() for h in hidden]
            layer_cls = [h[0].float().cpu().numpy() for h in hidden]
        # Each layer is L2-normalized before concatenation (see PATCH_LAYERS above).
        scene_patch.append(np.concatenate([v / np.linalg.norm(v) for v in layer_vec]))
        scene_cls.append(np.concatenate([v / np.linalg.norm(v) for v in layer_cls]))

    patch_avg.append(np.stack(scene_patch))  # (n_images, len(layers) * dim)
    cls.append(np.stack(scene_cls))          # (n_images, len(layers) * dim)
    if (i + 1) % 25 == 0 or i + 1 == len(frames):
        print(f"{i + 1}/{len(frames)} {key(scenes[i])}", flush=True)

# ---- output ----
np.savez(
    out,
    patch=np.stack(patch_avg).astype(np.float16),  # (N, n_images, len(layers) * dim)
    cls=np.stack(cls).astype(np.float16),          # same, from the CLS token; unused here
    scenes=[e["scene"] for e in scenes],
    subsets=[e["subset"] for e in scenes],
    frames=[[str(f) for f in fs] for fs in frames],      # (N, n_images)
    image_hw=np.array([[H, W]] * len(scenes)),           # one shape by construction
    n_images=np.array(args.n_images),
    patch_layers=np.array(layers),
    final_ln=np.array(args.final_ln),
    has_depth=np.array([bool(e.get("has_depth")) for e in scenes]),
    # Identifies the run, so cluster.py can refuse to pool incompatible features.
    features=np.array("patch-avg"),
    dataset=np.array(source.name),
    model=np.array(args.model),
    # The final layernorm's learned affine, kept for reference. Note that -- unlike the
    # CLS case -- it does NOT let a consumer recover --final-ln features from these: the
    # layernorm happens per token, and only the averaged grid was stored.
    ln_weight=model.norm.weight.detach().float().cpu().numpy(),
    ln_bias=model.norm.bias.detach().float().cpu().numpy(),
)
print(f"wrote {out} ({len(scenes)} {source.name} scenes, {args.n_images} frames/scene, "
      f"layers {layers}{', final-ln' if args.final_ln else ''})")
print(f"  next: .venv/bin/python patch_avg_clustering/cluster.py --k 150")
