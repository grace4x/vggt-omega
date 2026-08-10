"""Community-detect the scene embeddings, so training can sample across clusters.

Adapted from the clustering-imgs script: cosine similarity on the CLS vectors,
sparsified to a graph, partitioned by modularity. Two differences worth knowing:

* `METHOD` picks Louvain (networkx, no extra deps) or Leiden (leidenalg). Same
  objective -- RBConfiguration is resolution-gamma modularity -- but Leiden's
  refinement step guarantees connected communities, which Louvain does not.
* Each scene is now `extract_features.py`'s N_IMAGES CLS vectors, not one. The edge
  weight between two scenes is the average of all N_IMAGES x N_IMAGES pairwise cosine
  similarities between their sampled images, computed without an O(n^2) loop: that
  average equals the dot product of the two scenes' (non-renormalized) centroids of
  per-image unit vectors.

    ~/clustering-imgs/.venv/bin/python clustering/cluster.py
    ~/clustering-imgs/.venv/bin/python clustering/cluster.py \
        --features clustering/dinov3_features.npz clustering/6k_dinov3_features.npz \
        --out combined_clusters

Several `--features` npzs are concatenated into one scene pool before clustering, so eval
scenes land in the same communities as the training scenes they resemble.

Writes `<stem>.json` -- the cluster -> scene mapping the training-set sampler reads -- plus
`<stem>.html`, `<stem>_cosine.npy` and `<stem>_<method>_labels.npy`.
"""

import argparse
import json
import os
import numpy as np, networkx as nx
from pathlib import Path
from PIL import Image

METHOD = "leiden"  # "leiden" or "louvain"
THRESH = 0     # sparsification quantile: keep the top (1-THRESH) most-similar pairs as edges
RESOLUTION = 2.0  # resolution; >1 penalizes merging, giving more/smaller clusters
THUMB = 160       # contact-sheet thumbnail width, px

HERE = Path(__file__).parent

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--features", type=Path, nargs="+", default=[HERE / "dinov3_features_layered.npz"],
               help="extract_features.py npz(s), concatenated into one scene pool; they must "
                    "share a CLS geometry (same --layers / --final-ln) and N_IMAGES")
p.add_argument("--out", type=Path, default=None,
               help="output stem, bare names land next to this script (default: from --features)")
p.add_argument("--thumbs", type=Path, default=HERE / "thumbs",
               help="contact-sheet thumbnail dir; files are named per scene, so several "
                    "feature sets can share one dir")
args = p.parse_args()

stem = args.out or Path("+".join(f.stem.replace("dinov3_features", "clusters") for f in args.features))
if stem.parent == Path("."):
    stem = HERE / stem


def geometry(d):
    """(post_final_ln, cls_layers) for a feature npz, including ones predating the flags.

    An npz with neither key is the original extract, which read last_hidden_state CLS --
    i.e. post-final-LayerNorm -- so it combines with a `--final-ln` npz.
    """
    layers = tuple(int(l) for l in d["cls_layers"]) if "cls_layers" in d.files else ()
    ln = bool(d["post_final_ln"]) if "post_final_ln" in d.files else "cls_layers" not in d.files
    return ln, layers


ds = [np.load(f) for f in args.features]
for f, d in zip(args.features[1:], ds[1:]):
    for what, got, want in [("CLS geometry", geometry(d), geometry(ds[0])),
                            ("CLS shape", d["cls"].shape[1:], ds[0]["cls"].shape[1:]),
                            ("n_images", int(d["n_images"]), int(ds[0]["n_images"]))]:
        if got != want:
            raise SystemExit(f"{f}: {what} {got} != {want} of {args.features[0]}")
cat = lambda k: np.concatenate([d[k] for d in ds])
cls, scenes, subsets, frames = cat("cls").astype(np.float32), cat("scenes"), cat("subsets"), cat("frames")
n_images = int(ds[0]["n_images"])  # frames sampled per scene by extract_features.py
if len(ds) > 1:
    print(f"pooled {len(cls)} scenes: " + ", ".join(f"{len(d['cls'])} from {f.name}"
                                                    for f, d in zip(args.features, ds)))

# DL3DVDataset keeps a single stored (H, W) so scenes can stack into a batch, and
# drops the rest. Clustering scenes it would never load only dilutes the samples
# drawn from them, so apply the same filter here.
hw = cat("image_hw")
shapes, counts = np.unique(hw, axis=0, return_counts=True)
target = shapes[counts.argmax()]
keep_scene = (hw == target).all(1)
if not keep_scene.all():
    print(f"keeping {keep_scene.sum()} scenes at {tuple(int(v) for v in target)}; "
          f"dropped {(~keep_scene).sum()} at mismatched sizes")
    cls, scenes, subsets, frames = cls[keep_scene], scenes[keep_scene], subsets[keep_scene], frames[keep_scene]

# extract_features.py --final-ln writes post-LN last_hidden_state CLS (already discriminative;
# no DC strip). --layers writes pre-LN hidden_states CLS, optionally concatenated.
post_final_ln, cls_layers = geometry(ds[0])
n_layers = len(cls_layers)
if n_layers > 1:
    # Each layer's CLS vector, after extract_features.py's per-layer L2-normalize, turns out to be
    # ~90-99.6% (worse in earlier layers) a "DC" direction that's shared by every image regardless
    # of content -- not scene signal. Averaged into the combined cosine similarity, that near-constant
    # component drowns out the actual per-scene variation, collapsing sim into a narrow, uniformly-high
    # band with nothing for modularity to partition on (hence all-singleton clusters). Subtract each
    # layer's dataset-mean direction and re-normalize before combining layers, to strip that bias out.
    # Skipped for --final-ln / single pre-LN layer: post-LN already has low DC; single pre-LN
    # layer still has ~88% DC but we leave that path alone (use --final-ln to get the old geometry).
    layer_dims = cls.shape[-1] // n_layers
    flat = cls.reshape(-1, cls.shape[-1])
    X = np.empty_like(flat)
    for k in range(n_layers):
        seg = flat[:, k * layer_dims:(k + 1) * layer_dims]
        seg = seg - seg.mean(axis=0)
        X[:, k * layer_dims:(k + 1) * layer_dims] = seg / np.linalg.norm(seg, axis=-1, keepdims=True)
    X = X.reshape(cls.shape)
    X /= np.linalg.norm(X, axis=-1, keepdims=True)  # renormalize the full per-image vector to unit length
else:
    X = cls / np.linalg.norm(cls, axis=-1, keepdims=True)  # normalize each sampled image's CLS vector
print(f"features: post_final_ln={post_final_ln}, cls_layers={list(cls_layers) or 'n/a'}")
centroid = X.mean(axis=1)                                # (N, dim); NOT re-normalized
sim = centroid @ centroid.T  # == average of the n_images^2 pairwise cosine similarities per scene pair
np.save(stem.with_name(stem.name + "_cosine.npy"), sim)

i, j = np.triu_indices(len(cls), 1)
w = sim[i, j]
keep = (w >= np.quantile(w, THRESH)) & (w > 0)  # drop negatives: modularity's 2m needs one sign
G = nx.Graph()
G.add_nodes_from(range(len(cls)))
G.add_weighted_edges_from(zip(i[keep], j[keep], w[keep]))

if METHOD == "louvain":
    communities = nx.community.louvain_communities(G, weight="weight", resolution=RESOLUTION, seed=0)
elif METHOD == "leiden":
    import igraph as ig, leidenalg

    g = ig.Graph(n=len(cls), edges=list(zip(i[keep].tolist(), j[keep].tolist())),
                 edge_attrs={"weight": w[keep].tolist()})
    communities = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                           weights="weight", resolution_parameter=RESOLUTION,
                                           seed=0)
else:
    raise ValueError(f"unknown METHOD {METHOD!r}")

parts = sorted((set(c) for c in communities), key=len, reverse=True)
labels = np.empty(len(cls), int)
for c, part in enumerate(parts):
    labels[list(part)] = c
np.save(HERE / f"cls_{METHOD}_labels.npy", labels)

# ---- cluster -> scene mapping, ordered by within-cluster cohesion ----
# `order` puts the most-typical scene first, so a sampler that wants k scenes from
# a cluster can take a prefix and get its centre rather than its fringe.
order = [sorted(part, key=lambda n: -sim[n, sorted(part)].mean()) for part in parts]
(HERE / "clusters_layered.json").write_text(json.dumps({
    "method": METHOD,
    "threshold": THRESH,
    "resolution": RESOLUTION,
    "n_images": n_images,
    "post_final_ln": post_final_ln,
    "cls_layers": d["cls_layers"].tolist() if "cls_layers" in d.files else None,
    "num_scenes": len(cls),
    "clusters": [{
        "cluster": c,
        "size": len(idx),
        "cohesion": float(sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()) if len(idx) > 1 else None,
        "scenes": [{"subset": str(subsets[n]), "scene": str(scenes[n])} for n in idx],
    } for c, idx in enumerate(order)],
}, indent=1))

# ---- contact sheet: thumbnails grouped by cluster ----
(HERE / "thumbs").mkdir(exist_ok=True)
for n, fs in enumerate(frames):
    if not (t := HERE / f"thumbs/{n:04d}.jpg").exists():
        im = Image.open(fs[0]).convert("RGB")  # first of the scene's sampled frames
        im.resize((THUMB, round(THUMB * im.height / im.width))).save(t, quality=82)

html = ["<style>body{background:#111;color:#ddd;font:13px system-ui;margin:0;padding:16px}"
        "h2{font-size:14px;font-weight:600;margin:24px 0 8px;position:sticky;top:0;background:#111;padding:6px 0}"
        "div{display:flex;flex-wrap:wrap;gap:4px}figure{margin:0}img{display:block;border-radius:3px}"
        "figcaption{font-size:9px;color:#888;font-family:ui-monospace;max-width:160px;overflow:hidden}</style>"]
for c, idx in enumerate(order):
    html.append(f"<h2>cluster {c} &mdash; n={len(idx)}, mean pairwise cos "
                f"{sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean() if len(idx) > 1 else float('nan'):.3f}</h2><div>")
    html += [f'<figure><img src="thumbs/{n:04d}.jpg" width={THUMB} loading=lazy>'
             f'<figcaption>{c}.{k} {subsets[n]}/{scenes[n][:10]}</figcaption></figure>' for k, n in enumerate(idx)]
    html.append("</div>")
(HERE / "6k_clusters.html").write_text("\n".join(html))

print(f"cos sim: min {w.min():.3f} median {np.median(w):.3f} max {w.max():.3f}, {keep.sum()} edges")
print(f"{len(parts)} clusters ({sum(len(p) > 1 for p in parts)} non-singleton), "
      f"sizes {[len(p) for p in parts][:10]}, modularity@1 {nx.community.modularity(G, parts, weight='weight'):.3f}")
print("wrote clusters_layered.json, clusters_layered.html")
