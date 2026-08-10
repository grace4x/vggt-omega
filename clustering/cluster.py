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

    ~/clustering-imgs/.venv/bin/python clustering/cluster.py [--out PATH]

Writes `clusters.json` -- the cluster -> scene mapping the training-set sampler reads.
"""

import argparse
import json
import numpy as np, networkx as nx
from pathlib import Path
from PIL import Image

METHOD = "leiden"  # "leiden" or "louvain"
THRESH = 0     # sparsification quantile: keep the top (1-THRESH) most-similar pairs as edges
RESOLUTION = 2.0  # resolution; >1 penalizes merging, giving more/smaller clusters
THUMB = 160       # contact-sheet thumbnail width, px

HERE = Path(__file__).parent
OUT = HERE / "clusters.json"  # cluster -> scene mapping the training-set sampler reads
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=OUT, metavar="PATH",
                     help="path to write the cluster -> scene mapping (default: %(default)s)")
OUT = parser.parse_args().out
d = np.load(HERE / "6k_features.npz")
cls, scenes, subsets, frames = d["cls"].astype(np.float32), d["scenes"], d["subsets"], d["frames"]
n_images = int(d["n_images"])  # frames sampled per scene by extract_features.py

# DL3DVDataset keeps a single stored (H, W) so scenes can stack into a batch, and
# drops the rest. Clustering scenes it would never load only dilutes the samples
# drawn from them, so apply the same filter here.
hw = d["image_hw"]
shapes, counts = np.unique(hw, axis=0, return_counts=True)
target = shapes[counts.argmax()]
keep_scene = (hw == target).all(1)
if not keep_scene.all():
    print(f"keeping {keep_scene.sum()} scenes at {tuple(int(v) for v in target)}; "
          f"dropped {(~keep_scene).sum()} at mismatched sizes")
    cls, scenes, subsets, frames = cls[keep_scene], scenes[keep_scene], subsets[keep_scene], frames[keep_scene]

n_layers = len(d["cls_layers"])
if n_layers > 1:
    # Each layer's CLS vector, after extract_features.py's per-layer L2-normalize, turns out to be
    # ~90-99.6% (worse in earlier layers) a "DC" direction that's shared by every image regardless
    # of content -- not scene signal. Averaged into the combined cosine similarity, that near-constant
    # component drowns out the actual per-scene variation, collapsing sim into a narrow, uniformly-high
    # band with nothing for modularity to partition on (hence all-singleton clusters). Subtract each
    # layer's dataset-mean direction and re-normalize before combining layers, to strip that bias out.
    # (Only done when combining multiple layers -- with a single layer there's no cross-layer
    # averaging to dilute, so leave its DC component alone.)
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
centroid = X.mean(axis=1)                                # (N, dim); NOT re-normalized
sim = centroid @ centroid.T  # == average of the n_images^2 pairwise cosine similarities per scene pair

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

# ---- cluster -> scene mapping, ordered by within-cluster cohesion ----
# `order` puts the most-typical scene first, so a sampler that wants k scenes from
# a cluster can take a prefix and get its centre rather than its fringe.
order = [sorted(part, key=lambda n: -sim[n, sorted(part)].mean()) for part in parts]
OUT.write_text(json.dumps({
    "method": METHOD,
    "threshold": THRESH,
    "resolution": RESOLUTION,
    "n_images": n_images,
    "num_scenes": len(cls),
    "clusters": [{
        "cluster": c,
        "size": len(idx),
        "cohesion": float(sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()) if len(idx) > 1 else None,
        "scenes": [{"subset": str(subsets[n]), "scene": str(scenes[n])} for n in idx],
    } for c, idx in enumerate(order)],
}, indent=1))

# ---- contact sheet: thumbnails grouped by cluster ----
# Named `{subset}_{scene}.jpg` -- the thumbs-dir layout get_cluster_loss_html.py reads. Keying on
# the scene rather than its row index keeps the cache valid across runs, which the index does not:
# any change to the subset or to the size filter above renumbers every row.
(HERE / "thumbs").mkdir(exist_ok=True)
thumb_names = [f"{subsets[n]}_{scenes[n]}.jpg" for n in range(len(frames))]
for n, fs in enumerate(frames):
    if not (t := HERE / "thumbs" / thumb_names[n]).exists():
        im = Image.open(fs[0]).convert("RGB")  # first of the scene's sampled frames
        im.resize((THUMB, round(THUMB * im.height / im.width))).save(t, quality=82)

html = ["<style>body{background:#111;color:#ddd;font:13px system-ui;margin:0;padding:16px}"
        "h2{font-size:14px;font-weight:600;margin:24px 0 8px;position:sticky;top:0;background:#111;padding:6px 0}"
        "div{display:flex;flex-wrap:wrap;gap:4px}figure{margin:0}img{display:block;border-radius:3px}"
        "figcaption{font-size:9px;color:#888;font-family:ui-monospace;max-width:160px;overflow:hidden}</style>"]
for c, idx in enumerate(order):
    html.append(f"<h2>cluster {c} &mdash; n={len(idx)}, mean pairwise cos "
                f"{sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean() if len(idx) > 1 else float('nan'):.3f}</h2><div>")
    html += [f'<figure><img src="thumbs/{thumb_names[n]}" width={THUMB} loading=lazy>'
             f'<figcaption>{c}.{k} {subsets[n]}/{scenes[n][:10]}</figcaption></figure>' for k, n in enumerate(idx)]
    html.append("</div>")
(HERE / "clusters.html").write_text("\n".join(html))

print(f"cos sim: min {w.min():.3f} median {np.median(w):.3f} max {w.max():.3f}, {keep.sum()} edges")
print(f"{len(parts)} clusters ({sum(len(p) > 1 for p in parts)} non-singleton), "
      f"sizes {[len(p) for p in parts][:10]}, modularity@1 {nx.community.modularity(G, parts, weight='weight'):.3f}")
print(f"wrote {OUT.name}, clusters.html")
