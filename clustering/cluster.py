"""Community-detect the scene embeddings, so training can sample across clusters.

Adapted from the clustering-imgs script: cosine similarity on the CLS vectors,
sparsified to a graph, partitioned by modularity. Two differences worth knowing:

* `METHOD` picks Louvain (networkx, no extra deps) or Leiden (leidenalg). Same
  objective -- RBConfiguration is resolution-gamma modularity -- but Leiden's
  refinement step guarantees connected communities, which Louvain does not.

    ~/clustering-imgs/.venv/bin/python clustering/cluster.py

Writes `clusters.json` -- the cluster -> scene mapping the training-set sampler reads.
"""

import json
import numpy as np, networkx as nx
from pathlib import Path
from PIL import Image

METHOD = "leiden"  # "leiden" or "louvain"
THRESH = 0     # sparsification quantile: keep the top (1-THRESH) most-similar pairs as edges
RESOLUTION = 2.0  # resolution; >1 penalizes merging, giving more/smaller clusters
THUMB = 160       # contact-sheet thumbnail width, px

HERE = Path(__file__).parent
d = np.load(HERE / "dinov3_features.npz")
cls, scenes, subsets, frames = d["cls"].astype(np.float32), d["scenes"], d["subsets"], d["frames"]

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

X = cls / np.linalg.norm(cls, axis=1, keepdims=True)
sim = X @ X.T
np.save(HERE / "cls_cosine.npy", sim)

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
(HERE / "clusters.json").write_text(json.dumps({
    "method": METHOD,
    "threshold": THRESH,
    "resolution": RESOLUTION,
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
for n, f in enumerate(frames):
    if not (t := HERE / f"thumbs/{n:04d}.jpg").exists():
        im = Image.open(f).convert("RGB")
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
(HERE / "clusters.html").write_text("\n".join(html))

print(f"cos sim: min {w.min():.3f} median {np.median(w):.3f} max {w.max():.3f}, {keep.sum()} edges")
print(f"{len(parts)} clusters ({sum(len(p) > 1 for p in parts)} non-singleton), "
      f"sizes {[len(p) for p in parts][:10]}, modularity@1 {nx.community.modularity(G, parts, weight='weight'):.3f}")
print("wrote clusters.json, clusters.html")
