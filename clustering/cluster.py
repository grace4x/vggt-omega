"""Community-detect the scene embeddings, so training can sample across clusters.

Adapted from the clustering-imgs script: cosine similarity on the CLS vectors,
then a partition. `--method` picks:

* `leiden` / `louvain` -- sparsify the pairwise-cosine graph and cut it by
  RBConfiguration modularity. Leiden's refinement step guarantees connected
  communities, which Louvain does not.
* `kmeans` -- spherical k-means on the scene centroids (cosine assignment,
  centres re-normalized). `--k` defaults to 150. Same typicality/cohesion
  scores as the modularity path, so `subset.py` can prefix the json either way.

After the partition, `--dedup` drops near-duplicate scenes *inside* a cluster:
if two L2-normalized scene centroids have cosine >= the threshold, the less
typical one is removed so the sampler does not draw the same visual twice.
Typicality is the same mean-to-other-members score used to order the json.
`--dedup 1` disables. Default is 0.90 -- 0.99 matches almost nothing in this
embedding (nearest in-cluster pair tops out around 0.97).

Each scene is `extract_features.py`'s N_IMAGES CLS vectors, not one. The edge
weight between two scenes is the average of all N_IMAGES x N_IMAGES pairwise cosine
similarities between their sampled images, computed without an O(n^2) loop: that
average equals the dot product of the two scenes' (non-renormalized) centroids of
per-image unit vectors.

    ~/clustering-imgs/.venv/bin/python clustering/cluster.py [--out PATH]
    ~/clustering-imgs/.venv/bin/python clustering/cluster.py --method kmeans [--k 150]
    ~/clustering-imgs/.venv/bin/python clustering/cluster.py --dedup 0.90

Writes `clusters.json` (or `clusters_kmeans_k<k>.json`) -- the cluster -> scene
mapping the training-set sampler reads.
"""

import argparse
import json
import numpy as np, networkx as nx
from pathlib import Path
from PIL import Image

METHOD = "leiden"  # "leiden", "louvain", or "kmeans"
THRESH = 0     # sparsification quantile: keep the top (1-THRESH) most-similar pairs as edges
RESOLUTION = 2.0  # resolution; >1 penalizes merging, giving more/smaller clusters
K = 150           # k-means cluster count; ignored unless METHOD is kmeans
N_INIT = 10       # k-means++ restarts; best run (mean cosine to centre) is kept
MAX_ITER = 300
SEED = 0
DEDUP = 0.90      # drop a scene if cosine to a kept cluster-mate is >= this; >=1 disables
THUMB = 160       # contact-sheet thumbnail width, px

HERE = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--method", choices=("leiden", "louvain", "kmeans"), default=METHOD)
parser.add_argument("--k", type=int, default=K, help="k-means cluster count (ignored otherwise)")
parser.add_argument("--n-init", type=int, default=N_INIT,
                    help="k-means++ restarts; best run is kept")
parser.add_argument("--max-iter", type=int, default=MAX_ITER)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--out", type=Path, default=None, metavar="PATH",
                    help="path to write the cluster -> scene mapping "
                         "(default: clusters.json, or clusters_kmeans_k<k>.json for k-means)")
parser.add_argument("--dedup", type=float, default=DEDUP, metavar="COS",
                    help="within a cluster, drop a scene whose cosine (unit centroid) "
                         "to a kept member is >= this (default 0.90). >=1 disables")
args = parser.parse_args()
METHOD, K, N_INIT, MAX_ITER, SEED = args.method, args.k, args.n_init, args.max_iter, args.seed
DEDUP = args.dedup
if METHOD == "kmeans" and K < 1:
    raise SystemExit("--k must be >= 1")
OUT = args.out or HERE / (f"clusters_kmeans_k{K}.json" if METHOD == "kmeans" else "clusters.json")


def spherical_kmeans(C, k, seed, n_init, max_iter):
    """Spherical k-means on scene centroids, returned largest-cluster-first.

    Rows of `C` are L2-normalized so assignment is cosine (equivalent to Euclidean on the
    sphere). Each centre is the mean of its assigned unit vectors, re-normalized. k-means++
    init, `n_init` restarts, best run kept by mean cosine to assigned centre. Empty clusters
    are re-seeded at the worst-assigned point and dropped from the return if they stay empty.
    """
    X = np.asarray(C, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, 1e-12)
    n, d = X.shape
    k = min(int(k), n)
    rng = np.random.default_rng(seed)

    best_obj = -np.inf
    best_labels = None
    for _ in range(n_init):
        centers = np.empty((k, d), dtype=X.dtype)
        centers[0] = X[rng.integers(n)]
        closest = np.full(n, np.inf)
        for c in range(1, k):
            # ||x - c||^2 = 2(1 - cos) on the unit sphere
            d2 = np.maximum(0.0, 2.0 - 2.0 * (X @ centers[c - 1]))
            closest = np.minimum(closest, d2)
            s = float(closest.sum())
            centers[c] = X[rng.integers(n)] if s <= 0 else X[rng.choice(n, p=closest / s)]
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

        labels = np.full(n, -1)
        for _it in range(max_iter):
            sim_c = X @ centers.T
            new_labels = sim_c.argmax(1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = np.empty_like(centers)
            for c in range(k):
                m = labels == c
                if m.any():
                    mu = X[m].mean(0)
                    mu_n = np.linalg.norm(mu)
                    new_centers[c] = mu / mu_n if mu_n > 1e-12 else centers[c]
                else:
                    new_centers[c] = X[sim_c.max(1).argmin()]
            shift = 1.0 - float((new_centers * centers).sum(1).mean())
            centers = new_centers
            if shift < 1e-6:
                break

        obj = float((X * centers[labels]).sum())
        if obj > best_obj:
            best_obj = obj
            best_labels = labels.copy()

    print(f"spherical k-means: k={k} n_init={n_init}, "
          f"mean cosine to assigned centre {best_obj / n:.4f}")
    parts = [np.where(best_labels == c)[0].tolist() for c in range(k)]
    empty = sum(1 for p in parts if not p)
    if empty:
        print(f"warning: dropped {empty} empty cluster(s)")
    return sorted((sorted(p) for p in parts if p), key=len, reverse=True)


def dedup_parts(parts, centroid, sim, thresh):
    """Drop near-duplicates inside each cluster.

    Cosine is between L2-normalized scene centroids -- two copies of the same place
    sit near 1 even when their own frames disagree (the unnormalized `sim` used for
    graph edges / typicality would then be ~self-similarity, not ~1). Within a
    cluster, scenes are considered most-typical-first (mean `sim` to other members,
    same score as the json order); a scene is kept iff its cosine to every already-
    kept member is < `thresh`. Always keeps at least one scene. `thresh >= 1`
    disables. Returns (parts, n_dropped), still largest-cluster-first.
    """
    if not (0.0 < thresh < 1.0):
        return parts, 0
    nrm = np.linalg.norm(centroid, axis=1, keepdims=True)
    unit = np.asarray(centroid, dtype=np.float64) / np.maximum(nrm, 1e-12)
    out, n_drop = [], 0
    for part in parts:
        m = np.asarray(sorted(part), dtype=np.int64)
        if len(m) <= 1:
            out.append(m.tolist())
            continue
        s = sim[np.ix_(m, m)].copy()
        np.fill_diagonal(s, 0.0)
        typ = s.sum(1) / (len(m) - 1)
        local = unit[m]
        cos = local @ local.T
        kept_local = []
        for li in np.argsort(-typ, kind="stable"):
            li = int(li)
            if not kept_local or float(cos[li, kept_local].max()) < thresh:
                kept_local.append(li)
        n_drop += len(m) - len(kept_local)
        out.append(m[kept_local].tolist())
    return sorted(out, key=len, reverse=True), n_drop


d = np.load(HERE / "train_features_layers.npz")
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

if METHOD == "kmeans":
    parts = spherical_kmeans(centroid, K, SEED, N_INIT, MAX_ITER)
else:
    i, j = np.triu_indices(len(cls), 1)
    w = sim[i, j]
    keep = w >= np.quantile(w, THRESH)
    ew = w[keep] + 1  # shift cos sim [-1,1] -> [0,2] so modularity sees one sign
    G = nx.Graph()
    G.add_nodes_from(range(len(cls)))
    G.add_weighted_edges_from(zip(i[keep], j[keep], ew))
    print(f"cos sim: min {w.min():.3f} median {np.median(w):.3f} max {w.max():.3f}, "
          f"{keep.sum()} of {len(w)} pairs kept as edges (weight +1 -> [{ew.min():.3f}, {ew.max():.3f}])")

    if METHOD == "louvain":
        communities = nx.community.louvain_communities(G, weight="weight", resolution=RESOLUTION, seed=SEED)
    elif METHOD == "leiden":
        import igraph as ig, leidenalg

        g = ig.Graph(n=len(cls), edges=list(zip(i[keep].tolist(), j[keep].tolist())),
                     edge_attrs={"weight": ew.tolist()})
        communities = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                               weights="weight", resolution_parameter=RESOLUTION,
                                               seed=SEED)
    else:
        raise ValueError(f"unknown METHOD {METHOD!r}")
    parts = sorted((set(c) for c in communities), key=len, reverse=True)
    # Modularity needs a partition of every graph node; compute it before dedup
    # removes scenes, otherwise nx.community.modularity rejects the cover.
    mod = nx.community.modularity(G, parts, weight="weight")

parts, n_dropped = dedup_parts(parts, centroid, sim, DEDUP)
if 0.0 < DEDUP < 1.0:
    print(f"dedup @{DEDUP:g}: dropped {n_dropped} near-duplicate scene(s) "
          f"({len(cls) - n_dropped} kept)")

# ---- cluster -> scene mapping, ordered by within-cluster cohesion ----
# Score each scene by its mean similarity to the OTHER members of its cluster, ascending,
# so `order` puts the LEAST-typical scene first: a sampler that wants k scenes from a
# cluster takes a prefix and gets its fringe -- the outliers/hard cases -- rather than its
# centre. (Negate `score` for most-typical-first.) The diagonal is excluded because
# sim[n, n] is the mean pairwise cosine among scene n's own sampled frames, not a distance
# to anything in the cluster; leaving it in would rank internally inconsistent scenes --
# big camera motion, scene cuts, exposure swings -- as fringe wherever they actually sit.
order = []
for part in parts:
    m = sorted(part)
    s = sim[np.ix_(m, m)].copy()
    np.fill_diagonal(s, 0.0)
    score = s.sum(1) / max(len(m) - 1, 1)  # max(): a singleton cluster has no other members
    order.append([m[k] for k in np.argsort(score, kind="stable")])
n_kept = sum(len(p) for p in parts)
meta = {"method": METHOD, "n_images": n_images, "num_scenes": n_kept,
        "n_dropped": n_dropped, "dedup": DEDUP, "seed": SEED}
if METHOD == "kmeans":
    meta.update(k=K, n_init=N_INIT)
else:
    meta.update(threshold=THRESH, resolution=RESOLUTION)
OUT.write_text(json.dumps({
    **meta,
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
(HERE / f"{OUT.name}.html").write_text("\n".join(html))

print(f"{len(parts)} clusters ({sum(len(p) > 1 for p in parts)} non-singleton), "
      f"{n_kept} scenes" + (f" ({n_dropped} dropped)" if n_dropped else "")
      + f", sizes {[len(p) for p in parts][:10]}"
      + (f", modularity@1 {mod:.3f}" if METHOD != "kmeans" else f" (k-means k={K})"))
print(f"wrote {OUT}, {HERE / (OUT.name + '.html')}")
