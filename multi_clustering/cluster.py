"""Cluster DL3DV and ScanNet scenes together with spherical k-means.

`clustering/cluster.py` partitions one dataset; this partitions the *pool*, so a
cluster is a visual mode of the mixed training set rather than of either half, and
`subset.py` can sample across modes instead of across datasets. Everything else is
the same recipe: cosine similarity on the concatenated per-layer CLS vectors,
per-layer DC removal, spherical k-means, in-cluster dedup, then an ordering by
within-cluster typicality.

`--target` / `--target-pct` add density-based pruning on top of the dedup, following
"Effective Pruning of Web-Scale Datasets" (arXiv:2401.04578, Sec. 3 + App. A). Dedup
removes redundancy *within* a cluster pairwise; this removes it *between* clusters, by
deciding how many scenes each cluster deserves. A cluster is scored by its complexity
C_j = d_inter,j * d_intra,j -- the mean cosine distance to its l nearest neighbouring
centroids, times the mean cosine distance of its own members to its centroid. Tight
clusters sitting in a crowded neighbourhood are cheap to cover and give up scenes;
loose, isolated ones keep theirs. softmax(C/tau) turns that into a target share of the
pruned size N, and Eq. 3 reconciles the shares with the actual cluster sizes. Within a
cluster the *least* prototypical scenes are kept, as in SSP-Pruning -- which is already
the order this script writes, so pruning is a prefix of it.

The scene list comes from `scene_sets.py`, i.e. from the real `DL3DVDataset` under
the same flags as the training run, so what gets clustered is what gets trained on.
Features must be poolable -- same model, layers, --final-ln and frames/scene -- and
mismatches are refused rather than silently averaged.

`--center` is the one genuinely new knob:

* `pooled` (default) subtracts each layer's mean direction over all scenes, both
  datasets included. Cosine then still carries the (large) "indoor ScanNet scan vs
  outdoor DL3DV video" offset, so k-means may split largely along dataset lines --
  which is the honest answer if the two sets really do not overlap visually.
* `per-dataset` subtracts each layer's mean *within* each dataset, deleting that
  offset, so clusters form on content and tend to mix. Use it when the point is
  one-scene-per-mode coverage across both sets rather than a faithful similarity.

Either way the per-cluster dataset mix is printed and recorded, so the choice can
be checked rather than assumed.

    .venv/bin/python multi_clustering/cluster.py --k 150
    .venv/bin/python multi_clustering/cluster.py --k 150 --center per-dataset

Writes `clusters_dl3dv_scannet_k<k>.json` (+ a contact sheet beside it) for `subset.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, however this is invoked

import numpy as np
from PIL import Image

from scene_sets import DEFAULTS, HERE, REPO, add_source_args, key, resolve, sources_from_args

K = 150       # k-means cluster count
N_INIT = 10   # k-means++ restarts; best run (mean cosine to centre) is kept
MAX_ITER = 300
SEED = 0
DEDUP = 0.90  # drop a scene if cosine to a kept cluster-mate is >= this; >=1 disables
L_NEIGH = 20  # centroids averaged into d_inter; the paper's l, ablated in its Sec. 5.4
TAU = 0.1     # softmax temperature turning complexity into a target share
THUMB = 160   # contact-sheet thumbnail width, px


def spherical_kmeans(C, k, seed, n_init, max_iter):
    """Spherical k-means on scene centroids, returned largest-cluster-first.

    Rows of `C` are L2-normalized so assignment is cosine (equivalent to Euclidean on
    the sphere). Each centre is the mean of its assigned unit vectors, re-normalized.
    k-means++ init, `n_init` restarts, best run kept by mean cosine to assigned centre.
    Empty clusters are re-seeded at the worst-assigned point and dropped if they stay
    empty.
    """
    X = np.asarray(C, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    n, d = X.shape
    k = min(int(k), n)
    rng = np.random.default_rng(seed)

    best_obj, best_labels = -np.inf, None
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
            best_obj, best_labels = obj, labels.copy()

    print(f"spherical k-means: k={k} n_init={n_init}, "
          f"mean cosine to assigned centre {best_obj / n:.4f}")
    parts = [np.where(best_labels == c)[0].tolist() for c in range(k)]
    if empty := sum(1 for p in parts if not p):
        print(f"warning: dropped {empty} empty cluster(s)")
    return sorted((sorted(p) for p in parts if p), key=len, reverse=True)


def dedup_parts(parts, centroid, sim, thresh):
    """Drop near-duplicates inside each cluster; returns (parts, n_dropped).

    Cosine is between L2-normalized scene centroids -- two captures of the same place
    sit near 1 even when their own frames disagree. Within a cluster, scenes are
    considered most-typical-first (mean `sim` to other members); a scene is kept iff
    its cosine to every already-kept member is < `thresh`. Always keeps one.
    """
    if not (0.0 < thresh < 1.0):
        return parts, 0
    unit = np.asarray(centroid, dtype=np.float64)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
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
        kept = []
        for li in np.argsort(-typ, kind="stable"):
            li = int(li)
            if not kept or float(cos[li, kept].max()) < thresh:
                kept.append(li)
        n_drop += len(m) - len(kept)
        out.append(m[kept].tolist())
    return sorted(out, key=len, reverse=True), n_drop


def cluster_geometry(parts, centroid, l):
    """Per-cluster (d_inter, d_intra), both mean cosine distances in [0, 2].

    d_intra is the mean cosine distance from a cluster's members to its own centroid --
    small when the cluster is tight, i.e. when its scenes are near-redundant with each
    other. d_inter is the mean cosine distance to the `l` nearest *other* centroids --
    small when the cluster sits in a crowded part of the embedding space, where the
    neighbouring modes already cover much of what it covers. Scene vectors are
    L2-normalized first (`centroid` is a plain mean over a scene's frames, so its length
    carries how much those frames agree, which is not what a cosine distance should
    weight); a cluster centroid is the renormalized mean of its members' unit vectors,
    i.e. the same centre spherical k-means converged to, recomputed post-dedup.
    """
    unit = np.asarray(centroid, dtype=np.float64)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    members = [np.asarray(p, dtype=np.int64) for p in parts]
    centers = np.stack([unit[m].mean(0) for m in members])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    d_intra = np.array([float(np.mean(1.0 - unit[m] @ centers[j]))
                        for j, m in enumerate(members)])
    l = min(int(l), len(parts) - 1)
    if l < 1:
        return np.ones(len(parts)), d_intra  # nothing to be near; d_inter drops out
    cc = centers @ centers.T
    np.fill_diagonal(cc, -np.inf)                 # a centroid is not its own neighbour
    nearest = np.sort(cc, axis=1)[:, -l:]         # l largest cosines == l nearest centroids
    return 1.0 - nearest.mean(axis=1), d_intra


def complexity_probs(d_inter, d_intra, tau):
    """Eq. 1-2: the softmax over cluster complexity C_j = d_inter,j * d_intra,j.

    The product, not the sum: a cluster is complex only if it is *both* spread out and
    far from its neighbours, so either distance being small pulls its share down.
    """
    z = (np.asarray(d_inter) * np.asarray(d_intra)) / tau
    e = np.exp(z - z.max())  # softmax is shift-invariant; subtract the max for stability
    return e / e.sum()


def allocate(desired, sizes, total, floor=1):
    """Integer keep-counts as close to `desired` as the cluster sizes allow.

    Eq. 3 of the paper: minimize sum_j (x_j - desired_j)^2 subject to sum_j x_j = total
    and floor <= x_j <= M_j. That is separable in x with one equality constraint and a
    box, so KKT gives the optimum in closed form -- x_j = clip(desired_j + lam, floor,
    M_j) for the single multiplier `lam` that makes the sum come out right. The sum is
    non-decreasing in lam, so bisection pins it down and the QP solver the paper reaches
    for (qpsolvers) is not needed; this is the same optimum, not an approximation of it.
    The continuous solution is then floored and the leftover units given to the largest
    fractional parts, which holds the total exactly and lands within one scene per
    cluster of the real optimum.
    """
    d = np.asarray(desired, dtype=np.float64)
    hi = np.asarray(sizes, dtype=np.float64)
    lo = np.minimum(float(floor), hi)  # a cluster smaller than the floor cannot honour it
    total = float(min(max(total, lo.sum()), hi.sum()))
    a, b = float((lo - d).min()), float((hi - d).max())
    for _ in range(200):
        lam = 0.5 * (a + b)
        if np.clip(d + lam, lo, hi).sum() < total:
            a = lam
        else:
            b = lam
    x = np.clip(d + 0.5 * (a + b), lo, hi)
    keep = np.floor(x).astype(np.int64)
    # sum(x) == total and keep == floor(x), so the shortfall is under one unit per
    # cluster and every cluster with a fractional part has room for the +1 it may get.
    for i in np.argsort(-(x - keep), kind="stable"):
        if int(keep.sum()) >= total:
            break
        if keep[i] < sizes[i]:
            keep[i] += 1
    return keep


def density_prune(order, centroid, total, l, tau, floor=1):
    """Prune to `total` scenes, taking fewer from dense and crowded clusters.

    `order` is least-prototypical-first per cluster (see below), so keeping a cluster's
    first x_j scenes keeps its fringe -- SSP-Pruning's choice, and the reason this runs
    after the ordering rather than before it. Returns the pruned lists and a per-cluster
    record of what drove each decision; clusters allocated nothing are dropped from
    both, which can only happen when `total` is below the cluster count.
    """
    sizes = np.array([len(o) for o in order], dtype=np.int64)
    d_inter, d_intra = cluster_geometry(order, centroid, l)
    prob = complexity_probs(d_inter, d_intra, tau)
    total = int(min(total, sizes.sum()))
    keep = allocate(prob * total, sizes, total, floor=min(floor, total // max(len(order), 1)))
    pruned, stats = [], []
    for o, n, size, di, da, pj in zip(order, keep, sizes, d_inter, d_intra, prob):
        if not (n := int(n)):
            continue
        pruned.append(o[:n])
        stats.append({"size_before": int(size), "d_inter": round(float(di), 5),
                      "d_intra": round(float(da), 5),
                      "complexity": round(float(di * da), 6), "p": round(float(pj), 6),
                      "desired": round(float(pj * total), 2)})
    return pruned, stats


def load_pool(sources, split, num_frames):
    """Gather the wanted scenes' features from each source's npz into one pool.

    Returns (cls, rows) where `cls` is (N, n_images, dim) float32 and `rows` carries
    the per-scene bookkeeping (dataset, subset, scene, first frame path) in the same
    order. Selection is by (subset, scene), not by row index, so an npz covering a
    superset of the wanted scenes -- e.g. clustering/'s full 4288-scene DL3DV
    extraction -- is used as a cache rather than re-extracted.
    """
    cls_parts, rows, recipe, models = [], [], None, set()
    for source in sources:
        wanted, _hw = resolve(source, split, num_frames)
        if not source.features.exists():
            raise SystemExit(
                f"missing {source.features}\n"
                f"  run: .venv/bin/python multi_clustering/extract_features.py "
                f"--dataset {source.name}")
        d = np.load(source.features, allow_pickle=False)
        # Cosine across datasets only means anything if both sides came out of the same
        # model under the same recipe, so pooling incompatible npz files is an error.
        here = (int(d["n_images"]), tuple(int(v) for v in d["cls_layers"]), bool(d["final_ln"]))
        if recipe is None:
            recipe = here
        elif here != recipe:
            raise SystemExit(f"{source.features} was extracted as "
                             f"(n_images, layers, final_ln)={here}, incompatible with {recipe}")
        # `clustering/extract_features.py` predates the `model` field, so an npz without
        # one is taken on trust rather than rejected -- the layer indices above only line
        # up between two runs of the same architecture anyway.
        models.add(str(d["model"]) if "model" in d.files else None)
        index = {f"{s}/{n}": i for i, (s, n) in enumerate(zip(d["subsets"], d["scenes"]))}
        picked = [index.get(key(e)) for e in wanted]
        if (missing := [key(e) for e, i in zip(wanted, picked) if i is None]):
            raise SystemExit(
                f"{len(missing)} of {len(wanted)} {source.name} scenes are not in "
                f"{source.features} (e.g. {missing[:3]}); re-extract with the same flags")
        take = np.asarray(picked, dtype=np.int64)
        cls_parts.append(d["cls"][take].astype(np.float32))
        rows += [{"dataset": source.name, "subset": str(e["subset"]), "scene": str(e["scene"]),
                  "frame": str(d["frames"][i][0])} for e, i in zip(wanted, picked)]
        print(f"[{source.name}] {len(take)} scenes from {source.features.name}")
    dims = {c.shape[1:] for c in cls_parts}
    if len(dims) > 1:
        raise SystemExit(f"feature shapes disagree across sources: {dims}")
    if len(recorded := models - {None}) > 1:
        raise SystemExit(f"features come from different models: {sorted(recorded)}")
    if None in models and recorded:
        print(f"warning: an npz records no model; assuming {sorted(recorded)[0]}")
    return np.concatenate(cls_parts), rows, (*recipe, next(iter(recorded), "unrecorded"))


def centered_unit(cls, n_layers, groups):
    """Per-layer DC removal, then L2-normalize each image's full vector.

    After `extract_features.py`'s per-layer L2-normalize, each layer's CLS is ~90-99%
    a "DC" direction shared by every image regardless of content. Averaged into the
    combined cosine similarity that near-constant component drowns out the actual
    per-scene variation, leaving similarities in a narrow, uniformly-high band with
    nothing to partition on. Subtract each layer's mean direction and renormalize to
    strip it out. `groups` selects the population the mean is taken over: one group of
    everything (pooled) or one per dataset (which also deletes the between-dataset
    offset). With a single layer there is no cross-layer averaging to dilute, so the
    DC component is left alone -- matching clustering/cluster.py.
    """
    flat = cls.reshape(-1, cls.shape[-1])
    if n_layers == 1:
        X = flat / np.linalg.norm(flat, axis=-1, keepdims=True)
        return X.reshape(cls.shape)
    per_image_groups = np.repeat(groups, cls.shape[1])  # a scene's frames share its group
    layer_dims = cls.shape[-1] // n_layers
    X = np.empty_like(flat)
    for g in np.unique(per_image_groups):
        m = per_image_groups == g
        for k in range(n_layers):
            sl = slice(k * layer_dims, (k + 1) * layer_dims)
            seg = flat[m, sl] - flat[m, sl].mean(axis=0)
            X[m, sl] = seg / np.linalg.norm(seg, axis=-1, keepdims=True)
    X /= np.linalg.norm(X, axis=-1, keepdims=True)  # renormalize the full per-image vector
    return X.reshape(cls.shape)


p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--k", type=int, default=K, help="k-means cluster count (default: %(default)s)")
p.add_argument("--datasets", nargs="+", choices=tuple(DEFAULTS), default=list(DEFAULTS),
               metavar="NAME", help="sources to pool (default: %(default)s)")
p.add_argument("--center", choices=("pooled", "per-dataset"), default="pooled",
               help="population each layer's DC mean is taken over (default: %(default)s); "
                    "per-dataset deletes the between-dataset offset, mixing the clusters")
p.add_argument("--n-init", type=int, default=N_INIT, help="k-means++ restarts; best run kept")
p.add_argument("--max-iter", type=int, default=MAX_ITER)
p.add_argument("--seed", type=int, default=SEED)
p.add_argument("--dedup", type=float, default=DEDUP, metavar="COS",
               help="within a cluster, drop a scene whose cosine to a kept member is "
                    ">= this (default: %(default)s; >=1 disables)")
how = p.add_mutually_exclusive_group()
how.add_argument("--target", type=int, default=None, metavar="N",
                 help="density-based pruning: keep this many scenes in total, spread over "
                      "the clusters by complexity (default: off, keep everything)")
how.add_argument("--target-pct", type=float, default=None, metavar="PCT",
                 help="same, as a percent of the post-dedup scene count (e.g. 30)")
p.add_argument("--neighbors", type=int, default=L_NEIGH, metavar="L",
               help="centroids averaged into d_inter (default: %(default)s)")
p.add_argument("--tau", type=float, default=TAU, metavar="T",
               help="softmax temperature over cluster complexity (default: %(default)s); "
                    "smaller concentrates the budget on the most complex clusters")
p.add_argument("--min-keep", type=int, default=1, metavar="N",
               help="scenes every surviving cluster keeps when pruning (default: %(default)s)")
p.add_argument("--out", type=Path, default=None, metavar="PATH",
               help="cluster -> scene mapping "
                    "(default: clusters_<datasets>_k<k>[_per-dataset].json)")
p.add_argument("--thumbs", type=Path, default=REPO / "clustering" / "thumbs", metavar="DIR",
               help="contact-sheet thumbnail cache, shared with clustering/ (default: %(default)s)")
p.add_argument("--no-html", action="store_true", help="skip the contact sheet")
add_source_args(p)
args = p.parse_args()
if args.k < 1:
    raise SystemExit("--k must be >= 1")
if args.target is not None and args.target < 1:
    raise SystemExit("--target must be >= 1")
if args.target_pct is not None and not (0 < args.target_pct <= 100):
    raise SystemExit("--target-pct must be in (0, 100]")
if args.tau <= 0:
    raise SystemExit("--tau must be > 0")
out = args.out or HERE / ("clusters_" + "_".join(args.datasets) + f"_k{args.k}"
                          + ("_per-dataset" if args.center == "per-dataset" else "") + ".json")

sources = sources_from_args(args, names=args.datasets)
cls, rows, recipe = load_pool(sources, args.split, args.num_frames)
n_images, cls_layers, final_ln, model = recipe
datasets = np.array([r["dataset"] for r in rows])
print(f"pooling {len(rows)} scenes "
      f"({', '.join(f'{n}={int((datasets == n).sum())}' for n in dict.fromkeys(datasets))}), "
      f"{n_images} frames/scene, layers {cls_layers}"
      f"{', final-ln' if final_ln else ''}, model {model}")

groups = datasets if args.center == "per-dataset" else np.zeros(len(rows), dtype=np.int8)
X = centered_unit(cls, len(cls_layers), groups)
centroid = X.mean(axis=1)     # (N, dim); NOT re-normalized
sim = centroid @ centroid.T   # == the mean of the n_images^2 pairwise cosines per scene pair

# How separable the two datasets are in this space -- if the cross-dataset block is far
# below both within-dataset blocks, expect k-means to split along dataset lines and
# consider --center per-dataset.
names = list(dict.fromkeys(datasets))
if len(names) > 1:
    off = ~np.eye(len(rows), dtype=bool)
    for a in names:
        for b in names:
            m = np.outer(datasets == a, datasets == b) & off
            print(f"  mean cos {a:>8} x {b:<8} {sim[m].mean():+.4f}")

parts = spherical_kmeans(centroid, args.k, args.seed, args.n_init, args.max_iter)
parts, n_dropped = dedup_parts(parts, centroid, sim, args.dedup)
if 0.0 < args.dedup < 1.0:
    print(f"dedup @{args.dedup:g}: dropped {n_dropped} near-duplicate scene(s) "
          f"({len(rows) - n_dropped} kept)")

# ---- cluster -> scene mapping, ordered least-typical-first ----
# Each scene is scored by its mean similarity to the OTHER members of its cluster,
# ascending, so a sampler taking a prefix gets the cluster's fringe -- its outliers and
# hard cases -- rather than its centre. (`subset.py` samples at random instead, but the
# order is what makes the json readable and matches clustering/cluster.py.) The diagonal
# is excluded: sim[n, n] is the agreement among scene n's own frames, not a distance to
# anything in the cluster.
order = []
for part in parts:
    m = sorted(part)
    s = sim[np.ix_(m, m)].copy()
    np.fill_diagonal(s, 0.0)
    score = s.sum(1) / max(len(m) - 1, 1)  # a singleton has no other members
    order.append([m[i] for i in np.argsort(score, kind="stable")])

# ---- density-based pruning: how many scenes each cluster deserves ----
# The dedup above is pairwise and blind to the rest of the pool, so a cluster can survive
# it intact and still be redundant -- twenty tight clusters of the same kind of room cover
# little more than a few of them do. Complexity (Eq. 1) prices that in: fewer scenes from
# clusters that are tight (low d_intra) or hemmed in by neighbours (low d_inter).
prune_stats = None
n_pooled = sum(len(o) for o in order)
if args.target is not None or args.target_pct is not None:
    target = (args.target if args.target is not None
              else max(1, round(n_pooled * args.target_pct / 100.0)))
    if target >= n_pooled:
        print(f"prune: target {target} >= {n_pooled} post-dedup scenes, nothing to prune")
    else:
        order, prune_stats = density_prune(order, centroid, target, args.neighbors,
                                           args.tau, floor=args.min_keep)
        parts = [sorted(o) for o in order]
        loose = max(range(len(prune_stats)), key=lambda j: prune_stats[j]["complexity"])
        tight = min(range(len(prune_stats)), key=lambda j: prune_stats[j]["complexity"])
        print(f"prune to {target} of {n_pooled} scenes "
              f"(l={min(args.neighbors, len(prune_stats) - 1)}, tau={args.tau:g}): "
              f"{len(order)} clusters kept, sizes "
              f"{[len(o) for o in sorted(order, key=len, reverse=True)][:10]}")
        for label, j in (("most complex ", loose), ("least complex", tight)):
            st = prune_stats[j]
            print(f"  {label} cluster {j}: d_inter={st['d_inter']:.3f} "
                  f"d_intra={st['d_intra']:.3f} C={st['complexity']:.4f} "
                  f"-> {len(order[j])}/{st['size_before']} kept "
                  f"(P*N={st['desired']:.1f})")


def mix(idx):
    """Per-dataset scene counts for one cluster."""
    return {n: int(sum(rows[i]["dataset"] == n for i in idx)) for n in names}


def cohesion(idx):
    return (float(sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean())
            if len(idx) > 1 else None)


n_kept = sum(len(p) for p in parts)
out.write_text(json.dumps({
    "method": "kmeans",
    "k": args.k,
    "center": args.center,
    "n_init": args.n_init,
    "n_images": n_images,
    "cls_layers": list(cls_layers),
    "final_ln": final_ln,
    "model": model,
    "num_scenes": n_kept,
    "n_dropped": n_dropped,
    "dedup": args.dedup,
    "prune": None if prune_stats is None else {
        "method": "complexity-softmax",   # arXiv:2401.04578 Eq. 1-3
        "target": target,
        "n_before": n_pooled,
        "neighbors": min(args.neighbors, len(prune_stats) - 1),
        "tau": args.tau,
        "min_keep": args.min_keep,
    },
    "seed": args.seed,
    "split": args.split,
    "num_frames": args.num_frames,
    "sources": {s.name: {"root": str(s.root), "scenes": int((datasets == s.name).sum()),
                         "limit": s.limit, "features": str(s.features)} for s in sources},
    "clusters": [{
        "cluster": c,
        "size": len(idx),
        "cohesion": cohesion(idx),
        "datasets": mix(idx),
        **({} if prune_stats is None else prune_stats[c]),
        "scenes": [{"dataset": rows[i]["dataset"], "subset": rows[i]["subset"],
                    "scene": rows[i]["scene"]} for i in idx],
    } for c, idx in enumerate(order)],
}, indent=1))

# ---- contact sheet: thumbnails grouped by cluster ----
# `{subset}_{scene}.jpg`, the layout clustering/get_cluster_loss_html.py reads, and
# keyed on the scene rather than a row index so the cache survives a change of caps.
if not args.no_html:
    args.thumbs.mkdir(parents=True, exist_ok=True)
    thumb = [f"{r['subset']}_{r['scene']}.jpg" for r in rows]
    for i, r in enumerate(rows):
        if not (t := args.thumbs / thumb[i]).exists():
            im = Image.open(r["frame"]).convert("RGB")  # first of the scene's sampled frames
            im.resize((THUMB, round(THUMB * im.height / im.width))).save(t, quality=82)
    rel = Path(os.path.relpath(args.thumbs, out.parent))  # the html lives next to `out`
    html = ["<style>body{background:#111;color:#ddd;font:13px system-ui;margin:0;padding:16px}"
            "h2{font-size:14px;font-weight:600;margin:24px 0 8px;position:sticky;top:0;"
            "background:#111;padding:6px 0}div{display:flex;flex-wrap:wrap;gap:4px}"
            "figure{margin:0}img{display:block;border-radius:3px}"
            "figcaption{font-size:9px;font-family:ui-monospace;max-width:160px;overflow:hidden}"
            ".dl3dv figcaption{color:#7ab8ff}.scannet figcaption{color:#ffc46b}</style>"]
    for c, idx in enumerate(order):
        coh = cohesion(idx)
        st = "" if prune_stats is None else (
            f", pruned from {prune_stats[c]['size_before']} "
            f"(C={prune_stats[c]['complexity']:.4f})")
        html.append(f"<h2>cluster {c} &mdash; n={len(idx)}, "
                    f"{', '.join(f'{k}={v}' for k, v in mix(idx).items())}, mean pairwise cos "
                    f"{coh if coh is not None else float('nan'):.3f}{st}</h2><div>")
        html += [f'<figure class="{rows[i]["dataset"]}">'
                 f'<img src="{rel / thumb[i]}" width={THUMB} loading=lazy>'
                 f'<figcaption>{c}.{j} {rows[i]["dataset"]}/{rows[i]["scene"][:12]}</figcaption>'
                 f'</figure>' for j, i in enumerate(idx)]
        html.append("</div>")
    (html_path := out.with_suffix(".json.html")).write_text("\n".join(html))

pure = {n: sum(1 for idx in order if mix(idx)[n] == len(idx)) for n in names}
print(f"{len(parts)} clusters ({sum(len(p) > 1 for p in parts)} non-singleton), {n_kept} scenes"
      + (f" ({n_dropped} dropped)" if n_dropped else "")
      + (f" ({n_pooled - n_kept} pruned)" if prune_stats is not None else "")
      + f", sizes {sorted((len(p) for p in parts), reverse=True)[:10]}")
if len(names) > 1:
    print(f"dataset mix: {sum(1 for idx in order if all(v for v in mix(idx).values()))} mixed "
          f"clusters, " + ", ".join(f"{v} pure {n}" for n, v in pure.items()))
print(f"wrote {out}" + ("" if args.no_html else f", {html_path}"))
print(f"  next: .venv/bin/python multi_clustering/subset.py --k 5 --clusters {out.name}")
