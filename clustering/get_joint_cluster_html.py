"""Re-cluster the training *and* eval scenes together, and render a contact sheet.

`get_nearest_cluster_html.py` holds `cluster.py`'s training partition fixed and drops each
eval scene into the cluster it is closest to. This script does the other thing: it throws both
feature files into one pile and re-runs `cluster.py`'s pipeline over the union, so the eval
scenes get a vote on where the boundaries are. An eval scene that belongs to no training
cluster can then form (or join) its own community instead of being forced into the least-bad
existing one -- which is the failure mode the nearest-cluster sheet cannot show you.

The clustering is `cluster.py`'s, re-implemented over the concatenated scenes: per-layer
de-mean and renormalize, average each scene's N_IMAGES unit vectors into a centroid, take the
dense centroid dot products as edge weights, and partition by modularity (Leiden or Louvain,
RBConfiguration at `--resolution`). The one thing that necessarily differs is the de-meaning:
the per-layer means are taken over the union here, not over the training set, because the union
is the space being partitioned. Keep `--resolution`/`--threshold`/`--method` equal to
`cluster.py`'s constants unless you mean to compare partitions, and pass `--clusters` to have
the run print how far the joint partition drifts from the training-only one (adjusted Rand
index over the shared training scenes).

Writes `<features>_joint_<metric>.html`: one section per joint cluster, its scenes ordered
**most typical first** -- descending mean cosine similarity to the cluster's other members, so
the cluster's centre reads left-to-right into its fringe. Eval scenes are coloured by loss,
training scenes are grey (they were never evaluated). Note this is the reverse of the order
`cluster.py` writes into its json, which is least-typical-first so that `subset.py` can take a
prefix and get a cluster's outliers; `--out-json` here keeps that convention, and only the
sheet is flipped.

    ~/clustering-imgs/.venv/bin/python clustering/get_joint_cluster_html.py
    ... --metric loss_depth --resolution 2.0 --clusters clustering/clusters.json

Both npz files must come from the same `extract_features.py` settings (same --layers and
--final-ln).
"""

import argparse
import html as _html
import json
import os
import statistics
from pathlib import Path

import numpy as np

# The colour ramp, its interpolation and the quantile helper are shared with the other two
# sheets on purpose: they get compared side by side, so a given colour has to mean the same
# loss in all of them. `centroids` is cluster.py's feature transform, `spearman` the rank
# correlation the console summary reports.
from get_cluster_loss_html import NO_DATA, RAMP, quantile, ramp
from get_nearest_cluster_html import centroids, spearman

HERE = Path(__file__).parent


def adjusted_rand(a, b):
    """ARI between two labelings of the same items: 1 identical, ~0 chance-level."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return float("nan")
    cont = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(cont, (a, b), 1)
    comb2 = lambda x: (x * (x - 1) // 2).sum()
    ij, ai, bj = comb2(cont), comb2(cont.sum(1)), comb2(cont.sum(0))
    exp = ai * bj / comb2(np.array([n]))
    denom = (ai + bj) / 2 - exp
    return float((ij - exp) / denom) if denom else float("nan")


def partition(sim, method, threshold, resolution, seed):
    """cluster.py's graph build + community detection, returned largest-cluster-first."""
    import networkx as nx

    i, j = np.triu_indices(len(sim), 1)
    w = sim[i, j]
    keep = w >= np.quantile(w, threshold)
    ew = w[keep] + 1  # shift cos sim [-1,1] -> [0,2] so modularity sees one sign
    print(f"cos sim: min {w.min():.3f} median {np.median(w):.3f} max {w.max():.3f}, "
          f"{keep.sum()} of {len(w)} pairs kept as edges (weight +1 -> [{ew.min():.3f}, {ew.max():.3f}])")
    if method == "louvain":
        G = nx.Graph()
        G.add_nodes_from(range(len(sim)))
        G.add_weighted_edges_from(zip(i[keep], j[keep], ew))
        communities = nx.community.louvain_communities(G, weight="weight",
                                                       resolution=resolution, seed=seed)
    elif method == "leiden":
        import igraph as ig, leidenalg

        g = ig.Graph(n=len(sim), edges=list(zip(i[keep].tolist(), j[keep].tolist())),
                     edge_attrs={"weight": ew.tolist()})
        communities = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                               weights="weight", resolution_parameter=resolution,
                                               seed=seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    return sorted((sorted(c) for c in communities), key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=HERE / "6k_features_layers.npz",
                    help="eval-set features from extract_features.py")
    ap.add_argument("--train-features", type=Path, default=HERE / "train_features_layers.npz",
                    help="training-set features")
    ap.add_argument("--eval", type=Path,
                    default=HERE.parent / "runs/small-v5_default/eval/windows-small-v5-latest.jsonl",
                    help="eval jsonl; one row per scene per repeat")
    ap.add_argument("--metric", default="loss",
                    help="jsonl field to average (loss, loss_depth, abs_rel, ...)")
    ap.add_argument("--clusters", type=Path, default=None,
                    help="cluster.py's training-only clusters json; only used to report how "
                         "far this joint partition drifts from it")
    # Defaults mirror cluster.py's module constants; change both together to compare partitions.
    ap.add_argument("--method", choices=("leiden", "louvain"), default="leiden")
    ap.add_argument("--resolution", type=float, default=2.0,
                    help=">1 penalizes merging, giving more/smaller clusters")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="sparsification quantile: keep the top (1-t) most-similar pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hw-filter", action="store_true",
                    help="keep training scenes whose stored (H, W) is not the majority shape; "
                         "cluster.py drops them because DL3DVDataset never loads them")
    ap.add_argument("--thumbs", type=Path, default=HERE / "thumbs")
    ap.add_argument("--thumb-pattern", default="{subset}_{scene}.jpg",
                    help="thumbnail filename, formatted with {subset} and {scene}")
    ap.add_argument("--thumb-width", type=int, default=160)
    ap.add_argument("--out", type=Path, default=None,
                    help="output html (default <features>_joint_<metric>.html)")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="also write the joint partition as a cluster.py-shaped json "
                         "(least-typical-first, so subset.py can prefix it -- but its scenes "
                         "are a mix of train and eval, tagged \"split\", so a subset taken "
                         "off it is not a clean training list)")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"), default=(0.02, 0.98),
                    help="quantiles the colour scale saturates at")
    ap.add_argument("--vmin", type=float, default=None, help="override colour-scale minimum")
    ap.add_argument("--vmax", type=float, default=None, help="override colour-scale maximum")
    args = ap.parse_args()

    out_html = args.out or args.features.with_name(f"{args.features.stem}_joint_{args.metric}.html")
    for p in (args.features, args.train_features, args.eval):
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
    if args.clusters is not None and not args.clusters.exists():
        raise SystemExit(f"{args.clusters} does not exist")

    # ---- per-scene metric, averaged over repeats ----
    per_repeat = {}
    for line in args.eval.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if args.metric in r:
                per_repeat.setdefault(r["scene_id"], []).append(float(r[args.metric]))
    if not per_repeat:
        raise SystemExit(f"no rows with field {args.metric!r} in {args.eval}")
    scene_val = {s: sum(v) / len(v) for s, v in per_repeat.items()}
    print(f"{len(scene_val)} scenes x {min(map(len, per_repeat.values()))}-"
          f"{max(map(len, per_repeat.values()))} repeats from {args.eval}")

    # ---- features ----
    dt = np.load(args.train_features)
    de = np.load(args.features)
    for key in ("cls_layers", "final_ln"):
        if not np.array_equal(dt[key], de[key]):
            raise SystemExit(f"{key} differs between the two feature files "
                             f"({dt[key]} vs {de[key]}); they are not the same feature space")
    if dt["cls"].shape[-1] != de["cls"].shape[-1]:
        raise SystemExit("feature dimensions differ between the two feature files")
    n_layers = len(dt["cls_layers"])

    # cluster.py drops the training scenes whose stored (H, W) is not the majority shape --
    # DL3DVDataset would never load them, so clustering them only dilutes the samples drawn
    # from their cluster. The eval scenes are kept whatever their shape: they were actually
    # evaluated, so they have a loss to place on the sheet.
    hw = dt["image_hw"]
    shapes, counts = np.unique(hw, axis=0, return_counts=True)
    keep = np.ones(len(hw), bool) if args.no_hw_filter else (hw == shapes[counts.argmax()]).all(1)
    if not keep.all():
        print(f"keeping {keep.sum()} training scenes at "
              f"{tuple(int(v) for v in shapes[counts.argmax()])}; dropped {(~keep).sum()} "
              f"at mismatched sizes")

    n_train = int(keep.sum())
    cls = np.concatenate([dt["cls"][keep], de["cls"]])
    subsets = [str(s) for s in dt["subsets"][keep]] + [str(s) for s in de["subsets"]]
    scenes = [str(s) for s in dt["scenes"][keep]] + [str(s) for s in de["scenes"]]
    is_eval = np.zeros(len(cls), bool)
    is_eval[n_train:] = True
    print(f"clustering {n_train} training + {len(cls) - n_train} eval scenes jointly, "
          f"{n_layers} layer(s) x {cls.shape[-1] // n_layers} dims, "
          f"n_images {int(dt['n_images'])}/{int(de['n_images'])}")

    C, _ = centroids(cls, n_layers)
    del cls
    # Centroids of per-image unit vectors, deliberately not renormalized, so this dot product
    # is the average of the n_images^2 pairwise cosine similarities -- cluster.py's edge weight.
    sim = C @ C.T
    parts = partition(sim, args.method, args.threshold, args.resolution, args.seed)

    label = np.empty(len(sim), dtype=int)
    for c, part in enumerate(parts):
        label[part] = c

    # ---- typicality: mean cosine to the cluster's OTHER members ----
    # cluster.py's cohesion score, kept ascending there so a subset prefix gets the fringe.
    # Here the order is reversed -- the sheet reads centre first -- so a cluster's identity is
    # the first thing you see and its outliers trail off to the right. The diagonal is excluded
    # because sim[n, n] is the mean pairwise cosine among scene n's own sampled frames, not a
    # distance to anything in the cluster.
    typ = np.zeros(len(sim), dtype=np.float32)
    for part in parts:
        s = sim[np.ix_(part, part)].copy()
        np.fill_diagonal(s, 0.0)
        typ[part] = s.sum(1) / max(len(part) - 1, 1)  # max(): a singleton has no other members

    # ---- drift from the training-only partition ----
    if args.clusters is not None:
        cj = json.loads(args.clusters.read_text())
        old = {(s["subset"], s["scene"]): c["cluster"]
               for c in cj["clusters"] for s in c["scenes"]}
        shared = [i for i in range(n_train) if (subsets[i], scenes[i]) in old]
        if len(shared) < 2:
            print(f"warning: {args.clusters.name} shares no scenes with {args.train_features.name}")
        else:
            a = np.array([old[(subsets[i], scenes[i])] for i in shared])
            print(f"\nvs {args.clusters.name}: ARI {adjusted_rand(a, label[shared]):+.3f} over "
                  f"{len(shared)} shared training scenes, {len(cj['clusters'])} clusters -> "
                  f"{len(parts)} joint\n  training-only sizes "
                  f"{[c['size'] for c in cj['clusters']][:10]}\n  joint sizes "
                  f"{[len(p) for p in parts][:10]}")

    # ---- rows ----
    thumb_rel = Path(os.path.relpath(args.thumbs, out_html.parent)).as_posix()
    missing_val = 0
    rows = []
    for c, part in enumerate(parts):
        order = sorted(part, key=lambda n: -typ[n])  # most typical first
        tiles, vals = [], []
        for n in order:
            v = scene_val.get(scenes[n]) if is_eval[n] else None
            missing_val += is_eval[n] and v is None
            name = args.thumb_pattern.format(subset=subsets[n], scene=scenes[n])
            has = (args.thumbs / name).exists()
            tiles.append({"subset": subsets[n], "scene": scenes[n], "value": v,
                          "repeats": per_repeat.get(scenes[n], []) if is_eval[n] else [],
                          "kind": "eval" if is_eval[n] else "train", "typ": float(typ[n]),
                          "missing_thumb": not has,
                          "thumb": f"{thumb_rel}/{name}" if has else None})
            if v is not None:
                vals.append(v)
        ev = is_eval[order]
        n_eval = int(ev.sum())
        pair = sim[np.ix_(order, order)][np.triu_indices(len(order), 1)]
        # Mean typicality over *all* the members would be exactly `cohesion` (both are the
        # mean pairwise cosine), so split it by split instead: typ_eval well below typ_train
        # says the eval scenes joined this cluster from its fringe rather than its centre.
        rows.append({
            "cluster": c, "size": len(order), "n_train": len(order) - n_eval, "n_eval": n_eval,
            "cohesion": float(pair.mean()) if len(order) > 1 else None,
            "n": len(vals),
            "mean": sum(vals) / len(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None,
            "min": min(vals) if vals else None, "max": max(vals) if vals else None,
            "typ_train": float(typ[order][~ev].mean()) if n_eval < len(order) else None,
            "typ_eval": float(typ[order][ev].mean()) if n_eval else None,
            "scenes": tiles,
        })
    if missing_val:
        print(f"warning: {missing_val}/{int(is_eval.sum())} eval scenes have no {args.metric} "
              f"in {args.eval.name}")
    unscored = len(scene_val) - sum(1 for r in rows for s in r["scenes"] if s["value"] is not None)
    if unscored:
        print(f"warning: {unscored}/{len(scene_val)} scored scenes are absent from "
              f"{args.features.name} and do not appear on the sheet")
    missing_thumb = sum(s["missing_thumb"] for r in rows for s in r["scenes"])
    if missing_thumb:
        print(f"warning: {missing_thumb}/{len(sim)} scenes have no "
              f"{args.thumbs}/{args.thumb_pattern} thumbnail")

    allv = [s["value"] for r in rows for s in r["scenes"] if s["value"] is not None]
    vmin = args.vmin if args.vmin is not None else quantile(allv, args.clip[0])
    vmax = args.vmax if args.vmax is not None else quantile(allv, args.clip[1])
    span = (vmax - vmin) or 1.0
    overall = sum(allv) / len(allv)

    # ---- console summary ----
    scored = [r for r in rows if r["mean"] is not None]
    print(f"\n{len(parts)} joint clusters ({sum(r['size'] > 1 for r in rows)} non-singleton), "
          f"{sum(r['n_eval'] > 0 for r in rows)} containing eval scenes, "
          f"{sum(r['n_train'] == 0 for r in rows)} eval-only")
    print(f"overall mean {args.metric} {overall:.4f} over {len(allv)} eval scenes "
          f"(colour scale {vmin:.3f}-{vmax:.3f})")
    print(f"{'cluster':>8} {'size':>5} {'train':>6} {'eval':>5} {'mean':>8} {'median':>8} "
          f"{'std':>7} {'cohesion':>9} {'typ.tr':>7} {'typ.ev':>7}")

    def col(v, w=8, p=4):
        return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"

    for r in rows:
        print(f"{r['cluster']:>8} {r['size']:>5} {r['n_train']:>6} {r['n_eval']:>5} "
              f"{col(r['mean'])} {col(r['median'])} {col(r['std'], 7)} "
              f"{col(r['cohesion'], 9)} {col(r['typ_train'], 7)} {col(r['typ_eval'], 7)}")

    if len(scored) >= 3:
        print(f"\nspearman(cluster size, cluster mean {args.metric}) = "
              f"{spearman([r['size'] for r in scored], [r['mean'] for r in scored]):+.3f}, "
              f"spearman(cluster train count, mean {args.metric}) = "
              f"{spearman([r['n_train'] for r in scored], [r['mean'] for r in scored]):+.3f} "
              f"over {len(scored)} clusters")
    per_scene = [(r["size"], r["n_train"], s["typ"], s["value"])
                 for r in rows for s in r["scenes"] if s["value"] is not None]
    if len(per_scene) >= 3:
        sz, tn, tp, lv = zip(*per_scene)
        print(f"spearman(cluster size, scene {args.metric}) = {spearman(sz, lv):+.3f}, "
              f"spearman(training scenes in that cluster, scene {args.metric}) = "
              f"{spearman(tn, lv):+.3f}, spearman(typicality within cluster, scene "
              f"{args.metric}) = {spearman(tp, lv):+.3f} over {len(per_scene)} eval scenes")

    # ---- json, least-typical-first to match cluster.py/subset.py ----
    if args.out_json is not None:
        args.out_json.write_text(json.dumps({
            "method": args.method,
            "threshold": args.threshold,
            "resolution": args.resolution,
            "n_images": int(dt["n_images"]),
            "num_scenes": len(sim),
            "joint": {"train_features": str(args.train_features), "features": str(args.features),
                      "n_train": n_train, "n_eval": int(is_eval.sum())},
            "clusters": [{
                "cluster": r["cluster"], "size": r["size"], "cohesion": r["cohesion"],
                "n_train": r["n_train"], "n_eval": r["n_eval"],
                "scenes": [{"subset": s["subset"], "scene": s["scene"], "split": s["kind"],
                            "typicality": round(s["typ"], 6)}
                           for s in reversed(r["scenes"])],
            } for r in rows],
        }, indent=1))
        print(f"wrote {args.out_json} (least-typical-first, {len(rows)} clusters)")

    # ---- html ----
    def norm(v):
        return (v - vmin) / span

    worst_mean = max((r["mean"] for r in scored), default=1.0)
    biggest = max(r["size"] for r in rows)
    body = []
    for r in rows:
        head_col = ramp(norm(r["mean"])) if r["mean"] is not None else NO_DATA
        mean_txt = f"{r['mean']:.3f}" if r["mean"] is not None else "n/a"
        bar = (r["mean"] / worst_mean * 100) if r["mean"] is not None else 0
        cohesion = f", cos {r['cohesion']:.3f}" if r["cohesion"] is not None else ""
        body.append(
            f'<section class=cl data-mean="{r["mean"] if r["mean"] is not None else -1}" '
            f'data-cluster="{r["cluster"]}" data-size="{r["size"]}" data-neval="{r["n_eval"]}" '
            f'data-ntrain="{r["n_train"]}" data-frac="{r["n_eval"] / r["size"]:.4f}">'
            f'<h2><span class=sw style="background:{head_col}"></span>'
            f'cluster {r["cluster"]} <b>{mean_txt}</b> '
            f'<span class=dim>mean {args.metric} of {r["n_eval"]} eval &middot; '
            f'{r["n_train"]} train &middot; n={r["size"]}{cohesion}</span>'
            f'<span class=bar><i style="width:{bar:.1f}%;background:{head_col}"></i></span>'
            f'<span class=bar title="size, eval share lighter">'
            f'<i style="width:{r["n_train"] / biggest * 100:.1f}%;background:#4a4a46"></i>'
            f'<i style="width:{r["n_eval"] / biggest * 100:.1f}%;background:#8a8a82"></i>'
            f'</span></h2><div class=grid>')
        for k, s in enumerate(r["scenes"]):
            v, ev = s["value"], s["kind"] == "eval"
            col = ramp(norm(v)) if v is not None else NO_DATA
            txt = f"{v:.3f}" if v is not None else ("n/a" if ev else "train")
            reps = " ".join(f"{x:.3f}" for x in s["repeats"])
            tip = (f'{s["subset"]}/{s["scene"]}\n'
                   + (f'{args.metric} {txt}' if ev else "training scene, not evaluated")
                   + (f"\nrepeats {reps}" if reps else "")
                   + f'\ntypicality {s["typ"]:.4f} (rank {k + 1}/{r["size"]} in cluster '
                     f'{r["cluster"]})')
            img = (f'<img src="{_html.escape(s["thumb"])}" width={args.thumb_width} loading=lazy>'
                   if s["thumb"] else f'<div class=noimg style="width:{args.thumb_width}px"></div>')
            body.append(
                f'<figure class={"ev" if ev else "tr"} data-kind="{s["kind"]}" '
                f'data-v="{v if v is not None else -1}" data-typ="{s["typ"]:.6f}" '
                f'title="{_html.escape(tip)}" style="--c:{col}">{img}'
                f'<figcaption><span class=dot></span><span class=v>{txt}</span>'
                f'<span class=dim>{r["cluster"]}.{k} {s["subset"]}/{s["scene"][:10]}</span>'
                f'</figcaption></figure>')
        body.append("</div></section>")

    # Header table: size, composition and mean loss in one sorted list, rather than by
    # scrolling the sections.
    def cell(v, p=4):
        return "-" if v is None else f"{v:.{p}f}"

    srows = "".join(
        f'<tr><td>{r["cluster"]}</td><td>{r["size"]}</td><td>{r["n_train"]}</td>'
        f'<td>{r["n_eval"]}</td><td>{cell(r["mean"])}</td><td>{cell(r["median"])}</td>'
        f'<td>{cell(r["cohesion"], 3)}</td><td>{cell(r["typ_train"], 3)}</td>'
        f'<td>{cell(r["typ_eval"], 3)}</td>'
        f'<td><span class=tbar><i style="width:'
        f'{(r["mean"] / worst_mean * 100) if r["mean"] is not None else 0:.1f}%;'
        f'background:{ramp(norm(r["mean"])) if r["mean"] is not None else NO_DATA}"></i></span></td>'
        f'</tr>' for r in rows)

    legend = "".join(f'<i style="background:{c}"></i>' for c in RAMP)
    doc = f"""<!doctype html><meta charset=utf-8>
<title>{args.metric} by joint cluster &mdash; {_html.escape(args.features.name)}</title>
<style>
body{{background:#111;color:#ddd;font:13px system-ui;margin:0;padding:16px}}
header{{position:sticky;top:0;background:#111;padding:8px 0 10px;z-index:2;border-bottom:1px solid #222}}
h1{{font-size:15px;font-weight:600;margin:0 0 6px}}
.dim{{color:#888;font-weight:400}}
.controls{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:8px}}
label{{color:#888;display:flex;gap:5px;align-items:center}}
select{{background:#1c1c1b;color:#ddd;border:1px solid #333;border-radius:4px;padding:3px 6px;font:12px system-ui}}
.legend{{display:flex;gap:6px;align-items:center;color:#888;font:11px ui-monospace}}
.legend .ramp{{display:flex;height:10px;width:160px;border-radius:2px;overflow:hidden}}
.legend i{{flex:1}}
details{{margin-top:8px}}
summary{{color:#888;cursor:pointer}}
table{{border-collapse:collapse;font:11px ui-monospace;margin-top:6px}}
th,td{{text-align:right;padding:2px 8px;border-bottom:1px solid #1e1e1d}}
th{{color:#888;font-weight:400}}
.tbar{{display:block;width:120px;height:6px;background:#1c1c1b;border-radius:3px;overflow:hidden}}
.tbar i{{display:block;height:100%}}
h2{{font-size:13px;font-weight:600;margin:20px 0 8px;display:flex;gap:8px;align-items:center}}
h2 b{{font-family:ui-monospace;font-weight:600}}
.sw{{width:10px;height:10px;border-radius:2px;flex:none}}
.bar{{flex:1;max-width:220px;height:6px;background:#1c1c1b;border-radius:3px;overflow:hidden;display:flex}}
.bar i{{display:block;height:100%}}
.grid{{display:flex;flex-wrap:wrap;gap:6px}}
figure{{margin:0;max-width:{args.thumb_width}px}}
figure img,.noimg{{display:block;border-radius:3px;box-shadow:0 0 0 3px var(--c)}}
.noimg{{height:90px;background:#1a1a19}}
.tr img,.tr .noimg{{opacity:.5}}
figcaption{{font:10px ui-monospace;display:flex;gap:4px;align-items:center;margin-top:5px;overflow:hidden;white-space:nowrap}}
.dot{{width:8px;height:8px;border-radius:2px;background:var(--c);flex:none}}
figcaption .v{{color:#ddd;font-weight:600}}
.tr figcaption .v{{color:#777;font-weight:400}}
.hide{{display:none}}
</style>
<header>
<h1>{args.metric} by joint cluster <span class=dim>&mdash;
{_html.escape(args.train_features.name)} + {_html.escape(args.features.name)}
&times; {_html.escape(args.eval.name)}</span></h1>
<div class=dim>{len(allv)} scored eval scenes &middot; {n_train} training scenes &middot;
{len(rows)} clusters re-detected over the union ({args.method}, resolution {args.resolution:g})
&middot; overall mean {overall:.4f} &middot; median {statistics.median(allv):.4f} &middot;
range {min(allv):.3f}&ndash;{max(allv):.3f} &middot; scenes ordered most typical first</div>
<div class=controls>
<label>clusters <select id=sc>
<option value=size>size</option><option value=mean-desc>worst mean first</option>
<option value=mean-asc>best mean first</option><option value=neval>eval count</option>
<option value=frac>eval share</option><option value=id>cluster id</option></select></label>
<label>images <select id=si>
<option value=orig>most typical first</option><option value=typ-asc>least typical first</option>
<option value=desc>worst first</option><option value=asc>best first</option></select></label>
<label>show <select id=sk>
<option value=all>eval + train</option><option value=eval>eval only</option>
<option value=train>train only</option></select></label>
<label>min cluster size <select id=ms>
<option value=1>1</option><option value=2>2</option><option value=5>5</option>
<option value=10>10</option></select></label>
<div class=legend><span>{vmin:.2f}</span><span class=ramp>{legend}</span><span>{vmax:.2f}</span>
<span>{args.metric} (clipped)</span></div>
</div>
<details><summary>cluster table</summary>
<table><tr><th>cluster</th><th>size</th><th>train</th><th>eval</th><th>mean</th><th>median</th>
<th>cohesion</th><th>typ.tr</th><th>typ.ev</th><th>mean {args.metric}</th></tr>
{srows}</table></details>
</header>
<main id=m>
{chr(10).join(body)}
</main>
<script>
const m = document.getElementById('m');
const secs = [...m.children];
secs.forEach((s, i) => s.dataset.orig = i);
const num = (el, k) => parseFloat(el.dataset[k]);
function sortClusters() {{
  const key = {{'mean-desc': s => -num(s, 'mean'),
                'mean-asc': s => num(s, 'mean') < 0 ? 1e9 : num(s, 'mean'),
                'id': s => num(s, 'cluster'),
                'neval': s => -num(s, 'neval'),
                'frac': s => -num(s, 'frac'),
                'size': s => -num(s, 'size')}}[sc.value];
  [...secs].sort((a, b) => key(a) - key(b) || num(a, 'cluster') - num(b, 'cluster'))
           .forEach(s => m.appendChild(s));
}}
function sortImages() {{
  const key = {{'orig': f => num(f, 'orig'),
                'typ-asc': f => num(f, 'typ'),
                'desc': f => -num(f, 'v'),
                'asc': f => num(f, 'v') < 0 ? 1e9 : num(f, 'v')}}[si.value];
  for (const s of secs) {{
    const g = s.querySelector('.grid'), figs = [...g.children];
    figs.forEach((f, i) => f.dataset.orig ??= i);
    figs.sort((a, b) => key(a) - key(b)).forEach(f => g.appendChild(f));
  }}
}}
function filterClusters() {{
  secs.forEach(s => s.classList.toggle('hide', num(s, 'size') < +ms.value));
}}
function filterKind() {{
  for (const f of m.querySelectorAll('figure'))
    f.classList.toggle('hide', sk.value !== 'all' && f.dataset.kind !== sk.value);
}}
sc.onchange = sortClusters; si.onchange = sortImages;
ms.onchange = filterClusters; sk.onchange = filterKind;
sortClusters();
</script>
"""
    out_html.write_text(doc)
    print(f"\nwrote {out_html}")


if __name__ == "__main__":
    main()
