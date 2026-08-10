"""Assign every eval scene to its nearest *training* cluster, and render a contact sheet.

`cluster.py` partitions the training scenes; this script takes that partition as fixed and
drops the eval scenes into it: each eval scene joins the cluster it is closest to, scored
against the cluster's mean member centroid. Two scorings, `--score`:

* `mean` -- mean over the cluster's training scenes of the average pairwise cosine similarity
  between the two scenes' N_IMAGES CLS vectors, i.e. exactly `cluster.py`'s edge weight
  averaged over the members. Equals a dot product against the (unnormalized) mean centroid,
  so the whole assignment is one (n_eval x dim) @ (dim x n_clusters) matmul.
* `unit` (default) -- the same, but against the L2-normalized mean centroid, i.e. the cosine
  to the cluster's *direction*.

`mean` is the obvious reading of "average the cosine similarities", but it is biased towards
small clusters: the mean of unit-ish centroids is short when a cluster is internally diverse,
and a singleton cluster keeps its full length, so tiny clusters outscore big ones on scenes
that belong to neither. On this data that bias is not subtle -- feeding the *training* scenes
back in, `mean` returns only ~71% of them to their own cluster and hands 454 of 4288 to a
one-scene cluster, while `unit` returns ~84% and roughly reproduces the true size
distribution. Since the question being asked here is "do the low-loss scenes land in the big
clusters?", a scorer that inflates small clusters is the one thing that must not be the
default. Every run prints that training-scene self-recovery so the choice stays checkable.

Writes `<features>_nearest_<metric>.html` -- `get_cluster_loss_html.py`'s grid, but each
cluster now holds its training scenes (grey, no loss -- they were never evaluated) followed by
the eval scenes assigned to it (coloured by loss). The point is the question "do the
low-loss eval images land in the big clusters?", so the console summary and the header
table pair each cluster's training size against the mean loss of the eval scenes that chose it.

    ~/clustering-imgs/.venv/bin/python clustering/get_nearest_cluster_html.py
    ... --features clustering/6k_features_layers.npz --metric loss_depth

Both npz files must come from the same `extract_features.py` settings (same --layers and
--final-ln); the eval features are de-meaned with the *training* per-layer mean, since that
is the space the clusters were found in.
"""

import argparse
import html as _html
import json
import os
import statistics
from pathlib import Path

import numpy as np

# The colour ramp, its interpolation and the quantile helper are shared with the
# loss-by-cluster sheet on purpose: the two pages get compared side by side, so a given
# colour has to mean the same loss in both.
from get_cluster_loss_html import NO_DATA, RAMP, quantile, ramp

HERE = Path(__file__).parent


def centroids(cls, n_layers, layer_mean=None):
    """cluster.py's per-image feature transform, reduced to one centroid per scene.

    `cls` is (N, n_images, n_layers * dim) as written by extract_features.py, i.e. already
    L2-normalized per layer. Reproduces cluster.py: subtract each layer's dataset-mean
    direction (the near-constant "DC" component that otherwise dominates the combined cosine
    similarity), re-normalize that layer, then re-normalize the concatenated vector and
    average the scene's images. The centroid is deliberately *not* re-normalized -- the dot
    product of two centroids is then the average of the n_images^2 pairwise cosine
    similarities, which is the quantity cluster.py partitioned on.

    Pass `layer_mean` to de-mean with another run's means (eval features must use the
    training means to land in the same space); with None the means are taken from `cls`
    itself and returned alongside the centroids. Single-layer features skip de-meaning
    entirely, as cluster.py does.
    """
    n, n_images, total = cls.shape
    dim = total // n_layers
    out = np.empty((n, total), dtype=np.float32)
    means = []
    # Layer by layer: a float32 copy of one layer's slice is n_layers times smaller than a
    # copy of the whole feature block, and the concatenated vector's layer-k slice depends
    # on layer k alone, so the centroid can be filled in slices.
    for k in range(n_layers):
        sl = slice(k * dim, (k + 1) * dim)
        seg = cls[:, :, sl].astype(np.float32).reshape(-1, dim)
        if n_layers > 1:
            mu = seg.mean(axis=0) if layer_mean is None else layer_mean[k]
            means.append(mu)
            seg = seg - mu
            seg /= np.linalg.norm(seg, axis=-1, keepdims=True)
        out[:, sl] = seg.reshape(n, n_images, dim).mean(axis=1)
    # Each layer slice of a per-image vector is unit norm, so the full vector's norm is a
    # constant sqrt(n_layers); dividing the averaged slices by it == cluster.py's final
    # per-image renormalize followed by the average.
    if n_layers > 1:
        out /= np.sqrt(n_layers)
    return (out, means) if layer_mean is None else out


def spearman(a, b):
    """Rank correlation, ties averaged. Small n here, so no need for anything clever."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        for v in np.unique(x):  # average the ranks of each tied group
            m = x == v
            r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=HERE / "6k_features_layers.npz",
                    help="eval-set features from extract_features.py")
    ap.add_argument("--train-features", type=Path, default=HERE / "train_features_layers.npz",
                    help="training-set features the clusters were built from")
    ap.add_argument("--clusters", type=Path, default=HERE / "train_features_layers_clusters",
                    help="clusters json written by cluster.py over --train-features")
    ap.add_argument("--eval", type=Path,
                    default=HERE.parent / "runs/small-v5_default/eval/windows-small-v5-latest.jsonl",
                    help="eval jsonl; one row per scene per repeat")
    ap.add_argument("--metric", default="loss",
                    help="jsonl field to average (loss, loss_depth, abs_rel, ...)")
    ap.add_argument("--score", choices=("unit", "mean"), default="unit",
                    help="cosine to the cluster's unit-normalized mean centroid (unit, default) "
                         "or the plain mean cosine similarity to its members (mean, biased "
                         "towards small clusters -- see the module docstring)")
    ap.add_argument("--thumbs", type=Path, default=HERE / "thumbs")
    ap.add_argument("--thumb-pattern", default="{subset}_{scene}.jpg",
                    help="thumbnail filename, formatted with {subset} and {scene}")
    ap.add_argument("--thumb-width", type=int, default=160)
    ap.add_argument("--out", type=Path, default=None,
                    help="output html (default <features>_nearest_<metric>.html)")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"), default=(0.02, 0.98),
                    help="quantiles the colour scale saturates at")
    ap.add_argument("--vmin", type=float, default=None, help="override colour-scale minimum")
    ap.add_argument("--vmax", type=float, default=None, help="override colour-scale maximum")
    args = ap.parse_args()

    out_html = args.out or args.features.with_name(f"{args.features.stem}_nearest_{args.metric}.html")
    for p in (args.features, args.train_features, args.clusters, args.eval):
        if not p.exists():
            raise SystemExit(f"{p} does not exist")

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
    cj = json.loads(args.clusters.read_text())
    dt = np.load(args.train_features)
    de = np.load(args.features)
    for key in ("cls_layers", "final_ln"):
        if not np.array_equal(dt[key], de[key]):
            raise SystemExit(f"{key} differs between the two feature files "
                             f"({dt[key]} vs {de[key]}); they are not the same feature space")
    if dt["cls"].shape[-1] != de["cls"].shape[-1]:
        raise SystemExit("feature dimensions differ between the two feature files")

    n_layers = len(dt["cls_layers"])

    # cluster.py drops the scenes whose stored (H, W) is not the majority shape, so the
    # clusters json -- not the npz -- is the authority on which training scenes are in the
    # space, and the per-layer means have to be taken over exactly those.
    row = {(str(s), str(c)): i for i, (s, c) in enumerate(zip(dt["subsets"], dt["scenes"]))}
    idx_of = {}
    for c in cj["clusters"]:
        for s in c["scenes"]:
            k = (s["subset"], s["scene"])
            if k not in row:
                raise SystemExit(f"clustered scene {k[0]}/{k[1]} is not in {args.train_features}")
            idx_of[k] = row[k]
    keep = np.array(sorted(idx_of.values()))
    pos = {r: p for p, r in enumerate(keep)}  # npz row -> row in the transformed matrix
    train_c, layer_mean = centroids(dt["cls"][keep], n_layers)
    eval_c = centroids(de["cls"], n_layers, layer_mean)
    print(f"{len(keep)} training scenes in {len(cj['clusters'])} clusters, "
          f"{len(eval_c)} eval scenes, {n_layers} layer(s) x "
          f"{dt['cls'].shape[-1] // n_layers} dims, n_images {int(dt['n_images'])}/{int(de['n_images'])}")

    # ---- assignment ----
    # The mean similarity to a cluster's members is a dot product against the mean of their
    # centroids, so all n_clusters scores for all eval scenes are one matmul.
    members = [np.array([pos[idx_of[(s["subset"], s["scene"])]] for s in c["scenes"]])
               for c in cj["clusters"]]
    C = np.stack([train_c[m].mean(axis=0) for m in members])
    if args.score == "unit":
        C /= np.linalg.norm(C, axis=-1, keepdims=True)
    scores = eval_c @ C.T                        # (n_eval, n_clusters)

    # Same assignment run on the training scenes, whose cluster is known: how often it comes
    # back is the one cheap check on whether these assignments mean anything. Not 100% even in
    # principle -- Leiden optimizes modularity over a sparsified graph, not distance to a
    # centroid -- but a low number, or a predicted size profile unlike the true one, says the
    # scoring is the wrong shape for this partition.
    label = np.empty(len(train_c), dtype=int)
    for c, m in enumerate(members):
        label[m] = c
    pred = (train_c @ C.T).argmax(axis=1)
    print(f"training-scene self-recovery {(pred == label).mean():.3f} under --score {args.score} "
          f"({len(train_c)} scenes)")
    sizes, pred_sizes = np.bincount(label, minlength=len(C)), np.bincount(pred, minlength=len(C))
    worst = np.abs(pred_sizes - sizes).argmax()
    print(f"  predicted cluster sizes {pred_sizes.tolist()} vs true {sizes.tolist()} "
          f"(largest drift: cluster {worst}, {sizes[worst]} -> {pred_sizes[worst]})")
    best = scores.argmax(axis=1)
    order = np.argsort(-scores, axis=1)
    top = scores[np.arange(len(scores)), order[:, 0]]
    second = scores[np.arange(len(scores)), order[:, 1]] if scores.shape[1] > 1 else top

    assigned = [[] for _ in cj["clusters"]]
    thumb_rel = Path(os.path.relpath(args.thumbs, out_html.parent)).as_posix()

    def tile(subset, scene, value, repeats, kind, sim=None, margin=None):
        name = args.thumb_pattern.format(subset=subset, scene=scene)
        has = (args.thumbs / name).exists()
        return {"subset": subset, "scene": scene, "value": value, "repeats": repeats,
                "kind": kind, "sim": sim, "margin": margin, "missing_thumb": not has,
                "thumb": f"{thumb_rel}/{name}" if has else None}

    missing_val = 0
    for i, (sub, sc) in enumerate(zip(de["subsets"], de["scenes"])):
        sub, sc = str(sub), str(sc)
        v = scene_val.get(sc)
        missing_val += v is None
        assigned[best[i]].append(tile(sub, sc, v, per_repeat.get(sc, []), "eval",
                                     float(top[i]), float(top[i] - second[i])))
    if missing_val:
        print(f"warning: {missing_val}/{len(eval_c)} eval scenes have no {args.metric} "
              f"in {args.eval.name}")
    unscored = len(scene_val) - sum(1 for c in assigned for s in c if s["value"] is not None)
    if unscored:
        print(f"warning: {unscored}/{len(scene_val)} scored scenes are absent from "
              f"{args.features.name} and do not appear on the sheet")

    # Most-typical-first inside each cluster, mirroring cluster.py's cohesion ordering of the
    # training scenes (which the json already stores in that order).
    for a in assigned:
        a.sort(key=lambda s: -s["sim"])

    rows = []
    for c, cl in enumerate(cj["clusters"]):
        ev = assigned[c]
        vals = [s["value"] for s in ev if s["value"] is not None]
        tr = [tile(s["subset"], s["scene"], None, [], "train") for s in cl["scenes"]]
        rows.append({
            "cluster": cl["cluster"], "size": cl["size"], "cohesion": cl.get("cohesion"),
            "n_eval": len(ev), "n": len(vals),
            "mean": sum(vals) / len(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None,
            "min": min(vals) if vals else None, "max": max(vals) if vals else None,
            "sim": sum(s["sim"] for s in ev) / len(ev) if ev else None,
            "scenes": ev + tr,
        })
    missing_thumb = sum(s["missing_thumb"] for r in rows for s in r["scenes"])
    if missing_thumb:
        print(f"warning: {missing_thumb}/{sum(len(r['scenes']) for r in rows)} scenes have no "
              f"{args.thumbs}/{args.thumb_pattern} thumbnail")

    allv = [s["value"] for r in rows for s in r["scenes"] if s["value"] is not None]
    vmin = args.vmin if args.vmin is not None else quantile(allv, args.clip[0])
    vmax = args.vmax if args.vmax is not None else quantile(allv, args.clip[1])
    span = (vmax - vmin) or 1.0
    overall = sum(allv) / len(allv)

    # ---- console summary: the size-vs-loss question the sheet exists to answer ----
    scored = [r for r in rows if r["mean"] is not None]
    print(f"\noverall mean {args.metric} {overall:.4f} over {len(allv)} eval scenes, "
          f"{len(scored)}/{len(rows)} clusters received any (colour scale {vmin:.3f}-{vmax:.3f})")
    print(f"{'cluster':>8} {'train':>6} {'eval':>5} {'mean':>8} {'median':>8} {'std':>7} "
          f"{'cohesion':>9} {'sim':>7} {'margin':>7}")
    def col(v, w=8, p=4):
        return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"

    for r in sorted(rows, key=lambda r: -r["size"]):
        ev = [s for s in r["scenes"] if s["kind"] == "eval"]
        mg = sum(s["margin"] for s in ev) / len(ev) if ev else float("nan")
        print(f"{r['cluster']:>8} {r['size']:>6} {r['n_eval']:>5} {col(r['mean'])} "
              f"{col(r['median'])} {col(r['std'], 7)} {col(r['cohesion'], 9)} {col(r['sim'], 7)} "
              f"{mg:>7.4f}")

    if len(scored) >= 3:
        print(f"\nspearman(cluster train size, cluster mean {args.metric}) = "
              f"{spearman([r['size'] for r in scored], [r['mean'] for r in scored]):+.3f} "
              f"over {len(scored)} clusters")
    per_scene = [(r["size"], s["value"], s["sim"])
                 for r in rows for s in r["scenes"] if s["kind"] == "eval" and s["value"] is not None]
    if len(per_scene) >= 3:
        sz, lv, sm = zip(*per_scene)
        print(f"spearman(assigned cluster size, scene {args.metric}) = {spearman(sz, lv):+.3f}, "
              f"spearman(similarity to that cluster, scene {args.metric}) = {spearman(sm, lv):+.3f} "
              f"over {len(per_scene)} scenes")

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
            f'data-cluster="{r["cluster"]}" data-size="{r["size"]}" data-neval="{r["n_eval"]}">'
            f'<h2><span class=sw style="background:{head_col}"></span>'
            f'cluster {r["cluster"]} <b>{mean_txt}</b> '
            f'<span class=dim>mean {args.metric} of {r["n_eval"]} eval &middot; '
            f'{r["size"]} train{cohesion}</span>'
            f'<span class=bar><i style="width:{bar:.1f}%;background:{head_col}"></i></span>'
            f'<span class=bar title="train size"><i style="width:{r["size"] / biggest * 100:.1f}%;'
            f'background:#4a4a46"></i></span></h2>'
            f'<div class=grid>')
        for k, s in enumerate(r["scenes"]):
            v, ev = s["value"], s["kind"] == "eval"
            col = ramp(norm(v)) if v is not None else NO_DATA
            txt = f"{v:.3f}" if v is not None else ("n/a" if ev else "train")
            reps = " ".join(f"{x:.3f}" for x in s["repeats"])
            tip = (f'{s["subset"]}/{s["scene"]}\n'
                   + (f'{args.metric} {txt}' if ev else "training scene, not evaluated")
                   + (f"\nrepeats {reps}" if reps else "")
                   + (f'\ncos to cluster {r["cluster"]} {s["sim"]:.4f} '
                      f'(margin over next {s["margin"]:+.4f})' if ev else ""))
            img = (f'<img src="{_html.escape(s["thumb"])}" width={args.thumb_width} loading=lazy>'
                   if s["thumb"] else f'<div class=noimg style="width:{args.thumb_width}px"></div>')
            body.append(
                f'<figure class={"ev" if ev else "tr"} data-kind="{s["kind"]}" '
                f'data-v="{v if v is not None else -1}" '
                f'data-sim="{s["sim"] if s["sim"] is not None else -1}" '
                f'title="{_html.escape(tip)}" style="--c:{col}">{img}'
                f'<figcaption><span class=dot></span><span class=v>{txt}</span>'
                f'<span class=dim>{r["cluster"]}.{k} {s["subset"]}/{s["scene"][:10]}</span>'
                f'</figcaption></figure>')
        body.append("</div></section>")

    # Header table: the whole point of the sheet is cluster size against mean loss, and that
    # comparison is easier in one sorted list than by scrolling 13 sections.
    def cell(v, p=4):
        return "-" if v is None else f"{v:.{p}f}"

    srows = "".join(
        f'<tr><td>{r["cluster"]}</td><td>{r["size"]}</td><td>{r["n_eval"]}</td>'
        f'<td>{cell(r["mean"])}</td><td>{cell(r["median"])}</td>'
        f'<td>{cell(r["cohesion"], 3)}</td><td>{cell(r["sim"])}</td>'
        f'<td><span class=tbar><i style="width:'
        f'{(r["mean"] / worst_mean * 100) if r["mean"] is not None else 0:.1f}%;'
        f'background:{ramp(norm(r["mean"])) if r["mean"] is not None else NO_DATA}"></i></span></td>'
        f'</tr>' for r in sorted(rows, key=lambda r: -r["size"]))

    legend = "".join(f'<i style="background:{c}"></i>' for c in RAMP)
    doc = f"""<!doctype html><meta charset=utf-8>
<title>{args.metric} by nearest cluster &mdash; {_html.escape(args.features.name)}</title>
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
.bar{{flex:1;max-width:220px;height:6px;background:#1c1c1b;border-radius:3px;overflow:hidden}}
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
<h1>{args.metric} by nearest training cluster <span class=dim>&mdash;
{_html.escape(args.features.name)} &rarr; {_html.escape(args.clusters.name)}
&times; {_html.escape(args.eval.name)}</span></h1>
<div class=dim>{len(allv)} scored eval scenes &middot; {sum(r["size"] for r in rows)} training
scenes &middot; {len(rows)} clusters &middot; overall mean {overall:.4f} &middot; median
{statistics.median(allv):.4f} &middot; range {min(allv):.3f}&ndash;{max(allv):.3f} &middot;
scored by {"cosine to the unit cluster centroid" if args.score == "unit"
           else "mean cosine to the cluster's members"} &middot; self-recovery
{(pred == label).mean():.3f}</div>
<div class=controls>
<label>clusters <select id=sc>
<option value=size>train size</option>
<option value=mean-desc>worst mean first</option><option value=mean-asc>best mean first</option>
<option value=neval>eval count</option><option value=id>cluster id</option></select></label>
<label>images <select id=si>
<option value=orig>eval first, most typical</option><option value=desc>worst first</option>
<option value=asc>best first</option><option value=sim>most typical first</option></select></label>
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
<table><tr><th>cluster</th><th>train</th><th>eval</th><th>mean</th><th>median</th>
<th>cohesion</th><th>cos</th><th>mean {args.metric}</th></tr>
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
                'size': s => -num(s, 'size')}}[sc.value];
  [...secs].sort((a, b) => key(a) - key(b) || num(a, 'cluster') - num(b, 'cluster'))
           .forEach(s => m.appendChild(s));
}}
function sortImages() {{
  const key = {{'orig': f => num(f, 'orig'),
                'desc': f => -num(f, 'v'),
                'asc': f => num(f, 'v') < 0 ? 1e9 : num(f, 'v'),
                'sim': f => -num(f, 'sim')}}[si.value];
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
