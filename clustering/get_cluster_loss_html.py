"""Average an eval run's per-scene loss over each cluster, and render a contact sheet.

Joins a clusters json written by `cluster.py` against an eval jsonl (one row per scene per
repeat, as `runs/<run>/eval/windows-*.jsonl`), averages the metric over each scene's repeats
and then over each cluster's scenes, and writes:

`<clusters>_<metric>.html` -- the same thumbnail grid as `cluster.py`'s contact sheet, but
every thumbnail carries its scene's loss (colour + number) and every cluster header its mean,
with client-side sorting so the worst clusters/scenes can be pulled to the top.

    python3 clustering/cluster_loss.py --clusters clustering/6k_clusters_layered.json
    python3 clustering/cluster_loss.py --clusters clustering/clusters.json --metric loss_depth

Thumbnails are looked up by name -- `thumbs/{subset}_{scene}.jpg`, the layout the thumbs dir
uses -- so no feature npz is needed and any clusters json over any subset works. Scenes with
no thumbnail on disk render as text tiles; `--thumb-pattern` overrides the naming.
"""

import argparse
import html as _html
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

# Sequential ramp, one hue (blue 700 -> 100), for magnitude. On the dark surface the darkest
# step is the one allowed to recede, so low = dark, high = light.
RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
        "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
NO_DATA = "#3a3a38"


def ramp(t):
    """t in [0, 1] -> hex, linearly interpolated between ramp steps in sRGB."""
    t = min(max(t, 0.0), 1.0) * (len(RAMP) - 1)
    lo = min(int(t), len(RAMP) - 2)
    f = t - lo
    a, b = RAMP[lo], RAMP[lo + 1]
    return "#%02x%02x%02x" % tuple(
        round(int(a[1 + 2 * k:3 + 2 * k], 16) * (1 - f) + int(b[1 + 2 * k:3 + 2 * k], 16) * f)
        for k in range(3))


def quantile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = p * (len(xs) - 1)
    lo = int(i)
    return xs[lo] if lo + 1 >= len(xs) else xs[lo] + (xs[lo + 1] - xs[lo]) * (i - lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=Path, default=HERE / "6k_clusters_layered.json",
                    help="clusters json written by cluster.py")
    ap.add_argument("--eval", type=Path,
                    default=HERE.parent / "runs/small-v5_default/eval/windows-small-v5-latest.jsonl",
                    help="eval jsonl; one row per scene per repeat")
    ap.add_argument("--metric", default="loss",
                    help="jsonl field to average (loss, loss_depth, abs_rel, ...)")
    ap.add_argument("--thumbs", type=Path, default=HERE / "thumbs")
    ap.add_argument("--thumb-pattern", default="{subset}_{scene}.jpg",
                    help="thumbnail filename, formatted with {subset} and {scene}")
    ap.add_argument("--thumb-width", type=int, default=160)
    ap.add_argument("--out", type=Path, default=None, help="output html (default <clusters>_<metric>.html)")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"), default=(0.02, 0.98),
                    help="quantiles the colour scale saturates at")
    ap.add_argument("--vmin", type=float, default=None, help="override colour-scale minimum")
    ap.add_argument("--vmax", type=float, default=None, help="override colour-scale maximum")
    args = ap.parse_args()

    out_html = args.out or args.clusters.with_name(f"{args.clusters.stem}_{args.metric}.html")

    # ---- per-scene metric, averaged over repeats ----
    per_repeat = defaultdict(list)
    for line in args.eval.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if args.metric in r:
                per_repeat[r["scene_id"]].append(float(r[args.metric]))
    if not per_repeat:
        raise SystemExit(f"no rows with field {args.metric!r} in {args.eval}")
    scene_val = {s: sum(v) / len(v) for s, v in per_repeat.items()}
    print(f"{len(scene_val)} scenes x {min(map(len, per_repeat.values()))}-"
          f"{max(map(len, per_repeat.values()))} repeats from {args.eval}")

    # ---- cluster aggregation ----
    cj = json.loads(args.clusters.read_text())
    thumb_rel = Path(os.path.relpath(args.thumbs, out_html.parent)).as_posix()
    rows, missing_val, missing_thumb = [], 0, 0
    for c in cj["clusters"]:
        vals, scenes = [], []
        for s in c["scenes"]:
            v = scene_val.get(s["scene"])
            missing_val += v is None
            name = args.thumb_pattern.format(subset=s["subset"], scene=s["scene"])
            has_thumb = (args.thumbs / name).exists()
            missing_thumb += not has_thumb
            scenes.append({"subset": s["subset"], "scene": s["scene"], "value": v,
                           "repeats": per_repeat.get(s["scene"], []),
                           "thumb": f"{thumb_rel}/{name}" if has_thumb else None})
            if v is not None:
                vals.append(v)
        rows.append({
            "cluster": c["cluster"], "size": c["size"], "cohesion": c.get("cohesion"),
            "n": len(vals),
            "mean": sum(vals) / len(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None,
            "min": min(vals) if vals else None, "max": max(vals) if vals else None,
            "scenes": scenes,
        })
    n_scenes = sum(r["size"] for r in rows)
    if missing_val:
        print(f"warning: {missing_val}/{n_scenes} clustered scenes have no {args.metric} in the eval file")
    if missing_thumb:
        print(f"warning: {missing_thumb}/{n_scenes} clustered scenes have no "
              f"{args.thumbs}/{args.thumb_pattern} thumbnail")

    allv = list(scene_val.values())
    vmin = args.vmin if args.vmin is not None else quantile(allv, args.clip[0])
    vmax = args.vmax if args.vmax is not None else quantile(allv, args.clip[1])
    span = (vmax - vmin) or 1.0
    overall = sum(allv) / len(allv)

    # ---- console summary ----
    scored = sorted((r for r in rows if r["mean"] is not None), key=lambda r: -r["mean"])
    print(f"\noverall mean {args.metric} {overall:.4f} over {len(allv)} scenes, "
          f"{len(scored)}/{len(rows)} clusters scored (colour scale {vmin:.3f}-{vmax:.3f})")
    if scored:
        print(f"{'cluster':>8} {'n':>4} {'mean':>8} {'median':>8} {'std':>7}")
        for r in scored[:10]:
            print(f"{r['cluster']:>8} {r['n']:>4} {r['mean']:>8.4f} {r['median']:>8.4f} {r['std']:>7.4f}")
        if len(scored) > 15:
            print(f"{'...':>8}")
            for r in scored[-5:]:
                print(f"{r['cluster']:>8} {r['n']:>4} {r['mean']:>8.4f} {r['median']:>8.4f} {r['std']:>7.4f}")

    # ---- html ----
    def norm(v):
        return (v - vmin) / span

    worst_mean = max((r["mean"] for r in scored), default=1.0)
    body = []
    for r in rows:
        head_col = ramp(norm(r["mean"])) if r["mean"] is not None else NO_DATA
        mean_txt = f"{r['mean']:.3f}" if r["mean"] is not None else "n/a"
        bar = (r["mean"] / worst_mean * 100) if r["mean"] is not None else 0
        cohesion = f", cos {r['cohesion']:.3f}" if r["cohesion"] is not None else ""
        body.append(
            f'<section class=cl data-mean="{r["mean"] if r["mean"] is not None else -1}" '
            f'data-cluster="{r["cluster"]}" data-size="{r["size"]}">'
            f'<h2><span class=sw style="background:{head_col}"></span>'
            f'cluster {r["cluster"]} <b>{mean_txt}</b> '
            f'<span class=dim>mean {args.metric} &middot; n={r["size"]}{cohesion}</span>'
            f'<span class=bar><i style="width:{bar:.1f}%;background:{head_col}"></i></span></h2>'
            f'<div class=grid>')
        for k, s in enumerate(r["scenes"]):
            v = s["value"]
            col = ramp(norm(v)) if v is not None else NO_DATA
            txt = f"{v:.3f}" if v is not None else "n/a"
            reps = " ".join(f"{x:.3f}" for x in s["repeats"])
            tip = (f'{s["subset"]}/{s["scene"]}\n{args.metric} {txt}'
                   + (f"\nrepeats {reps}" if reps else ""))
            img = (f'<img src="{_html.escape(s["thumb"])}" width={args.thumb_width} loading=lazy>'
                   if s["thumb"] else f'<div class=noimg style="width:{args.thumb_width}px"></div>')
            body.append(
                f'<figure data-v="{v if v is not None else -1}" title="{_html.escape(tip)}" '
                f'style="--c:{col}">{img}'
                f'<figcaption><span class=dot></span><span class=v>{txt}</span>'
                f'<span class=dim>{r["cluster"]}.{k} {s["subset"]}/{s["scene"][:10]}</span>'
                f'</figcaption></figure>')
        body.append("</div></section>")

    legend = "".join(f'<i style="background:{c}"></i>' for c in RAMP)
    doc = f"""<!doctype html><meta charset=utf-8>
<title>{args.metric} by cluster &mdash; {_html.escape(args.clusters.name)}</title>
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
h2{{font-size:13px;font-weight:600;margin:20px 0 8px;display:flex;gap:8px;align-items:center}}
h2 b{{font-family:ui-monospace;font-weight:600}}
.sw{{width:10px;height:10px;border-radius:2px;flex:none}}
.bar{{flex:1;max-width:220px;height:6px;background:#1c1c1b;border-radius:3px;overflow:hidden}}
.bar i{{display:block;height:100%}}
.grid{{display:flex;flex-wrap:wrap;gap:6px}}
figure{{margin:0;max-width:{args.thumb_width}px}}
figure img,.noimg{{display:block;border-radius:3px;box-shadow:0 0 0 3px var(--c)}}
.noimg{{height:90px;background:#1a1a19}}
figcaption{{font:10px ui-monospace;display:flex;gap:4px;align-items:center;margin-top:5px;overflow:hidden;white-space:nowrap}}
.dot{{width:8px;height:8px;border-radius:2px;background:var(--c);flex:none}}
figcaption .v{{color:#ddd;font-weight:600}}
.hide{{display:none}}
</style>
<header>
<h1>{args.metric} by cluster <span class=dim>&mdash; {_html.escape(args.clusters.name)}
&times; {_html.escape(args.eval.name)}</span></h1>
<div class=dim>{len(allv)} scored scenes &middot; {len(rows)} clusters &middot; overall mean
{overall:.4f} &middot; median {statistics.median(allv):.4f} &middot;
range {min(allv):.3f}&ndash;{max(allv):.3f}</div>
<div class=controls>
<label>clusters <select id=sc>
<option value=mean-desc>worst mean first</option><option value=mean-asc>best mean first</option>
<option value=id>cluster id</option><option value=size>size</option></select></label>
<label>images <select id=si>
<option value=orig>cluster order</option><option value=desc>worst first</option>
<option value=asc>best first</option></select></label>
<label>min cluster size <select id=ms>
<option value=1>1</option><option value=2>2</option><option value=5>5</option>
<option value=10>10</option></select></label>
<div class=legend><span>{vmin:.2f}</span><span class=ramp>{legend}</span><span>{vmax:.2f}</span>
<span>{args.metric} (clipped)</span></div>
</div></header>
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
                'size': s => -num(s, 'size')}}[sc.value];
  [...secs].sort((a, b) => key(a) - key(b) || num(a, 'cluster') - num(b, 'cluster'))
           .forEach(s => m.appendChild(s));
}}
function sortImages() {{
  const key = {{'orig': f => num(f, 'orig'),
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
sc.onchange = sortClusters; si.onchange = sortImages; ms.onchange = filterClusters;
sortClusters();
</script>
"""
    out_html.write_text(doc)
    print(f"\nwrote {out_html}")


if __name__ == "__main__":
    main()
