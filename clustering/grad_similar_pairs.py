"""Find scene pairs with similar training-loss gradient directions and render them.

Backprops `VGGTOmegaLoss` once per scene, CountSketches each ∇θ, ranks pairs by
cosine similarity, and writes an HTML contact sheet of the top pairs for
qualitative inspection. Also includes a few contrast pairs (high feature cosine,
low gradient cosine) when `--features` is passed.

Default: `runs/default/final.pt` on the mixed DL3DV + ScanNet train subset from
that run's `args.json` (`multi_clustering/subset_dl3dv50pct_scannet.txt`).

    .venv/bin/python clustering/grad_similar_pairs.py
    .venv/bin/python clustering/grad_similar_pairs.py --max-scenes 200 --top 30
    .venv/bin/python clustering/grad_similar_pairs.py --run runs/default --force
"""

from __future__ import annotations

import argparse
import html as _html
import json
import math
import sys
import time
from heapq import nlargest, nsmallest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
DEFAULT_RUN = ROOT / "runs/default"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from training.dl3dv_dataset import DL3DVDataset, collate_scenes  # noqa: E402
from training.evaluate import read_payload  # noqa: E402
from training.losses import VGGTOmegaLoss  # noqa: E402
from training.model_config import build_model  # noqa: E402
from training.mixed_dataset import TaggedDataset, collate_mixed  # noqa: E402


def fold_into(sketch: torch.Tensor, g: torch.Tensor, salt: int) -> None:
    D = sketch.numel()
    n = g.numel()
    pad = (D - n % D) % D
    if pad:
        g = F.pad(g, (0, pad))
    idx = torch.arange(g.numel(), device=g.device, dtype=torch.int64)
    h = idx + (salt & 0x7FFFFFFF)
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB
    h = h ^ (h >> 31)
    signs = (h & 1).to(dtype=g.dtype).mul_(2).sub_(1)
    sketch.add_((g * signs).view(-1, D).sum(0))


def sketch_grad(model, D: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(D, device=device, dtype=torch.float32)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().flatten()
        fold_into(out, g, hash(name) & 0x7FFFFFFF)
    return out


def grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(sq)


def load_feature_vectors(path: Path) -> dict[str, np.ndarray]:
    """Layer-30 scene descriptors keyed by `subset/scene` and full npz keys."""
    d = np.load(path, allow_pickle=False)
    vecs = d["scene"].astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    out: dict[str, np.ndarray] = {}
    keys = d["keys"] if "keys" in d.files else [
        f"{ds}/{sub}/{sc}" if str(sub) != "scannet" else f"scannet/{sc}"
        for ds, sub, sc in zip(d["datasets"], d["subsets"], d["scenes"])
    ]
    for i, key in enumerate(keys):
        key = str(key)
        out[key] = vecs[i]
        if key.startswith("dl3dv/"):
            out[key[len("dl3dv/") :]] = vecs[i]
    return out


def scene_feature_key(rec: dict) -> str:
    if rec.get("dataset") == "scannet" or rec["subset"] == "scannet":
        return f"scannet/{rec['scene_id']}"
    return f"dl3dv/{rec['subset']}/{rec['scene_id']}"


def load_run_defaults(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return {}
    raw = json.loads(args_path.read_text())
    out = {}
    for k, v in raw.items():
        if v is None:
            continue
        if k in {"data_root", "depth_root", "scannet_root", "scannet_depth_root", "scene_list", "out"}:
            out[k] = Path(v) if k != "scene_list" else (ROOT / v if not Path(v).is_absolute() else Path(v))
    return out


def build_train_dataset(
    *,
    data_root: Path | None,
    depth_root: Path | None,
    scannet_root: Path | None,
    scannet_depth_root: Path | None,
    scene_list: Path | None,
    num_frames: int,
    resolution: int | None,
    dense_only: bool,
    sampling: str,
    seed: int,
    max_scenes: int,
) -> ConcatDataset:
    parts: dict[str, TaggedDataset] = {}
    if data_root is not None:
        parts["dl3dv"] = TaggedDataset(
            DL3DVDataset(
                data_root,
                split="train",
                num_frames=num_frames,
                resolution=resolution,
                sampling=sampling,
                augment=False,
                seed=seed,
                depth_root=depth_root,
                dense_only=dense_only,
                name="dl3dv",
            ),
            "dl3dv",
        )
    if scannet_root is not None:
        depth = scannet_depth_root or scannet_root / "depth"
        parts["scannet"] = TaggedDataset(
            DL3DVDataset(
                scannet_root,
                split="train",
                num_frames=num_frames,
                resolution=resolution,
                sampling=sampling,
                augment=False,
                seed=seed,
                depth_root=depth,
                dense_only=dense_only,
                name="scannet",
            ),
            "scannet",
        )
    if not parts:
        raise SystemExit("need --data-root and/or --scannet-root")

    wanted = set(scene_list.read_text().split()) if scene_list else None
    for name, tagged in parts.items():
        ds = tagged.dataset
        if wanted is not None:
            ds.scenes = [e for e in ds.scenes if f"{e['subset']}/{e['scene']}" in wanted]
        print(f"[{name}] {len(ds.scenes)} scenes", flush=True)

    datasets = [p for p in parts.values() if len(p.scenes) > 0]
    if not datasets:
        raise SystemExit("no training scenes survived filters")
    combined: ConcatDataset | Subset = ConcatDataset(datasets)
    if max_scenes:
        combined = balanced_subset(combined, datasets, max_scenes)
    return combined


def balanced_subset(
    combined: ConcatDataset,
    datasets: list[TaggedDataset],
    max_scenes: int,
) -> Subset:
    """Take roughly equal scenes from each dataset before capping at max_scenes."""
    if max_scenes <= 0 or len(datasets) <= 1:
        return Subset(combined, range(min(max_scenes or len(combined), len(combined))))

    n_parts = len(datasets)
    base, rem = divmod(max_scenes, n_parts)
    indices: list[int] = []
    offset = 0
    for i, tagged in enumerate(datasets):
        take = min(base + (1 if i < rem else 0), len(tagged))
        indices.extend(range(offset, offset + take))
        print(f"[sample] {tagged.name}: {take}/{len(tagged)} scenes", flush=True)
        offset += len(tagged)
    return Subset(combined, indices)


def thumb_path(thumbs: Path, subset: str, scene: str) -> Path | None:
    p = thumbs / f"{subset}_{scene}.jpg"
    return p if p.exists() else None


def pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def select_middle_pairs(
    all_grad: list[tuple[float, int, int]],
    middle: int,
    median_cos: float,
) -> list[tuple[float, int, int]]:
    """Pick pairs spread across the central slice of the cosine distribution."""
    if not all_grad or middle <= 0:
        return []
    cos_arr = np.array([c for c, _, _ in all_grad], dtype=np.float64)
    # Evenly sample target quantiles around the median (p30..p70 by default).
    lo_q = max(0.0, 0.5 - 0.20)
    hi_q = min(1.0, 0.5 + 0.20)
    targets = np.quantile(cos_arr, np.linspace(lo_q, hi_q, middle))
    selected: list[tuple[float, int, int]] = []
    used: set[tuple[int, int]] = set()
    for target in targets:
        best = min(all_grad, key=lambda x: (abs(x[0] - float(target)), x[1], x[2]))
        key = pair_key(best[1], best[2])
        if key in used:
            continue
        selected.append(best)
        used.add(key)
    if len(selected) < middle:
        for item in sorted(all_grad, key=lambda x: abs(x[0] - median_cos)):
            key = pair_key(item[1], item[2])
            if key in used:
                continue
            selected.append(item)
            used.add(key)
            if len(selected) >= middle:
                break
    return selected[:middle]


def collect_pairs(
    G: np.ndarray,
    feat: dict[str, np.ndarray] | None,
    records: list[dict],
    top: int,
    bottom: int,
    middle: int,
    contrast: int,
    min_cos: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, float]]:
    n = len(records)
    best_grad: list[tuple[float, int, int]] = []
    all_grad: list[tuple[float, int, int]] = []
    contrast_pairs: list[tuple[float, float, int, int]] = []

    for i in range(n):
        gi = G[i]
        for j in range(i + 1, n):
            cos_g = float(gi @ G[j])
            all_grad.append((cos_g, i, j))
            if cos_g >= min_cos:
                best_grad.append((cos_g, i, j))
            if feat is None:
                continue
            si, sj = records[i], records[j]
            fi = feat.get(scene_feature_key(si))
            fj = feat.get(scene_feature_key(sj))
            if fi is None or fj is None:
                continue
            cos_f = float(fi @ fj)
            if cos_f >= 0.85 and cos_g <= 0.05:
                contrast_pairs.append((cos_f - cos_g, cos_f, i, j))

    all_cos = np.array([c for c, _, _ in all_grad], dtype=np.float64)
    median_cos = float(np.median(all_cos)) if len(all_cos) else 0.0
    stats = {
        "median_cos": median_cos,
        "mean_cos": float(all_cos.mean()) if len(all_cos) else 0.0,
        "min_cos": float(all_cos.min()) if len(all_cos) else 0.0,
        "max_cos": float(all_cos.max()) if len(all_cos) else 0.0,
        "n_pairs": float(len(all_cos)),
    }

    top_pairs = nlargest(top, best_grad, key=lambda x: x[0])
    bottom_pairs = nsmallest(bottom, all_grad, key=lambda x: x[0])
    middle_pairs = select_middle_pairs(all_grad, middle, median_cos)
    contrast_sel = nlargest(contrast, contrast_pairs, key=lambda x: x[0])

    def pack(cos_g: float, i: int, j: int, kind: str, cos_f: float | None = None) -> dict:
        a, b = records[i], records[j]
        out = {
            "kind": kind,
            "grad_cosine": cos_g,
            "feature_cosine": cos_f,
            "a": a,
            "b": b,
        }
        if feat is not None and cos_f is None:
            fi = feat.get(scene_feature_key(a))
            fj = feat.get(scene_feature_key(b))
            if fi is not None and fj is not None:
                out["feature_cosine"] = float(fi @ fj)
        return out

    grad_rows = [pack(c, i, j, "top_grad") for c, i, j in top_pairs]
    bottom_rows = [pack(c, i, j, "bottom_grad") for c, i, j in bottom_pairs]
    middle_rows = [pack(c, i, j, "middle_grad") for c, i, j in middle_pairs]
    contrast_rows = [pack(cos_g, i, j, "contrast", cos_f) for _, cos_f, i, j in contrast_sel]
    return grad_rows, contrast_rows, bottom_rows, middle_rows, stats


def render_html(
    sections: list[tuple[str, str, list[dict]]],
    title: str,
    out: Path,
    thumbs: Path,
    checkpoint: Path,
    step: int,
    n_scenes: int,
    stats: dict[str, float],
) -> None:
    def scene_cell(rec: dict) -> str:
        tip = (
            f"{rec.get('dataset', '?')}/{rec['subset']}/{rec['scene_id']}\n"
            f"loss {rec['loss']:.3f}  depth {rec['loss_depth']:.3f}\n"
            f"||g|| {rec['grad_norm']:.2f}"
        )
        thumb = rec.get("thumb")
        if thumb:
            img = f'<img src="{_html.escape(thumb)}" width=220>'
        else:
            img = '<div class=noimg style="width:220px;height:124px"></div>'
        sid = rec["scene_id"][:12]
        ds = rec.get("dataset", rec["subset"])
        return (
            f'<figure title="{_html.escape(tip)}">{img}'
            f'<figcaption><span class=s>{_html.escape(ds)}</span>'
            f'<span class=id>{_html.escape(sid)}</span>'
            f'<span class=loss>L={rec["loss"]:.3f}</span></figcaption></figure>'
        )

    body = []
    nav = []
    n_pairs = sum(len(pairs) for _, _, pairs in sections)
    for section_id, section_title, pairs in sections:
        if not pairs:
            continue
        cos_vals = [p["grad_cosine"] for p in pairs]
        nav.append(
            f'<a href="#{section_id}">{_html.escape(section_title)} ({len(pairs)}, '
            f'{min(cos_vals):+.2f}&hellip;{max(cos_vals):+.2f})</a>'
        )
        body.append(f'<h2 class=section id="{section_id}">{_html.escape(section_title)}</h2>')
        for k, p in enumerate(pairs):
            a, b = p["a"], p["b"]
            cos_g = p["grad_cosine"]
            cos_f = p.get("feature_cosine")
            feat_txt = f" &middot; feat cos {cos_f:+.3f}" if cos_f is not None else ""
            kind = p["kind"].replace("_", " ")
            body.append(
                f'<section class=pair data-kind="{_html.escape(p["kind"])}">'
                f'<h3>#{k + 1} <b>grad cos {cos_g:+.3f}</b>'
                f'<span class=dim>{kind}{feat_txt}</span></h3>'
                f'<div class=row>{scene_cell(a)}{scene_cell(b)}</div></section>'
            )

    doc = f"""<!doctype html><meta charset=utf-8>
<title>{_html.escape(title)}</title>
<style>
body{{background:#111;color:#ddd;font:13px system-ui;margin:0;padding:16px}}
header{{position:sticky;top:0;background:#111;padding:8px 0 12px;z-index:2;border-bottom:1px solid #222}}
h1{{font-size:15px;font-weight:600;margin:0 0 4px}}
.dim{{color:#888;font-weight:400}}
.meta{{color:#888;font:11px ui-monospace;line-height:1.5}}
.nav{{display:flex;flex-wrap:wrap;gap:8px 14px;margin-top:8px}}
.nav a{{color:#9ec5f4;text-decoration:none;font-size:12px}}
.nav a:hover{{text-decoration:underline}}
.section{{font-size:14px;font-weight:600;margin:28px 0 14px;padding-top:8px;border-top:1px solid #333;color:#e8c4a0;scroll-margin-top:88px}}
.section:first-of-type{{margin-top:16px;border-top:none;padding-top:0}}
.pair{{margin:18px 0 24px;padding-bottom:18px;border-bottom:1px solid #222}}
h3{{font-size:13px;font-weight:600;margin:0 0 10px;display:flex;gap:10px;align-items:baseline}}
h3 b{{font-family:ui-monospace;color:#9ec5f4}}
.pair[data-kind="bottom_grad"] h3 b{{color:#f0a080}}
.pair[data-kind="middle_grad"] h3 b{{color:#9ed4a0}}
.row{{display:flex;gap:14px;flex-wrap:wrap}}
figure{{margin:0;max-width:220px}}
figure img,.noimg{{display:block;border-radius:4px;box-shadow:0 0 0 2px #333}}
.noimg{{background:#1a1a19}}
figcaption{{font:10px ui-monospace;display:grid;gap:2px;margin-top:6px}}
.s{{color:#86b6ef}}
.id{{color:#ccc}}
.loss{{color:#ddd;font-weight:600}}
</style>
<header>
<h1>{_html.escape(title)}</h1>
<p class=meta>{n_scenes} scenes &middot; {_html.escape(checkpoint.name)} step {step} &middot; {int(stats["n_pairs"])} total pairs &middot; all-pairs cos min {stats["min_cos"]:+.3f} med {stats["median_cos"]:+.3f} mean {stats["mean_cos"]:+.3f} max {stats["max_cos"]:+.3f}</p>
<nav class=nav>{"".join(nav)}</nav>
</header>
{"".join(body)}
"""
    out.write_text(doc)
    print(f"wrote {out} ({n_pairs} pairs across {len(nav)} sections)")


def os_path_relpath(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)


def main() -> int:
    run_defaults = load_run_defaults(DEFAULT_RUN)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="read data paths from <run>/args.json")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: <run>/final.pt")
    ap.add_argument("--data-root", type=Path, default=run_defaults.get("data_root"))
    ap.add_argument("--depth-root", type=Path, default=run_defaults.get("depth_root"))
    ap.add_argument("--scannet-root", type=Path, default=run_defaults.get("scannet_root"))
    ap.add_argument("--scannet-depth-root", type=Path, default=run_defaults.get("scannet_depth_root"))
    ap.add_argument("--scene-list", type=Path, default=run_defaults.get("scene_list"))
    ap.add_argument("--features", type=Path, default=ROOT / "train_with_eval/layer30_features.npz")
    ap.add_argument("--thumbs", type=Path, default=HERE / "thumbs")
    ap.add_argument("--sketch-dim", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-scenes", type=int, default=0, help="cap scenes (0 = all in scene list)")
    ap.add_argument("--top", type=int, default=25, help="top pairs by gradient cosine")
    ap.add_argument("--bottom", type=int, default=25, help="bottom pairs by gradient cosine")
    ap.add_argument("--middle", type=int, default=25, help="pairs closest to the all-pairs median cosine")
    ap.add_argument("--contrast", type=int, default=12, help="high-feature / low-grad contrast pairs")
    ap.add_argument("--min-cos", type=float, default=-1.0, help="only rank pairs with grad cos >= this")
    ap.add_argument("--cache", type=Path, default=HERE / "grad_sketches_default.npz",
                    help="save/load CountSketch vectors here")
    ap.add_argument("--out", type=Path, default=HERE / "grad_similar_pairs_default.html")
    ap.add_argument("--json", type=Path, default=HERE / "grad_similar_pairs_default.json")
    ap.add_argument("--force", action="store_true", help="recompute even if cache exists")
    args = ap.parse_args()

    if args.checkpoint is None:
        for name in ("final.pt", "latest.pt"):
            candidate = args.run / name
            if candidate.exists():
                args.checkpoint = candidate
                break
        if args.checkpoint is None:
            raise SystemExit(f"no checkpoint under {args.run}")

    records: list[dict]
    G: np.ndarray
    step: int

    if args.cache.exists() and not args.force:
        cache = np.load(args.cache, allow_pickle=False)
        cached_ckpt = str(cache.get("checkpoint", ""))
        if cached_ckpt and Path(cached_ckpt).resolve() != Path(args.checkpoint).resolve():
            print(f"cache is for {cached_ckpt}; pass --force to recompute for {args.checkpoint}", flush=True)
            return 1
        records = json.loads(str(cache["records"]))
        G = cache["sketches"].astype(np.float32)
        G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
        step = int(cache["step"])
        print(f"loaded {len(records)} sketches from {args.cache}", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict, train_args, step = read_payload(args.checkpoint)
        preset = train_args.get("preset") or "small"
        num_frames = train_args.get("num_frames") or 16
        resolution = train_args.get("resolution")
        dense_only = bool(train_args.get("dense_only"))
        sampling = train_args.get("sampling") or "covisibility"
        print(f"checkpoint {args.checkpoint.name} step {step} preset {preset} frames {num_frames} device {device}", flush=True)

        model = build_model(preset, use_checkpoint=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.train()
        criterion = VGGTOmegaLoss(
            weight_camera=train_args.get("weight_camera", 5.0),
            weight_depth=train_args.get("weight_depth", 1.0),
            weight_point=train_args.get("weight_point", 0.5),
            weight_gradient=train_args.get("weight_gradient", 1.0),
            depth_kwargs={"alpha": train_args.get("conf_alpha", 0.2)},
        )

        dataset = build_train_dataset(
            data_root=args.data_root,
            depth_root=args.depth_root,
            scannet_root=args.scannet_root,
            scannet_depth_root=args.scannet_depth_root,
            scene_list=args.scene_list,
            num_frames=num_frames,
            resolution=resolution,
            dense_only=dense_only,
            sampling=sampling,
            seed=args.seed,
            max_scenes=args.max_scenes,
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_mixed,
        )

        records, sketches = [], []
        started = time.time()
        for i, batch in enumerate(loader):
            tensors = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            model.zero_grad(set_to_none=True)
            predictions = model(tensors["images"])
            loss, logs = criterion(predictions, tensors)
            if not math.isfinite(float(loss.detach())):
                print(f"  skip {batch['scene_id'][0][:12]} non-finite loss", flush=True)
                continue
            loss.backward()
            sk = sketch_grad(model, args.sketch_dim, device)
            gnorm = grad_norm(model)
            scene_id = batch["scene_id"][0]
            subset = batch["subset"][0]
            dataset_name = batch["dataset"][0]
            rec = {
                "scene_id": scene_id,
                "subset": subset,
                "dataset": dataset_name,
                "loss": float(logs["loss"]),
                "loss_depth": float(logs.get("loss_depth", torch.nan)),
                "grad_norm": gnorm,
            }
            thumb = thumb_path(args.thumbs, subset, scene_id)
            if thumb:
                rec["thumb"] = Path(os_path_relpath(thumb, args.out.parent)).as_posix()
            records.append(rec)
            sketches.append(sk.cpu().numpy())
            if (i + 1) % 20 == 0 or i == 0:
                rate = (i + 1) / max(time.time() - started, 1e-9)
                print(f"  {i + 1}/{len(dataset)}  {rate:.2f} scenes/s  loss {rec['loss']:.3f}  {dataset_name}", flush=True)
            del predictions, loss, logs, sk, tensors

        G_raw = np.stack(sketches).astype(np.float32)
        np.savez(
            args.cache,
            sketches=G_raw,
            records=np.array(json.dumps(records)),
            step=step,
            checkpoint=str(args.checkpoint),
        )
        G = G_raw / np.maximum(np.linalg.norm(G_raw, axis=1, keepdims=True), 1e-12)
        print(f"sketched {len(records)} scenes in {time.time() - started:.1f}s, wrote {args.cache}", flush=True)

    feat = load_feature_vectors(args.features) if args.features.exists() else None
    if feat:
        hit = sum(1 for r in records if scene_feature_key(r) in feat)
        print(f"feature overlap {hit}/{len(records)} scenes", flush=True)

    top_pairs, contrast_pairs, bottom_pairs, middle_pairs, stats = collect_pairs(
        G, feat, records, args.top, args.bottom, args.middle, args.contrast, args.min_cos
    )
    if not top_pairs and not bottom_pairs and not middle_pairs:
        print("no gradient pairs found (try lowering --min-cos or adding scenes)", flush=True)
        return 1

    print(
        f"all-pairs grad cos: min {stats['min_cos']:+.3f}  median {stats['median_cos']:+.3f}  "
        f"mean {stats['mean_cos']:+.3f}  max {stats['max_cos']:+.3f}",
        flush=True,
    )
    if top_pairs:
        cos_vals = [p["grad_cosine"] for p in top_pairs]
        print(
            f"top grad cos: max {max(cos_vals):+.3f}  p90 {np.quantile(cos_vals, 0.9):+.3f}  "
            f"median {np.median(cos_vals):+.3f} over {len(top_pairs)} pairs",
            flush=True,
        )
    if middle_pairs:
        cos_vals = [p["grad_cosine"] for p in middle_pairs]
        print(
            f"middle grad cos: min {min(cos_vals):+.3f}  max {max(cos_vals):+.3f}  "
            f"median {np.median(cos_vals):+.3f} over {len(middle_pairs)} pairs "
            f"(target {stats['median_cos']:+.3f})",
            flush=True,
        )
    if bottom_pairs:
        cos_vals = [p["grad_cosine"] for p in bottom_pairs]
        print(
            f"bottom grad cos: min {min(cos_vals):+.3f}  p10 {np.quantile(cos_vals, 0.1):+.3f}  "
            f"median {np.median(cos_vals):+.3f} over {len(bottom_pairs)} pairs",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "n_scenes": len(records),
        "stats": stats,
        "top_pairs": top_pairs,
        "middle_pairs": middle_pairs,
        "bottom_pairs": bottom_pairs,
        "contrast_pairs": contrast_pairs,
    }
    args.json.write_text(json.dumps(payload, indent=2))
    sections = [
        ("top", "Highest gradient cosine", top_pairs),
        ("middle", f"Median gradient cosine (~{stats['median_cos']:+.3f})", middle_pairs),
        ("bottom", "Lowest gradient cosine", bottom_pairs),
    ]
    if contrast_pairs:
        sections.append(("contrast", "High feature / low gradient contrast", contrast_pairs))
    render_html(
        sections,
        "Training-loss gradient pairs",
        args.out,
        args.thumbs,
        args.checkpoint,
        step,
        len(records),
        stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
