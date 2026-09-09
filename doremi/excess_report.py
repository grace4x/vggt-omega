"""Step 0 for a proxy rerun: is there domain structure in the *excess* loss at all?

`report.py --eval-jsonl` asks whether clusters separate raw loss -- they do
(eta^2 = 0.14 with dataset removed, against a null p95 of 0.03). But DoReMi's
weight update never sees raw loss; it sees `max(0, L_proxy - L_ref)`. The first
proxy run showed that quantity is ~80% cluster *size* and barely correlated with
cluster hardness, so this script measures it directly from two finished
checkpoints -- 40 minutes of eval instead of a 32-hour proxy run.

Three questions, in the order that decides whether to rerun:

1. **Is there any signal?** eta^2 of the *signed* per-scene difference by cluster,
   against the shuffle null. Inside the null means the reference and the proxy
   differ by scene-level noise with no cluster structure -- no proxy run of any
   length or budget can learn a mixture from it, and the fix is to the design (a
   proxy with genuinely less capacity than the reference) rather than the run.

2. **Is the size artifact the clamp?** The same domain excess computed two ways:
   clamp-per-scene-then-average (what `train.py` does today, via
   `per_sample_loss.excess_losses`) against average-then-clamp. If only the first
   correlates with cluster size, the aggregation order is the bug.

3. **Does it track hardness?** corr(domain excess, cluster mean loss). This is
   what the mixture is supposed to be reweighting toward.

    python3 doremi/excess_report.py            # after runs/doremi-excess/*.jsonl exist
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doremi.domains import load_domains  # noqa: E402
from doremi.report import _eta2, _eta2_null  # noqa: E402


def read_windows_paired(path: Path, key: str) -> dict[tuple[str, int], float]:
    """`(scene_id, repeat)` -> `key`, so the two checkpoints can be differenced per window.

    Keyed by window rather than by scene because `evaluate.py` re-seeds the dataset
    identically for every checkpoint it is given (`dataset.seed = seed + 100003 *
    repeat`), so repeat r of scene s is the *same* 16 frames for both models. Pairing
    on that removes the frame-window variance from the difference, which is the
    largest term in it.
    """
    out: dict[tuple[str, int], float] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            value = row.get(key)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            out[(row["scene_id"], int(row.get("repeat", 0)))] = float(value)
    return out


def bare_name_map(domains) -> dict[str, int]:
    """`scene` -> domain index. The eval jsonl carries `scene_id` without the subset."""
    by_name: dict[str, int] = {}
    collisions = 0
    for full, d in domains.scene_to_domain.items():
        name = full.split("/", 1)[1]
        if name in by_name and by_name[name] != d:
            collisions += 1
        by_name.setdefault(name, d)
    if collisions:
        print(f"warning: {collisions} scene names map to more than one domain; "
              f"the first wins (dl3dv/scannet name clash)")
    return by_name


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-jsonl", type=Path,
                   default=Path("runs/doremi-excess/doremi-ref.jsonl"))
    p.add_argument("--proxy-jsonl", type=Path,
                   default=Path("runs/doremi-excess/doremi-proxy.jsonl"))
    p.add_argument("--clusters", type=Path, default=None)
    p.add_argument("--min-domain-size", type=int, default=10,
                   help="match the proxy run's --min-domain-size")
    p.add_argument("--metric", default="loss")
    p.add_argument("--draws", type=int, default=200, help="shuffle draws for the null")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--min-present", type=int, default=5,
                   help="a domain needs this many scenes in the eval to enter the\n"
                        "correlations. A cluster the pass only covers 1-2 scenes of has a\n"
                        "standard error of ~0 and a meaningless mean, which is why the\n"
                        "single-dataset halves must not be read on their own")
    args = p.parse_args()

    for path in (args.reference_jsonl, args.proxy_jsonl):
        if not path.exists():
            raise SystemExit(f"missing {path} -- run the two evaluate.py passes first")

    domains = load_domains(path=args.clusters, min_size=args.min_domain_size)
    by_name = bare_name_map(domains)

    ref = read_windows_paired(args.reference_jsonl, args.metric)
    prx = read_windows_paired(args.proxy_jsonl, args.metric)
    shared = sorted(set(ref) & set(prx))
    if not shared:
        raise SystemExit("no (scene, repeat) windows in common between the two jsonls")
    print(f"{len(ref)} reference windows, {len(prx)} proxy windows, "
          f"{len(shared)} paired")

    # Per-scene: mean over that scene's paired windows.
    diffs: dict[str, list[float]] = {}
    refs: dict[str, list[float]] = {}
    for k in shared:
        scene = k[0]
        diffs.setdefault(scene, []).append(prx[k] - ref[k])
        refs.setdefault(scene, []).append(ref[k])

    scenes = [s for s in diffs if s in by_name]
    if len(scenes) < 100:
        raise SystemExit(f"only {len(scenes)} scenes matched the clusters json; "
                         "is this eval over the training scenes?")
    signed = np.array([float(np.mean(diffs[s])) for s in scenes])          # L_proxy - L_ref
    clamped = np.array([float(np.mean(np.maximum(diffs[s], 0.0))) for s in scenes])
    ref_loss = np.array([float(np.mean(refs[s])) for s in scenes])
    dom = np.array([by_name[s] for s in scenes])
    # Dataset label, looked up on the full `subset/scene` key the jsonl has lost.
    full_of: dict[str, str] = {}
    for full in domains.scene_to_domain:
        full_of.setdefault(full.split("/", 1)[1], full)
    dset = np.array([domains.scene_to_dataset.get(full_of[s], "?") for s in scenes])

    sizes = np.array(domains.sizes, dtype=float)
    populated = np.unique(dom)
    cover = np.array([int((dom == d).sum()) for d in populated])
    print(f"{len(scenes)} scenes in {len(populated)}/{len(domains)} clusters "
          f"(scenes per cluster: min {cover.min()} median {int(np.median(cover))} "
          f"max {cover.max()})\n")

    # ---- 1. is there any cluster structure in the excess? ----
    print("=" * 72)
    print("1. does the cluster assignment explain the proxy-vs-reference difference?")
    print("=" * 72)
    print(f"  signed diff (L_proxy - L_ref): mean {signed.mean():+.4f}  sd {signed.std():.4f}")
    print(f"  reference loss              : mean {ref_loss.mean():+.4f}  sd {ref_loss.std():.4f}")
    print()
    for label, y, blocks in (
        ("signed diff, by cluster", signed, None),
        ("signed diff, by cluster (dataset removed)", None, dset),
        ("raw reference loss, by cluster  [the known-positive control]", ref_loss, None),
    ):
        if y is None:
            y = signed.copy()
            for b in np.unique(dset):
                m = dset == b
                y[m] -= y[m].mean()
        e = _eta2(y, dom)
        mean_null, p95 = _eta2_null(y, dom, blocks=blocks, draws=args.draws)
        verdict = "ABOVE the null" if e > p95 else "inside the null"
        print(f"  {label}")
        print(f"      eta^2 = {e:.3f}   null: mean {mean_null:.3f}, p95 {p95:.3f}   -> {verdict}")
    print()

    # ---- 2. is the size correlation created by the per-scene clamp? ----
    print("=" * 72)
    print("2. where does the cluster-size dependence come from?")
    print("=" * 72)
    per_dom = {}
    for d in populated:
        m = dom == d
        per_dom[d] = (signed[m], clamped[m], ref_loss[m])
    present = np.array([int((dom == d).sum()) for d in sorted(per_dom)])
    idx_all = np.array(sorted(per_dom))
    keep = present >= args.min_present
    if not keep.all():
        thin = [(domains.ids[d], int(c)) for d, c in zip(idx_all[~keep], present[~keep])]
        print(f"  dropping {int((~keep).sum())} domains with < {args.min_present} scenes in "
              f"this eval: {thin}")
        print(f"  (a pass covering only one dataset does this to ~14 of the 48 clusters)\n")
    idx = idx_all[keep]
    n = sizes[idx]
    clamp_first = np.array([per_dom[d][1].mean() for d in idx])            # mean_j max(0, diff)
    agg_first = np.array([max(0.0, per_dom[d][0].mean()) for d in idx])    # max(0, mean_j diff)
    signed_dom = np.array([per_dom[d][0].mean() for d in idx])
    hardness = np.array([per_dom[d][2].mean() for d in idx])

    print(f"  {'aggregation':<44s} {'sd':>7s} {'corr(.,log n)':>14s}")
    for label, v in (("clamp per scene, then average  [today]", clamp_first),
                     ("average, then clamp            [fix]", agg_first),
                     ("average, no clamp", signed_dom)):
        print(f"  {label:<44s} {v.std():>7.4f} {corr(v, np.log(n)):>+14.2f}")
    print()
    print("  (the proxy run's own EMA gave corr(excess, log n) = -0.80 -- reproduced here")
    print("   if `clamp per scene` is strongly negative and `average, then clamp` is not)")
    print()

    # ---- 3. does it track hardness, and does it beat its own noise? ----
    print("=" * 72)
    print("3. does the domain excess track domain hardness, and is it above noise?")
    print("=" * 72)
    print(f"  corr(domain excess [clamp per scene], cluster mean loss) = {corr(clamp_first, hardness):+.2f}")
    print(f"  corr(domain excess [average, no clamp], cluster mean loss) = {corr(signed_dom, hardness):+.2f}")
    print()
    se = np.array([per_dom[d][0].std() / max(math.sqrt((dom == d).sum()), 1.0) for d in idx])
    print(f"  cross-domain sd of the signed excess : {signed_dom.std():.4f}")
    print(f"  median standard error of one domain  : {np.median(se):.4f}")
    ratio = signed_dom.std() / max(np.median(se), 1e-12)
    print(f"  ratio                                : {ratio:.2f}x")
    print("  -> a ratio near 1 means the whole cross-domain spread is sampling noise")
    print()
    print(f"  hardest/easiest by signed excess (top {args.top}):")
    print(f"  {'clus':>6s} {'n':>5s} {'signed':>9s} {'clamped':>9s} {'ref_loss':>9s} {'se':>7s}  datasets")
    order = np.argsort(-signed_dom)
    show = list(order[:args.top]) + [None] + list(order[-args.top:])
    for i in show:
        if i is None:
            print(f"  {'...':>6s}")
            continue
        d = idx[i]
        mix = domains.meta[d].get("datasets") or {}
        mix_s = " ".join(f"{k}={v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1]))
        print(f"  {domains.ids[d]:>6d} {int(n[i]):>5d} {signed_dom[i]:>+9.4f} "
              f"{clamp_first[i]:>9.4f} {hardness[i]:>9.4f} {se[i]:>7.4f}  {mix_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
