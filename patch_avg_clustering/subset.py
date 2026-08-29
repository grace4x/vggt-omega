"""Take a random sample from every cluster -- a small, maximally diverse mixed training set.

`cluster.py` groups the pooled DL3DV + ScanNet scenes by visual mode (patch-token
means, see `extract_features.py`). Sampling k (or a
percentage) at random from each cluster keeps one-scene-per-mode coverage in a fraction
of the data, without biasing toward a cluster's centre or its fringe.

    python patch_avg_clustering/subset.py --k 5
    python patch_avg_clustering/subset.py --pct 10

Writes `subset_k5.txt`, one `subset/scene` per line. `train.py --scene-list` filters
*both* datasets against that one file (DL3DV's subsets are `1K`..`5K`, ScanNet's is
`scannet`, so the lines are unambiguous), which is why the mixed sample needs no
per-dataset bookkeeping downstream. The per-dataset counts are printed because the
cluster mix decides them: if the sample comes out lopsided and the run wants an even
mix, that is what `--dl3dv-weight` / `--scannet-weight` are for.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
how = p.add_mutually_exclusive_group()
how.add_argument("--k", type=int, default=None, help="scenes to take from each cluster")
how.add_argument("--pct", type=float, default=None,
                 help="percent of each cluster to take (e.g. 10 = 10%%)")
p.add_argument("--min-size", type=int, default=1, help="skip clusters smaller than this")
p.add_argument("--seed", type=int, default=0, help="RNG seed for the per-cluster sample")
p.add_argument("--clusters", type=Path, default=None, metavar="JSON",
               help="cluster.py output (default: the only clusters_*.json here, if unambiguous)")
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()

if args.k is None and args.pct is None:
    args.k = 5
if args.pct is not None and not (0 < args.pct <= 100):
    p.error("--pct must be in (0, 100]")

if args.clusters is None:
    found = sorted(HERE.glob("clusters_*.json"))
    if len(found) != 1:
        p.error(f"pass --clusters: found {len(found)} candidates in {HERE} "
                f"({[f.name for f in found]})")
    args.clusters = found[0]
elif not args.clusters.exists() and (HERE / args.clusters.name).exists():
    args.clusters = HERE / args.clusters.name  # accept a bare filename

index = json.loads(args.clusters.read_text())
clusters = [c for c in index["clusters"] if c["size"] >= args.min_size]


def take(c: dict) -> int:
    if args.k is not None:
        return min(args.k, c["size"])
    return min(max(1, round(c["size"] * args.pct / 100.0)), c["size"])


rng = random.Random(args.seed)
picked = [s for c in clusters for s in rng.sample(c["scenes"], take(c))]

out = args.out or HERE / (f"subset_k{args.k}.txt" if args.k is not None
                          else f"subset_pct{args.pct:g}.txt")
out.write_text("\n".join(f"{s['subset']}/{s['scene']}" for s in picked) + "\n")

per_dataset = {}
for s in picked:
    # `dataset` is written by this folder's cluster.py; a mapping from clustering/
    # has only subset/scene, so fall back to the subset name.
    per_dataset[s.get("dataset", s["subset"])] = per_dataset.get(s.get("dataset", s["subset"]), 0) + 1
label = f"k={args.k}" if args.k is not None else f"pct={args.pct:g}%"
print(f"wrote {out}: {len(picked)} scenes from {len(clusters)} clusters "
      f"({label}, seed={args.seed}, {args.clusters.name})")
print("  " + ", ".join(f"{n}={c}" for n, c in sorted(per_dataset.items())))
print(f"""
  python training/train.py --scene-list {out} \\
      --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth \\
      --scannet-root ~/scannet-train --dense-only \\
      --dl3dv-weight 1 --scannet-weight 1 --preset small \\
      --dinov3 checkpoints/dinov3_vits16.pt --num-frames 16 --batch-size 4 ...""")
