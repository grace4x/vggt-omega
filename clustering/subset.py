"""Take a prefix of every cluster -- a small, maximally diverse training set.

cluster.py orders each cluster by ascending cohesion, so a prefix is the cluster's
fringe -- its least-typical scenes -- rather than its centre. Taking k (or a
percentage) from each gives one-scene-per-visual-mode coverage in a fraction of
the data.

    python clustering/subset.py --k 5
    python clustering/subset.py --pct 10

Writes `subset_k5.txt` / `subset_pct10.txt`, one `subset/scene` per line, which
train.py reads:

    python training/train.py --scene-list clustering/subset_k5.txt \
        --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only ...
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
how = p.add_mutually_exclusive_group()
how.add_argument("--k", type=int, default=None, help="scenes to take from each cluster")
how.add_argument("--pct", type=float, default=None,
                 help="percent of each cluster to take (e.g. 10 = top 10%%)")
p.add_argument("--min-size", type=int, default=1, help="skip clusters smaller than this")
p.add_argument("--clusters", type=Path, default=HERE / "clusters.json")
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()

if args.k is None and args.pct is None:
    args.k = 5
if args.pct is not None and not (0 < args.pct <= 100):
    p.error("--pct must be in (0, 100]")

clusters = [c for c in json.loads(args.clusters.read_text())["clusters"] if c["size"] >= args.min_size]


def take(c: dict) -> int:
    if args.k is not None:
        return args.k
    n = max(1, round(c["size"] * args.pct / 100.0))
    return min(n, c["size"])


picked = [f"{s['subset']}/{s['scene']}" for c in clusters for s in c["scenes"][: take(c)]]

if args.out is not None:
    out = args.out
elif args.k is not None:
    out = HERE / f"subset_k{args.k}_n4_layered.txt"
else:
    out = HERE / f"subset_pct{args.pct:g}_n4_layered.txt"

out.write_text("\n".join(picked) + "\n")
label = f"k={args.k}" if args.k is not None else f"pct={args.pct:g}%"
print(f"wrote {out}: {len(picked)} scenes from {len(clusters)} clusters ({label})")
print(f"  python training/train.py --scene-list {out} --data-root ~/dl3dv-train ...")
