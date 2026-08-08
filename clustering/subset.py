"""Take the first k scenes of every cluster -- a small, maximally diverse training set.

cluster.py orders each cluster by cohesion, so a prefix is the cluster's centre
rather than its fringe. Taking k from each gives one-scene-per-visual-mode
coverage in a fraction of the data.

    python clustering/subset.py --k 5

Writes `subset_k5.txt`, one `subset/scene` per line, which train.py reads:

    python training/train.py --scene-list clustering/subset_k5.txt \
        --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only ...
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--k", type=int, default=5, help="scenes to take from each cluster")
p.add_argument("--min-size", type=int, default=1, help="skip clusters smaller than this")
p.add_argument("--clusters", type=Path, default=HERE / "clusters.json")
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()

clusters = [c for c in json.loads(args.clusters.read_text())["clusters"] if c["size"] >= args.min_size]
picked = [f"{s['subset']}/{s['scene']}" for c in clusters for s in c["scenes"][: args.k]]

out = args.out or HERE / f"subset_k{args.k}.txt"
out.write_text("\n".join(picked) + "\n")
print(f"wrote {out}: {len(picked)} scenes from {len(clusters)} clusters (k={args.k})")
print(f"  python training/train.py --scene-list {out} --data-root ~/dl3dv-train ...")
