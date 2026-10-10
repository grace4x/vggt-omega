#!/usr/bin/env python3
"""Gradient-cluster vs register-token-cluster agreement for one run.

Reads ``<run>/grad_clusters/epoch_*.npz`` (or a single ``.npz``) written by
``grad_cluster_train``. AMI is the score to compare across epochs and across k:
1 means the two partitions match, 0 means they agree only as much as chance
given the cluster sizes.

    python grad_cluster_train/cluster_agreement.py runs/my-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def _sizes(labels: np.ndarray) -> tuple[int, int, int]:
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    return int(counts.min()), int(np.median(counts)), int(counts.max())


def _snapshots(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    snaps = sorted((path / "grad_clusters").glob("epoch_*.npz"))
    if not snaps:
        raise SystemExit(f"no grad_clusters/epoch_*.npz under {path}")
    return snaps


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: cluster_agreement.py <run-dir-or-npz>")
    print(f"{'epoch':>5} {'k':>5} {'AMI':>7} {'ARI':>7} {'NMI':>7}  {'grad min/med/max':>18}  {'token min/med/max':>18}")
    for snap in _snapshots(Path(argv[0])):
        data = np.load(snap)
        grad = data["labels"]
        token = data["token_labels"]
        g = _sizes(grad)
        t = _sizes(token)
        print(
            f"{int(data['epoch']):5d} {int(data['k']):5d} "
            f"{adjusted_mutual_info_score(grad, token):7.3f} "
            f"{adjusted_rand_score(grad, token):7.3f} "
            f"{normalized_mutual_info_score(grad, token):7.3f}  "
            f"{g[0]:5d}/{g[1]:5d}/{g[2]:5d}  "
            f"{t[0]:5d}/{t[1]:5d}/{t[2]:5d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
