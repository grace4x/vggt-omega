#!/usr/bin/env python3
"""Average the model weights of several checkpoints into one.

Evaluating the mean of the last few checkpoints (e.g. steps 36k/38k/40k)
instead of `final.pt` alone damps the noise of where training happened to stop,
for the cost of one extra file.  Floating-point tensors are averaged; integer
buffers are taken from the last checkpoint, as are `args` and `step`, so
`training/evaluate.py` loads the result like any other checkpoint.

    python grad_cluster_train/average_checkpoints.py \\
        runs/uniform-s0/step036000.pt runs/uniform-s0/step038000.pt runs/uniform-s0/step040000.pt \\
        --out runs/uniform-s0/avg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    payloads = [torch.load(c, map_location="cpu", weights_only=False) for c in args.checkpoints]
    states = [pl["model"] if "model" in pl else pl for pl in payloads]
    keys = states[0].keys()
    for c, sd in zip(args.checkpoints, states):
        if sd.keys() != keys:
            raise SystemExit(f"{c} has different parameters from {args.checkpoints[0]}")

    averaged = {}
    for k in keys:
        t = states[-1][k]
        if t.is_floating_point():
            averaged[k] = sum(sd[k].double() for sd in states).div(len(states)).to(t.dtype)
        else:
            averaged[k] = t.clone()

    last = payloads[-1]
    torch.save(
        {
            "step": last.get("step", -1),
            "model": averaged,
            "args": last.get("args") or {},
            "averaged_from": [str(c) for c in args.checkpoints],
        },
        args.out,
    )
    print(f"averaged {len(states)} checkpoints -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
