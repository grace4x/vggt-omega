#!/usr/bin/env python3
"""Quick small-multiples view of a training log.jsonl.

Usage:
    python plot_log.py                  # plots ./log.jsonl -> ./log.png
    python plot_log.py path/to/log.jsonl [-o out.png] [--show]

Train rows (no "split" key) and val rows ("split": "val") are overlaid per
metric on a shared step axis. Train draws raw + EMA in the same hue.
"""

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 and 2 (light mode).
TRAIN = "#2a78d6"
VAL = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"

# Panels appear in this order when present; anything else follows, sorted.
PREFERRED = [
    "loss", "loss_camera", "loss_depth",
    "cam_trans", "cam_rot", "cam_fov",
    "depth_err", "depth_conf", "depth_valid",
    "rot_err_deg_mean", "rot_err_deg_median",
    "trans_err_deg_mean", "trans_err_deg_median", "trans_err_abs_mean",
    "rra_at_5", "rta_at_5", "auc_at_30",
    "fov_err_deg", "abs_rel", "delta_1.25", "scale_ratio",
    "grad_norm", "steps_per_sec", "lr", "lr_backbone",
]
LOG_Y = {"lr", "lr_backbone"}
SKIP = {"step", "split", "epoch", "time", "wall_time"}


def load(path):
    train, val = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"  skipping unparseable line {i}")
            continue
        (val if row.get("split") == "val" else train).append(row)
    return train, val


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if not math.isfinite(v):
            continue
        xs.append(r.get("step", len(xs)))
        ys.append(v)
    return xs, ys


def smooth(ys, window=9, alpha=0.25):
    """Rolling median then EMA — the median keeps a loss spike from
    poisoning the trend line for the rest of the run."""
    half = window // 2
    med = [sorted(ys[max(0, i - half):i + half + 1])[
               len(ys[max(0, i - half):i + half + 1]) // 2]
           for i in range(len(ys))]
    out, acc = [], None
    for y in med:
        acc = y if acc is None else alpha * y + (1 - alpha) * acc
        out.append(acc)
    return out


def robust_ylim(values, keep=0.995, blowup=8.0):
    """Y-limits that survive a single loss spike.

    Returns (lo, hi, clipped_max) where clipped_max is the true max if the
    full range is >`blowup`x the robust range, else None.
    """
    ys = sorted(values)
    if len(ys) < 8:
        return None, None, None
    lo, hi = ys[0], ys[-1]
    i = int((1 - keep) * len(ys))
    rlo, rhi = ys[i], ys[-1 - i]
    rspan, span = rhi - rlo, hi - lo
    if rspan <= 0 or span <= blowup * rspan:
        return None, None, None
    pad = 0.06 * rspan
    return rlo - pad, rhi + pad, hi


def metric_keys(*groups):
    keys = OrderedDict()
    for rows in groups:
        for r in rows:
            for k, v in r.items():
                if k in SKIP or isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    keys[k] = True
    ordered = [k for k in PREFERRED if k in keys]
    ordered += sorted(k for k in keys if k not in PREFERRED)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=Path(__file__).with_name("log.jsonl"),
                    type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--ncols", type=int, default=5)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    train, val = load(args.log)
    if not train and not val:
        raise SystemExit(f"no usable rows in {args.log}")
    keys = metric_keys(train, val)
    print(f"{args.log}: {len(train)} train rows, {len(val)} val rows, "
          f"{len(keys)} metrics")

    ncols = min(args.ncols, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.3 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, key in zip(axes.flat, keys):
        tx, ty = series(train, key)
        vx, vy = series(val, key)
        if tx:
            ax.plot(tx, ty, color=TRAIN, lw=1.6, alpha=0.22,
                    solid_capstyle="round")
            ax.plot(tx, smooth(ty), color=TRAIN, lw=1.6, label="train",
                    solid_capstyle="round")
        if vx:
            marker = "o" if len(vx) < 40 else None
            ax.plot(vx, vy, color=VAL, lw=1.6, label="val", marker=marker,
                    ms=3.5, solid_capstyle="round")
        clipped = None
        if key in LOG_Y:
            ax.set_yscale("log")
        else:
            lo, hi, clipped = robust_ylim(ty + vy)
            if lo is not None:
                ax.set_ylim(lo, hi)
        ax.set_title(key, fontsize=9, color=INK, pad=5)
        ax.tick_params(labelsize=7, colors=INK_MUTED, length=3)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

        # Last-value annotations in ink, not series color. Park them in
        # whichever right-hand corner the tail of the data isn't using.
        notes = []
        if ty:
            notes.append(f"train {ty[-1]:.4g}")
        if vy:
            notes.append(f"val {vy[-1]:.4g}")
        if clipped is not None:
            notes.append(f"clipped, max {clipped:.3g}")
        tail = (ty or vy)[-max(1, len((ty or vy)) // 4):]
        ylo, yhi = ax.get_ylim()
        mid = 0.5 * (ylo + yhi)
        top = sum(y > mid for y in tail) > len(tail) / 2
        ax.text(0.98, 0.05 if top else 0.96, "\n".join(notes),
                transform=ax.transAxes, ha="right",
                va="bottom" if top else "top", fontsize=6.5, color=INK_MUTED)

    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if len(labels) > 1:
        fig.legend(handles, labels, loc="upper right", frameon=False,
                   fontsize=9, ncol=len(labels),
                   bbox_to_anchor=(0.995, 0.995))

    last = max([r.get("step", 0) for r in train + val] or [0])
    fig.suptitle(f"{args.log.parent.name} — {len(keys)} metrics through "
                 f"step {last:,}", fontsize=12, color=INK, x=0.008, ha="left")
    fig.supxlabel("step", fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.012, 1, 0.985))

    out = args.out or args.log.with_suffix(".png")
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    if args.show:
        import subprocess
        subprocess.run(["xdg-open", str(out)], check=False)


if __name__ == "__main__":
    main()
