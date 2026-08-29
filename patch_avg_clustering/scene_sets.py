"""Which scenes a training command actually sees -- resolved once, shared by the pipeline.

`extract_features.py` has to embed exactly the scenes training draws from and
`cluster.py` has to partition exactly those, or the subset `subset.py` writes is a
sample of a different population than the run it feeds. So rather than
re-implement the filter chain (split, dense depth, frame count, stored image
shape) and the prefix cap, this module builds the real `DL3DVDataset` and hands
back `dataset.scenes` -- the same list `train.py` iterates over.

The defaults mirror the run being subset here:

    python training/train.py --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth \
        --scannet-root ~/scannet-train --dl3dv-scenes 1466 \
        --dl3dv-weight 1 --scannet-weight 1 --dense-only --num-frames 16

`--dl3dv-scenes` / `--scannet-scenes` / `--num-frames` therefore mean exactly what
they mean in `train.py`, and must be kept in sync with it: `--num-frames` decides
which scenes are dropped for being too short, which decides *which* N a prefix cap
keeps.

    python patch_avg_clustering/scene_sets.py        # print the resolved scene sets
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # `training.` imports work when run as a script

NUM_FRAMES = 16  # train.py's default, and what the run being subset uses
SPLIT = "train"


@dataclass
class Source:
    """One preprocessed dataset, as `train.py` is pointed at it."""

    name: str
    root: Path
    depth_root: Path | None
    dense_only: bool
    limit: int      # train.py's --<name>-scenes prefix cap; 0 = every scene
    features: Path  # extract_features.py output for this source


DEFAULTS: dict[str, Source] = {
    # Both sources are extracted fresh here. `clustering/train_features_layers.npz`
    # covers the full 4288-scene dense DL3DV split and `multi_clustering/` reuses it as
    # a cache, but it stores CLS descriptors -- the whole point of this folder is the
    # patch-token mean, so there is nothing to reuse and DL3DV costs its own ~5900
    # DINOv3 forward passes. `cluster.py` still selects rows by (subset, scene), so an
    # npz covering more scenes than a cap keeps is used as-is.
    "dl3dv": Source(
        name="dl3dv",
        root=Path("~/dl3dv-train").expanduser(),
        depth_root=Path("~/dl3dv-depth").expanduser(),
        dense_only=True,
        limit=1466,
        features=HERE / "dl3dv_features.npz",
    ),
    # ScanNet's depth is the metric sensor's and always present, so `dense_only`
    # would be a no-op -- see the same note in `train.py`.
    "scannet": Source(
        name="scannet",
        root=Path("~/scannet-train").expanduser(),
        depth_root=Path("~/scannet-train/depth").expanduser(),
        dense_only=False,
        limit=0,
        features=HERE / "scannet_features.npz",
    ),
}


def add_source_args(parser) -> None:
    """`--<name>-root/-depth-root/-scenes/-features`, plus `--split` and `--num-frames`."""
    parser.add_argument("--split", default=SPLIT, choices=("train", "val", "all"),
                        help="index.json split to cluster (default: %(default)s)")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                        help="train.py's --num-frames; scenes with fewer are dropped, "
                             "so this changes which scenes a cap keeps (default: %(default)s)")
    for s in DEFAULTS.values():
        parser.add_argument(f"--{s.name}-root", type=Path, default=s.root)
        parser.add_argument(f"--{s.name}-depth-root", type=Path, default=s.depth_root)
        parser.add_argument(f"--{s.name}-scenes", type=int, default=s.limit, metavar="N",
                            help=f"cap {s.name} at its first N scenes, as in train.py "
                                 f"(0 = all, default: {s.limit})")
        parser.add_argument(f"--{s.name}-features", type=Path, default=s.features, metavar="NPZ",
                            help=f"extract_features.py output for {s.name} "
                                 f"(default: {s.features.relative_to(REPO)})")


def sources_from_args(args, names=None) -> list[Source]:
    """The `Source`s named by `names` (default: all), overridden by the parsed flags."""
    out = []
    for name, s in DEFAULTS.items():
        if names is not None and name not in names:
            continue
        out.append(replace(
            s,
            root=getattr(args, f"{name}_root"),
            depth_root=getattr(args, f"{name}_depth_root"),
            limit=getattr(args, f"{name}_scenes"),
            features=getattr(args, f"{name}_features"),
        ))
    return out


def key(entry: dict) -> str:
    """`subset/scene` -- the identifier `train.py --scene-list` matches on."""
    return f"{entry['subset']}/{entry['scene']}"


def resolve(source: Source, split: str = SPLIT, num_frames: int = NUM_FRAMES):
    """(scenes, image_hw) for one source, exactly as `train.py` would see them.

    `scenes` are `index.json` entries in index order, after the loader's filters and
    after the prefix cap. `augment`/`sampling` are irrelevant here -- nothing calls
    `__getitem__` -- but are pinned so construction cannot depend on them.
    """
    from training.dl3dv_dataset import DL3DVDataset

    dataset = DL3DVDataset(
        source.root,
        name=source.name,
        split=split,
        num_frames=num_frames,
        depth_root=source.depth_root,
        dense_only=source.dense_only,
        augment=False,
        sampling="random",
    )
    scenes = dataset.scenes
    if source.limit and source.limit < len(scenes):
        # A plain prefix, matching train.py: the same N on every rerun, no seed.
        scenes = scenes[: source.limit]
        print(f"[{source.name}] capped to the first {len(scenes)} {split} scenes")
    else:
        print(f"[{source.name}] {len(scenes)} {split} scenes")
    return scenes, tuple(int(v) for v in dataset.image_hw)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(p)
    a = p.parse_args()
    total = 0
    for src in sources_from_args(a):
        scenes, hw = resolve(src, a.split, a.num_frames)
        total += len(scenes)
        print(f"  {src.name}: {len(scenes)} scenes at {hw}, features -> {src.features}"
              f" ({'present' if src.features.exists() else 'MISSING, run extract_features.py'})")
        print(f"  first/last: {key(scenes[0])} ... {key(scenes[-1])}")
    print(f"{total} scenes to cluster")
