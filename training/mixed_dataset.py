"""Training on DL3DV and ScanNet together.

`preprocess_scannet.py` writes the same on-disk contract as `preprocess_dl3dv.py`,
so both datasets load through the same `DL3DVDataset` and mixing them is just a
`ConcatDataset` -- provided they are stored at the same resolution. That is what
`preprocess_scannet.py --target-hw 224 384 --fit crop` is for: ScanNet's 4:3
frames become 224x384, matching a DL3DV set built at `--resolution 384`, and the
two become freely interchangeable inside a batch.

`assert_stackable` enforces that up front. Without it a shape mismatch surfaces
as a `collate_scenes` failure somewhere deep in an epoch, which is a confusing
way to learn that the two sets were preprocessed differently.

Mixing is proportional to scene count by default. `weights` oversamples a
dataset by listing its scenes more than once per epoch, for when the smaller set
deserves more attention than its size gives it.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset


class TaggedDataset(Dataset):
    """Passthrough that records which dataset a sample came from.

    Lets the training loop break loss and metrics down per dataset, which matters
    when mixing: a rising loss means something different if it is ScanNet's dense
    sensor depth than if it is DL3DV's estimated depth.
    """

    def __init__(self, dataset: Dataset, name: str) -> None:
        self.dataset = dataset
        self.name = name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        sample["dataset"] = self.name
        return sample

    # `train.py` reaches through to these for --overfit, --scene-list and the
    # stackability check.
    @property
    def scenes(self):
        return self.dataset.scenes

    @scenes.setter
    def scenes(self, value):
        self.dataset.scenes = value

    @property
    def image_hw(self):
        return self.dataset.image_hw


def assert_stackable(datasets: dict[str, Dataset], batch_size: int) -> tuple[int, int]:
    """Fail early unless every dataset is stored at one shape. Returns that shape.

    `collate_scenes` stacks a batch into (B, S, 3, H, W), so a batch spanning two
    resolutions cannot be formed at all. With `batch_size=1` there is nothing to
    stack and a mismatch is harmless, so it is only reported.
    """
    shapes = {name: tuple(d.image_hw) for name, d in datasets.items()}
    distinct = set(shapes.values())
    if len(distinct) == 1:
        return distinct.pop()
    detail = ", ".join(f"{n}={h}x{w}" for n, (h, w) in sorted(shapes.items()))
    if batch_size > 1:
        raise SystemExit(
            f"datasets are stored at different resolutions ({detail}), so batches of "
            f"{batch_size} cannot be stacked.\n"
            "Rebuild ScanNet at DL3DV's shape:\n"
            "    training/download_scannet.sh --target-hw 224 384 --fit crop\n"
            "or run with --batch-size 1, where every batch holds a single scene."
        )
    print(f"[mix] warning: datasets differ in resolution ({detail}); only --batch-size 1 will work")
    return sorted(distinct)[0]


def build_concat_trainset(
    datasets: dict[str, Dataset],
    weights: dict[str, float] | None = None,
    seed: int = 0,
):
    """(dataset, names, sizes, epoch_counts) for the configured datasets.

    With every weight at 1.0 this is a plain `ConcatDataset` and an epoch sees
    each scene exactly once, so the mix is proportional to scene count. A weight
    above 1.0 lists that dataset's scenes more than once per epoch -- drawn as
    whole shuffled passes, so an oversampled set cycles evenly instead of
    resampling the same few scenes.

    Returns the concatenation regardless of how many datasets there are, so the
    caller has one code path and one plain sampler.
    """
    names = [n for n, d in datasets.items() if d is not None and len(d) > 0]
    if not names:
        raise ValueError("no non-empty dataset to train on")
    tagged = [TaggedDataset(datasets[n], n) for n in names]
    sizes = [len(d) for d in tagged]
    concat = ConcatDataset(tagged)

    factors = [float((weights or {}).get(n, 1.0)) for n in names]
    if any(f <= 0 for f in factors):
        raise ValueError(f"weights must be positive, got {dict(zip(names, factors))}")
    if all(abs(f - 1.0) < 1e-9 for f in factors):
        return concat, names, sizes, list(sizes)

    rng = np.random.default_rng(seed)
    offsets = np.cumsum([0] + sizes[:-1])
    indices: list[int] = []
    counts: list[int] = []
    for offset, size, factor in zip(offsets, sizes, factors):
        target = max(int(round(size * factor)), 1)
        pool: list[int] = []
        while len(pool) < target:
            pool.extend((rng.permutation(size) + offset).tolist())
        indices.extend(pool[:target])
        counts.append(target)
    return Subset(concat, indices), names, sizes, counts


def collate_mixed(batch: list[dict]) -> dict:
    """`collate_scenes` plus the string `dataset` tag."""
    from training.dl3dv_dataset import collate_scenes

    names = [b.pop("dataset", "?") for b in batch]
    out = collate_scenes(batch)
    out["dataset"] = names
    return out


if __name__ == "__main__":
    import collections

    class Fake(Dataset):
        def __init__(self, n, hw):
            self.n, self.image_hw, self.scenes = n, hw, list(range(n))

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {"i": i}

    parts = {"dl3dv": Fake(4871, (224, 384)), "scannet": Fake(1500, (224, 384))}
    print("shape check (matched):", assert_stackable(parts, batch_size=4))

    for weights in (None, {"scannet": 3.0}):
        dataset, names, sizes, counts = build_concat_trainset(parts, weights, seed=0)
        seen = collections.Counter(dataset[i]["dataset"] for i in range(len(dataset)))
        total = sum(seen.values())
        assert total == len(dataset) == sum(counts)
        print(f"weights={weights or 'proportional'}: epoch={len(dataset)} samples, "
              f"sizes={dict(zip(names, sizes))}, mix="
              f"{ {n: round(seen[n] / total, 3) for n in names} }")

    mixed = {"dl3dv": Fake(10, (224, 384)), "scannet": Fake(10, (288, 384))}
    print("shape check (bs=1, mismatched):", assert_stackable(mixed, batch_size=1))
    try:
        assert_stackable(mixed, batch_size=4)
    except SystemExit as exc:
        print("shape check (bs=4, mismatched): raised ->", str(exc).splitlines()[0])
