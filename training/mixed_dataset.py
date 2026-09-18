"""Training on DL3DV, ScanNet and MegaSynth together.

`preprocess_scannet.py` and `preprocess_megasynth.py` write the same on-disk
contract as `preprocess_dl3dv.py`, so all three datasets load through the same
`DL3DVDataset` and mixing them is just a `ConcatDataset` -- provided they are
stored at the same resolution. That is what `--target-hw 224 384 --fit crop` is
for in both of the other preprocessors: ScanNet's 4:3 frames and MegaSynth's
square renders both become 224x384, matching a DL3DV set built at
`--resolution 384`, and all three become freely interchangeable inside a batch.

`assert_stackable` enforces that up front. Without it a shape mismatch surfaces
as a `collate_scenes` failure somewhere deep in an epoch, which is a confusing
way to learn that the sets were preprocessed differently.

Mixing is proportional to scene count by default. `weights` oversamples a
dataset by listing its scenes more than once per epoch, for when the smaller set
deserves more attention than its size gives it.

That listing is fixed for the run: a weight of 0.5 picks a random half under
`--seed` and then trains on those scenes every epoch. `dataset_mix_weights` is
the other experiment -- keep every scene eligible and let a with-replacement
sampler draw each dataset at a configured rate (equal by default, or a custom
`--mix-proportions` split), so the mix is not a frozen subset.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset


class TaggedDataset(Dataset):
    """Passthrough that records which dataset a sample came from.

    Lets the training loop break loss and metrics down per dataset, which matters
    when mixing: a rising loss means something different if it is ScanNet's dense
    sensor depth, DL3DV's estimated depth, or MegaSynth's exact rendered depth.
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
            "Rebuild the odd one out at DL3DV's shape:\n"
            "    training/download_scannet.sh --target-hw 224 384 --fit crop\n"
            "    training/download_megasynth.py --target-hw 224 384 --fit crop\n"
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


def dataset_mix_weights(
    names: list[str], sizes: list[int], proportions: dict[str, float] | None = None
) -> np.ndarray:
    """Per-index weights so each dataset occupies a configured share of the stream.

    With `proportions=None` every dataset gets an equal share `1/K` -- this is
    `--equal-mix`. Scene i in dataset n then has weight `share_n / |D_n|`. A draw
    from the concatenated pool is dataset-share-uniform, then scene-uniform within
    the dataset -- the larger set is not over-represented just because it has
    more scenes. Pair with `WeightedEpochSampler` so the mix is redrawn every
    epoch rather than frozen into a `Subset`.

    `proportions` (e.g. `{"dl3dv": 0.5, "megasynth": 0.5}`) must name exactly the
    configured datasets -- a silently-ignored typo would otherwise starve a
    dataset without saying so. Values are relative and normalised to sum to 1,
    so `{"dl3dv": 2, "megasynth": 1}` is the same mix as `{"dl3dv": 2/3,
    "megasynth": 1/3}`.
    """
    if len(names) != len(sizes) or not names:
        raise ValueError(f"names and sizes must be non-empty and aligned, got {names!r} / {sizes!r}")
    if any(s <= 0 for s in sizes):
        raise ValueError(f"every dataset needs at least one scene, got {dict(zip(names, sizes))}")
    if proportions is None:
        shares = {n: 1.0 / len(names) for n in names}
    else:
        if set(proportions) != set(names):
            raise ValueError(
                f"mix proportions must name exactly the configured datasets {sorted(names)}, "
                f"got {sorted(proportions)}"
            )
        total = sum(proportions.values())
        if total <= 0 or any(v <= 0 for v in proportions.values()):
            raise ValueError(f"mix proportions must be positive, got {proportions}")
        shares = {n: v / total for n, v in proportions.items()}
    weights = np.empty(sum(sizes), dtype=np.float64)
    offset = 0
    for name, size in zip(names, sizes):
        weights[offset : offset + size] = shares[name] / size
        offset += size
    return weights


class WeightedEpochSampler(Sampler[int]):
    """Draw `num_samples` indices per epoch with replacement, sharded across ranks.

    `WeightedRandomSampler` and `DistributedSampler` cannot be composed -- the
    former is not a distributed sampler and the latter cannot weight -- so this does
    both: one deterministic draw of the whole epoch from `seed + epoch`, identical on
    every rank, then rank r takes every world_size-th index. Every rank drawing the
    same epoch and slicing it is what keeps the realised mixture equal to the target
    globally rather than only in expectation per rank.

    `set_epoch` matches `DistributedSampler`'s interface so the training loop can
    call it unconditionally.
    """

    def __init__(
        self,
        weights: np.ndarray | torch.Tensor,
        num_samples: int | None = None,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.weights = torch.as_tensor(np.asarray(weights, dtype=np.float64), dtype=torch.double)
        if self.weights.ndim != 1 or len(self.weights) == 0:
            raise ValueError("weights must be a non-empty 1-D vector")
        if (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        self.total = int(num_samples if num_samples is not None else len(self.weights))
        self.rank, self.world_size, self.seed = rank, world_size, seed
        if drop_last:
            # Same count on every rank, so no rank runs dry mid-step under DDP.
            self.total -= self.total % world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 977 * self.epoch)
        drawn = torch.multinomial(self.weights, self.total, replacement=True, generator=generator)
        yield from drawn[self.rank :: self.world_size].tolist()


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

    parts = {
        "dl3dv": Fake(4871, (224, 384)),
        "scannet": Fake(1500, (224, 384)),
        "megasynth": Fake(5000, (224, 384)),
    }
    print("shape check (matched):", assert_stackable(parts, batch_size=4))

    for weights in (None, {"scannet": 3.0}, {"megasynth": 0.3}):
        dataset, names, sizes, counts = build_concat_trainset(parts, weights, seed=0)
        seen = collections.Counter(dataset[i]["dataset"] for i in range(len(dataset)))
        total = sum(seen.values())
        assert total == len(dataset) == sum(counts)
        print(f"weights={weights or 'proportional'}: epoch={len(dataset)} samples, "
              f"sizes={dict(zip(names, sizes))}, mix="
              f"{ {n: round(seen[n] / total, 3) for n in names} }")

    dataset, names, sizes, _ = build_concat_trainset(parts, None, seed=0)
    mix_w = dataset_mix_weights(names, sizes)
    assert len(mix_w) == len(dataset)
    offset = 0
    for name, size in zip(names, sizes):
        share = float(mix_w[offset : offset + size].sum())
        assert abs(share - 1.0 / len(names)) < 1e-9, (name, share)
        offset += size
    print(f"equal-mix: pool={len(dataset)}  per-dataset share="
          f"{ {n: round(1.0 / len(names), 3) for n in names} }")

    custom_proportions = {"dl3dv": 0.5, "megasynth": 0.5}
    mix_w = dataset_mix_weights(["dl3dv", "megasynth"], [sizes[names.index("dl3dv")], sizes[names.index("megasynth")]],
                                 custom_proportions)
    offset = 0
    for name in ("dl3dv", "megasynth"):
        size = sizes[names.index(name)]
        share = float(mix_w[offset : offset + size].sum())
        assert abs(share - 0.5) < 1e-9, (name, share)
        offset += size
    print(f"custom mix-proportions={custom_proportions}: shares match")

    mix_w = dataset_mix_weights(names, sizes)
    drawn = list(WeightedEpochSampler(mix_w, len(dataset), seed=0))
    shards = [list(WeightedEpochSampler(mix_w, len(dataset), rank=r, world_size=4, seed=0))
              for r in range(4)]
    assert len({len(s) for s in shards}) == 1, [len(s) for s in shards]
    assert sorted(i for s in shards for i in s) == sorted(drawn[: 4 * len(shards[0])])
    print(f"weighted sampler: 4 ranks x {len(shards[0])} = {4 * len(shards[0])} (epoch {len(drawn)})")

    mixed = {"dl3dv": Fake(10, (224, 384)), "scannet": Fake(10, (288, 384))}
    print("shape check (bs=1, mismatched):", assert_stackable(mixed, batch_size=1))
    try:
        assert_stackable(mixed, batch_size=4)
    except SystemExit as exc:
        print("shape check (bs=4, mismatched): raised ->", str(exc).splitlines()[0])
