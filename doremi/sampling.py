"""Sampling a training epoch from a learned domain mixture.

`build_concat_trainset` realises a mixture the only way a `ConcatDataset` and a
shuffle can: by *listing* a dataset's scenes more than once per epoch, which is why
`--scannet-weight` has to be a small number of whole-ish passes. Fifty domains with
fractional weights do not fit that -- a domain wanted at 0.6x its natural share
cannot be listed 0.6 times -- so the final run draws its epoch with replacement
instead, from per-scene weights.

`alpha_j / |C_j|` is the weight of a scene in domain j: the domain's target share of
the stream, spread evenly over the scenes carrying it. Normalised to mean 1 the
numbers read directly as oversampling factors, and the baseline mixture comes out as
all ones -- so `--mode final` with baseline weights is, up to sampling with
replacement, the run it is meant to be compared against.

Sampling with replacement means an epoch no longer visits every scene exactly once:
at `num_samples = len(dataset)` a uniform draw misses ~37% of scenes in any one epoch
and repeats others. Over a long run that evens out (and the frame window resampled
per visit means a repeat is not the same sample anyway), but it does make "epoch"
lose its meaning, which is why `train.py` counts only steps.

    python3 doremi/sampling.py     # realised mix vs target, and the DDP sharding
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def concat_scene_keys(dataset) -> list[tuple[str, str]]:
    """`(subset, scene)` per index of a `build_concat_trainset` result.

    Reaches through the `Subset(ConcatDataset(TaggedDataset(...)))` that
    `build_concat_trainset` may return, because the per-scene weights have to line up
    with dataset *indices* and only the leaf datasets know which scene an index is.
    """
    if isinstance(dataset, Subset):
        inner = concat_scene_keys(dataset.dataset)
        return [inner[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [key for part in dataset.datasets for key in concat_scene_keys(part)]
    scenes = getattr(dataset, "scenes", None)
    if scenes is None:
        raise TypeError(f"cannot enumerate scenes of {type(dataset).__name__}")
    return [(e["subset"], e["scene"]) for e in scenes]


def index_weights(dataset, factors: dict[str, float], *, default: float = 0.0) -> np.ndarray:
    """Per-index sampling weights from `subset/scene` -> factor.

    `default=0` means a scene the mixture does not cover is never drawn. That is the
    right default only because `train.py` filters the training pool to the clusters
    json first; the count is returned in the log line so a silent mass-exclusion is
    visible.
    """
    keys = concat_scene_keys(dataset)
    weights = np.array([factors.get(f"{s}/{n}", default) for s, n in keys], dtype=np.float64)
    if weights.sum() <= 0:
        raise SystemExit(
            "no training index has positive sampling weight -- the weights file and the "
            "training pool do not describe the same scenes"
        )
    return weights


if __name__ == "__main__":
    from torch.utils.data import Dataset

    from doremi.domains import load_domains, scene_factors

    class Fake(Dataset):
        def __init__(self, scenes):
            self.scenes = scenes
            self.image_hw = (224, 384)

        def __len__(self):
            return len(self.scenes)

        def __getitem__(self, i):
            return {"i": i}

    domains = load_domains(min_size=1)
    keys = domains.scene_list()
    parts = {
        "dl3dv": Fake([{"subset": k.split("/")[0], "scene": k.split("/")[1]}
                       for k in keys if not k.startswith("scannet/")]),
        "scannet": Fake([{"subset": k.split("/")[0], "scene": k.split("/")[1]}
                         for k in keys if k.startswith("scannet/")]),
    }

    from training.mixed_dataset import build_concat_trainset

    train_set, names, sizes, counts = build_concat_trainset(parts, None, seed=0)
    print(f"{len(train_set)} indices over {dict(zip(names, sizes))}")
    assert len(concat_scene_keys(train_set)) == len(train_set)

    rng = np.random.default_rng(0)
    # A synthetic mixture that wants the first 10 clusters 3x and the rest less.
    alpha = domains.baseline() * np.where(np.arange(len(domains)) < 10, 3.0, 1.0)
    alpha /= alpha.sum()
    factors = scene_factors(domains, alpha)
    weights = index_weights(train_set, factors)

    sampler = WeightedEpochSampler(weights, len(train_set), seed=0)
    keys_by_index = concat_scene_keys(train_set)
    drawn = list(sampler)
    realised = np.zeros(len(domains))
    for i in drawn:
        subset, scene = keys_by_index[i]
        realised[domains.domain_of(subset, scene)] += 1
    realised /= realised.sum()
    err = np.abs(realised - alpha).max()
    print(f"one epoch: {len(drawn)} draws, {len(set(drawn))} distinct scenes "
          f"({len(set(drawn)) / len(train_set):.0%} of the pool)")
    print(f"realised mix vs target: max abs error {err:.4f} "
          f"(target max {alpha.max():.4f}, realised max {realised.max():.4f})")
    assert err < 0.01, "one epoch should realise the target mixture closely"

    # DDP: the union of the ranks' shards is exactly the epoch, and every rank gets
    # the same count.
    shards = [list(WeightedEpochSampler(weights, len(train_set), rank=r, world_size=4, seed=0))
              for r in range(4)]
    assert len({len(s) for s in shards}) == 1, [len(s) for s in shards]
    assert sorted(i for s in shards for i in s) == sorted(drawn[: 4 * len(shards[0])])
    print(f"4 ranks x {len(shards[0])} indices = {4 * len(shards[0])} (epoch {len(drawn)})")

    # Baseline weights must reproduce the natural mixture.
    flat = index_weights(train_set, scene_factors(domains, domains.baseline()))
    assert np.allclose(flat, flat[0]), "baseline weights should be uniform over scenes"
    print("sampling self-test ok")
