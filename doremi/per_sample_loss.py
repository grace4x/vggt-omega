"""Per-scene losses out of `VGGTOmegaLoss`, and the alpha-weighted objective.

DoReMi needs a loss *per domain* inside a step, and `VGGTOmegaLoss` returns one
scalar for the whole batch: `camera_loss` ends in `.mean()` over (B, S, 9) and both
dense terms reduce through `_masked_mean`, a global `sum()/sum()`. Rather than
thread a `per_sample` flag through `training/losses.py` -- shared code that every
other run in this repo depends on -- this evaluates the criterion once per batch
element on a slice of it, which is what `evaluate.window_metrics` already does to
write its per-window jsonl.

That is exact, not an approximation: `_relative_scale` is already computed per batch
element, and a masked mean over a one-element slice is that element's masked mean. It
is also nearly free -- B calls each doing 1/B of the elementwise work -- and it keeps
one definition of the loss in the repo.

The one thing it is *not* is equal to the batch loss: `mean_i L_i` weights each scene
equally, while `criterion(batch)` weights each *valid pixel* equally, so a scene with
sparse GT counts less in the batch loss than in the mean of the per-scene losses.
Both are defensible; DoReMi's per-domain normalisation is the per-example one, so the
proxy objective built here trains on a slightly different loss from the reference run
it is compared against. `--per-sample-objective` in `train.py` makes that explicit.

`domain_weighted_loss` reduces to exactly `mean_i L_i` when the mixture sits at
baseline, so that flag is the *only* remaining difference between the two objectives.
It did not use to: the `"domain-mean"` scheme it replaced gave a scene a weight that
scaled with its cluster's size, which is what made `runs/doremi-proxy` unusable. That
history is in `domain_weighted_loss`'s docstring and measured in the self-test.

    python3 doremi/per_sample_loss.py     # exactness + gradient-flow checks
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.losses import VGGTOmegaLoss  # noqa: E402


@dataclass
class PerSample:
    """Per-scene losses for one batch.

    `loss` keeps its graph (this is what gets backpropagated); `logs` are detached
    per-scene copies of the criterion's own scalars, so `loss_camera` / `loss_depth`
    can be reweighted or inspected per domain without a second pass.
    """

    loss: torch.Tensor              # (B,), differentiable
    valid: torch.Tensor             # (B,) bool
    logs: dict[str, torch.Tensor]   # each (B,), detached

    def __len__(self) -> int:
        return int(self.loss.shape[0])


def slice_batch(batch: dict, index: int) -> dict:
    """Batch element `index`, keeping the leading axis so shapes stay (1, S, ...)."""
    return {
        k: (v[index : index + 1] if torch.is_tensor(v) or isinstance(v, (list, tuple)) else v)
        for k, v in batch.items()
    }


def per_sample_losses(criterion: VGGTOmegaLoss, predictions: dict, batch: dict) -> PerSample:
    """Evaluate `criterion` on each scene of the batch separately.

    A scene whose depth mask is empty (or whose scale normalisation gave up, i.e.
    `scale_ok` is False) is marked invalid: `depth_loss` and `point_loss` return a
    hard zero there, which would make the scene look *easy* to the reweighter rather
    than unusable. `train.py` drops invalid scenes from the DRO statistics and from
    the objective.
    """
    losses, logs, valid = [], {}, []
    for i in range(next(iter(predictions.values())).shape[0]):
        single_pred = {k: v[i : i + 1] for k, v in predictions.items()}
        single_batch = slice_batch(batch, i)
        loss, parts = criterion(single_pred, single_batch)
        losses.append(loss)
        for k, v in parts.items():
            logs.setdefault(k, []).append(v.reshape(()))

        mask = single_batch.get("depth_mask")
        scale_ok = single_batch.get("scale_ok")
        usable = bool(mask.any()) if mask is not None else True
        if scale_ok is not None:
            usable = usable and bool(scale_ok[0])
        valid.append(usable and bool(torch.isfinite(loss)))

    return PerSample(
        loss=torch.stack(losses),
        valid=torch.tensor(valid, device=losses[0].device),
        logs={k: torch.stack(v) for k, v in logs.items()},
    )


def domain_weighted_loss(
    losses: torch.Tensor,
    domain_ids: np.ndarray,
    alpha: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    valid: torch.Tensor | None = None,
    scheme: str = "scene-ratio",
) -> tuple[torch.Tensor, dict[int, float], dict[int, int]]:
    """DoReMi's objective, as a per-scene reweighting of the plain per-scene mean.

    Returns (objective, per-domain mean loss, per-domain count). Domains absent from
    the batch contribute nothing, and `-1` (a scene the clusters json does not cover)
    is ignored.

    `scheme="scene-ratio"` weights scene i by `alpha_j / baseline_j` for its domain j,
    normalised to mean 1 over the scenes in the step:

        L = sum_i w_i L_i / sum_i w_i,      w_i = alpha_{d(i)} / baseline_{d(i)}

    Two properties this has and `"domain-mean"` does not:

    * **At `alpha == baseline` every weight is exactly 1**, so the objective reduces
      to `mean_i L_i` -- the reference run's own loss under `--per-sample-objective`.
      The proxy therefore starts from the reference's objective and departs from it
      only as far as the mixture departs from baseline, which is what lets the excess
      loss be read as headroom rather than as the difference between two objectives.
    * **A scene's weight does not depend on how many of its domain-mates share the
      step.** That is what went wrong in `runs/doremi-proxy`. `"domain-mean"` is
      DoReMi's literal `sum_j alpha_j * mean_j(losses)`, renormalised by the alpha
      mass present, so scene i carries `alpha_j / (count_j * mass)`. Every minibatch
      in DoReMi contains every domain with many tokens, so `count_j` tracks the
      domain's share and the two agree. Here a step holds ~32 scenes over 48 domains
      and `count_j` is 1 whenever the domain appears at all, so the per-scene weight
      comes out proportional to cluster size instead of flat: measured at corr +1.00
      with log cluster size, a 3.1x spread from the 15-scene cluster to the 65-scene
      one. The proxy consequently undertrained its smallest clusters by ~3x, scored
      worse on them than the reference, and the reweighter read that self-inflicted
      gap as excess loss and upweighted them -- the observed corr(excess, log size)
      of -0.92. `"domain-mean"` is kept to reproduce that run, not to be used; the
      self-test below measures both.

    `baseline` is the mixture a plain shuffled epoch realises (`Domains.baseline()`,
    proportional to cluster size) -- not `--dro-prior`, which may be uniform. It is
    the denominator that makes `w` a ratio, so it is required for `"scene-ratio"`.
    """
    ids = np.asarray(domain_ids)
    keep = ids >= 0
    if valid is not None:
        keep &= valid.detach().cpu().numpy()
    if not keep.any():
        return losses.sum() * 0.0, {}, {}

    alpha = np.asarray(alpha, dtype=np.float64)
    rows_of: dict[int, np.ndarray] = {}
    means: dict[int, float] = {}
    counts: dict[int, int] = {}
    for d in np.unique(ids[keep]):
        rows = np.flatnonzero(keep & (ids == d))
        rows_of[int(d)] = rows
        means[int(d)] = float(losses[torch.as_tensor(rows, device=losses.device)].mean().detach())
        counts[int(d)] = len(rows)

    if scheme == "domain-mean":
        total = losses.sum() * 0.0
        mass = 0.0
        for d, rows in rows_of.items():
            mean = losses[torch.as_tensor(rows, device=losses.device)].mean()
            total = total + float(alpha[d]) * mean
            mass += float(alpha[d])
        return (total / mass if mass > 0 else total), means, counts

    if scheme != "scene-ratio":
        raise ValueError(f"scheme must be 'scene-ratio' or 'domain-mean', got {scheme!r}")
    if baseline is None:
        raise ValueError("scheme='scene-ratio' needs `baseline` (Domains.baseline()); "
                         "without it there is no denominator to make the weight a ratio")

    baseline = np.asarray(baseline, dtype=np.float64)
    rows = np.flatnonzero(keep)
    ratio = alpha[ids[rows]] / np.maximum(baseline[ids[rows]], 1e-12)
    weight = torch.as_tensor(ratio, device=losses.device, dtype=losses.dtype)
    selected = losses[torch.as_tensor(rows, device=losses.device)]
    mass = weight.sum()
    if float(mass) <= 0:
        # Every present domain has zero weight. Returning the unweighted mean keeps
        # the step from silently contributing no gradient.
        return selected.mean(), means, counts
    return (weight * selected).sum() / mass, means, counts


def excess_losses(
    proxy: torch.Tensor, reference: torch.Tensor, *, clamp: bool = True
) -> torch.Tensor:
    """`max(0, L_proxy - L_ref)` per scene -- DoReMi's excess loss.

    The clamp is what makes this a measure of *headroom* rather than of difficulty:
    a cluster the proxy already fits better than the reference has nothing left to
    gain from more samples, so it should not pull weight toward itself. Note the
    clamp also rectifies noise into positive mean excess, which is why a domain with
    few samples per step drifts upward on its own (see `dro.py`'s self-test).

    That rectification is *not* what broke `runs/doremi-proxy`: there the
    domain-aggregated excess correlated -0.92 with log cluster size whether the clamp
    was applied per scene (-0.87), after aggregating (-0.94), or not at all (-0.92),
    so the size dependence was in the signed difference already and came from the
    objective -- see `domain_weighted_loss`.

    It *is* what broke `runs/doremi-proxy-raw`, the from-scratch `scene-ratio` rerun
    that fixed the objective. With the size bias gone (corr with log n +0.22, was
    -0.80) the clamped domain excess turned out to be 90% predicted by the *variance*
    of the per-window difference on its own -- corr(clamped excess, sd of diff) =
    +0.95, R^2 0.897 -- because rectifying a difference centred near zero makes
    E[max(0,d)] a function of sd(d). sd(d) then tracks the domain's loss level
    (+0.71), which tracks the dataset, so the learned mixture correlated -0.77 with
    DL3DV share and +0.08 with the signed excess it was supposed to follow.

    Two things make the difference centred near zero, and both must be fixed:
    `--excess-clamp none` for the estimator, and a proxy with a real capacity deficit
    (`--preset tiny`) so that L_proxy - L_ref is a signal rather than window noise
    around a constant offset. Note `ReferenceCache` compares a single proxy *window*
    against the reference's *scene mean* over repeats, which adds one-sided window
    variance the paired eval in `excess_report.py` does not have -- another reason
    not to rectify.
    """
    diff = proxy.detach() - reference.detach()
    return diff.clamp_min(0.0) if clamp else diff


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #


def _fake_batch(B: int, S: int, H: int, W: int, *, valid_fraction=None, seed: int = 0):
    """A batch and a prediction dict of the shapes `VGGTOmegaLoss` consumes."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g)

    def pose_enc(B, S):
        quat = randn(B, S, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        fov = torch.full((B, S, 2), 0.9)
        return torch.cat([randn(B, S, 3) * 0.1, quat, fov], dim=-1)

    gt_depth = randn(B, S, H, W).abs() + 0.5
    mask = torch.ones(B, S, H, W, dtype=torch.bool)
    if valid_fraction is not None:
        for b, frac in enumerate(valid_fraction):
            keep = torch.rand(S, H, W, generator=g) < frac
            mask[b] = keep
    extrinsics = torch.eye(4, 4)[:3].expand(B, S, 3, 4).contiguous()
    intrinsics = torch.eye(3).expand(B, S, 3, 3).contiguous().clone()
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = float(max(H, W))
    intrinsics[..., 0, 2], intrinsics[..., 1, 2] = W / 2, H / 2

    batch = {
        "pose_enc": pose_enc(B, S),
        "depth": gt_depth,
        "depth_mask": mask,
        "point_map": randn(B, S, H, W, 3),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "scale_ok": torch.ones(B, dtype=torch.bool),
        "scene_scale": torch.ones(B),
        "scene_id": [f"scene{b}" for b in range(B)],
        "subset": ["scannet"] * B,
    }
    predictions = {
        "pose_enc": pose_enc(B, S).requires_grad_(True),
        "depth": (randn(B, S, H, W, 1).abs() + 0.5).requires_grad_(True),
        "depth_conf": (torch.ones(B, S, H, W, 1) + randn(B, S, H, W, 1) * 0.01).requires_grad_(True),
    }
    return batch, predictions


if __name__ == "__main__":
    torch.manual_seed(0)
    criterion = VGGTOmegaLoss(depth_kwargs={"alpha": 0.2})
    B, S, H, W = 4, 3, 16, 24

    # 1. Fully valid masks: every scene has the same valid-pixel count, so the
    #    per-pixel and per-scene normalisations coincide and the two agree exactly.
    batch, predictions = _fake_batch(B, S, H, W)
    total, _ = criterion(predictions, batch)
    ps = per_sample_losses(criterion, predictions, batch)
    gap = abs(ps.loss.mean().item() - total.item())
    print(f"dense GT: batch loss {total.item():.6f}  mean per-scene {ps.loss.mean().item():.6f}  "
          f"gap {gap:.2e}")
    assert gap < 1e-4, "per-scene losses must recover the batch loss on uniform masks"
    assert ps.valid.all() and len(ps) == B

    # 2. Uneven coverage: the gap is the reweighting, and it is not small. This is
    #    the semantic difference the module docstring warns about.
    batch, predictions = _fake_batch(B, S, H, W, valid_fraction=[1.0, 0.5, 0.05, 0.01], seed=1)
    total, _ = criterion(predictions, batch)
    ps = per_sample_losses(criterion, predictions, batch)
    print(f"mixed coverage: batch loss {total.item():.6f}  mean per-scene "
          f"{ps.loss.mean().item():.6f}  per-scene {[round(v, 3) for v in ps.loss.tolist()]}")

    # 3. An unusable scene is flagged rather than scored as easy.
    batch["depth_mask"][2] = False
    batch["scale_ok"][3] = False
    ps = per_sample_losses(criterion, predictions, batch)
    print(f"valid flags: {ps.valid.tolist()} (scene 2 has an empty mask, scene 3 no scale)")
    assert ps.valid.tolist() == [True, True, False, False]

    # 4. The objective at baseline is the plain per-scene mean -- the property that
    #    makes the excess loss a model comparison rather than an objective comparison.
    ids = np.array([0, 0, 7, 7])
    sizes8 = np.array([60, 50, 40, 30, 25, 20, 18, 15], dtype=np.float64)
    base8 = sizes8 / sizes8.sum()
    obj, means, counts = domain_weighted_loss(ps.loss, ids, base8, baseline=base8)
    print(f"at baseline: objective {obj.item():.6f} vs mean per-scene "
          f"{ps.loss.mean().item():.6f}  counts={counts}")
    assert abs(obj.item() - ps.loss.mean().item()) < 1e-6, \
        "alpha == baseline must reproduce the unweighted per-scene mean"

    #    Away from baseline it is the ratio-weighted mean, and domain 7 at 2x its
    #    share pulls the objective toward domain 7's scenes by exactly that ratio.
    tilted = base8.copy()
    tilted[7] *= 2.0
    tilted /= tilted.sum()
    obj_t, means_t, _ = domain_weighted_loss(ps.loss, ids, tilted, baseline=base8)
    w = tilted[ids] / base8[ids]
    expected = float((w * ps.loss.detach().numpy()).sum() / w.sum())
    print(f"domain 7 at {tilted[7] / base8[7]:.2f}x: objective {obj_t.item():.6f} "
          f"vs ratio-weighted mean {expected:.6f}")
    assert abs(obj_t.item() - expected) < 1e-5

    #    With `valid` passed, domain 7 -- both of whose scenes are unusable -- leaves
    #    the objective entirely instead of contributing a zero that reads as easy.
    obj_valid, means_valid, counts_valid = domain_weighted_loss(
        ps.loss, ids, base8, baseline=base8, valid=ps.valid)
    print(f"dropping unusable scenes: counts={counts_valid} objective {obj_valid.item():.6f}")
    assert counts_valid == {0: 2}, counts_valid
    assert abs(obj_valid.item() - means_valid[0]) < 1e-6

    ps_all = per_sample_losses(criterion, predictions, _fake_batch(B, S, H, W, seed=1)[0])
    obj, means, counts = domain_weighted_loss(ps_all.loss, ids, tilted, baseline=base8)
    obj.backward()
    grads = {k: float(v.grad.abs().sum()) for k, v in predictions.items()}
    print(f"weighted objective {obj.item():.6f}  counts={counts}  grad L1: "
          + "  ".join(f"{k}={g:.3g}" for k, g in grads.items()))
    assert all(g > 0 for g in grads.values()), "the objective must reach every head"

    # 5. Domains absent from the batch drop out; -1 (unclustered scene) is ignored.
    obj_partial, means_partial, counts_partial = domain_weighted_loss(
        ps_all.loss, np.array([0, -1, -1, 7]), base8, baseline=base8
    )
    assert counts_partial == {0: 1, 7: 1}, counts_partial
    print(f"unclustered scenes ignored: counts={counts_partial}")

    # 6. The regression that cost `runs/doremi-proxy`: at the *baseline* mixture, is
    #    a scene's effective gradient weight flat in its cluster's size?
    #
    #    Replayed on this repo's actual shape -- 48 clusters of 15..65 scenes, 32
    #    scenes per weight update, drawn proportional to cluster size. The weight a
    #    scene carries per visit is d(objective)/d(L_i), and `scene-ratio` is
    #    1/n_present by construction while `domain-mean` is alpha_j/(count_j * mass),
    #    which correlates +1.00 with log cluster size.
    print("\n  per-scene gradient weight at the baseline mixture, by cluster size:")
    rng = np.random.default_rng(0)
    # This repo's shape: 48 clusters, median 57 scenes, a tail of small ones.
    sizes = np.array([65, 64, 63, 63, 62, 62, 62, 61, 61, 61, 61, 61, 60, 60, 60, 60,
                      59, 58, 57, 57, 56, 56, 55, 55, 54, 53, 52, 51, 50, 49, 47, 46,
                      45, 45, 44, 41, 38, 37, 34, 30, 28, 26, 25, 25, 25, 24, 19, 15],
                     dtype=np.float64)
    base = sizes / sizes.sum()
    for scheme in ("scene-ratio", "domain-mean"):
        total = np.zeros(48)
        visits = np.zeros(48)
        for _ in range(4_000):
            drawn = rng.choice(48, size=32, p=base)
            losses = torch.zeros(32, requires_grad=True)
            obj, _, _ = domain_weighted_loss(
                losses, drawn, base, baseline=base, scheme=scheme)
            (g,) = torch.autograd.grad(obj, losses)
            np.add.at(total, drawn, g.numpy())
            np.add.at(visits, drawn, 1.0)
        per_visit = total / np.maximum(visits, 1)
        spread = per_visit.max() / per_visit.min()
        # `scene-ratio` at baseline is exactly 1/n_present for every scene, so the
        # weight vector has zero variance and the correlation is undefined -- which
        # is the ideal outcome, not a failure. Report the spread in that case.
        r = (float(np.corrcoef(per_visit, np.log(sizes))[0, 1]) if per_visit.std() > 0
             else float("nan"))
        shown = "flat (zero variance)" if np.isnan(r) else f"{r:+.2f}"
        print(f"    {scheme:<13s} corr(weight, log n) = {shown:<20s} spread {spread:.2f}x"
              f"   (n={int(sizes.min())} -> {per_visit[np.argmin(sizes)]:.4f}, "
              f"n={int(sizes.max())} -> {per_visit[np.argmax(sizes)]:.4f})")
        if scheme == "scene-ratio":
            assert spread < 1.05, f"scene-ratio weight must be flat in size (got {spread:.2f}x)"
            assert np.isnan(r) or abs(r) < 0.2, \
                f"scene-ratio weight must not track cluster size (got {r:+.2f})"
        else:
            assert r > 0.9, "the domain-mean scheme is supposed to exhibit the bug"

    #    Away from baseline the weight must track the *ratio* a domain was given and
    #    still not its size -- otherwise the flatness above is only an artifact of
    #    every ratio being 1. Tilt the mixture independently of size and check.
    tilt = np.exp(rng.normal(0, 0.4, size=48))
    tilted_mix = base * tilt
    tilted_mix /= tilted_mix.sum()
    want = tilted_mix / base
    total, visits = np.zeros(48), np.zeros(48)
    for _ in range(4_000):
        drawn = rng.choice(48, size=32, p=base)
        losses = torch.zeros(32, requires_grad=True)
        obj, _, _ = domain_weighted_loss(losses, drawn, tilted_mix, baseline=base)
        (g,) = torch.autograd.grad(obj, losses)
        np.add.at(total, drawn, g.numpy())
        np.add.at(visits, drawn, 1.0)
    got = total / np.maximum(visits, 1)
    print(f"    tilted mixture (ratio {want.min():.2f}..{want.max():.2f}, uncorrelated "
          f"with size):")
    print(f"      corr(weight, alpha/baseline) = {np.corrcoef(got, want)[0, 1]:+.2f}   "
          f"corr(weight, log n) = {np.corrcoef(got, np.log(sizes))[0, 1]:+.2f}")
    assert np.corrcoef(got, want)[0, 1] > 0.99, "the weight must be the ratio it was given"
    assert abs(np.corrcoef(got, np.log(sizes))[0, 1]) < 0.2, \
        "the weight must still not track cluster size away from baseline"
    print("  -> the reference run's plain criterion(batch) is flat in n; only")
    print("     `scene-ratio` matches it, which is what makes the excess loss valid.")

    # 6. Excess loss is one-sided.
    proxy = torch.tensor([1.0, 0.5, 2.0])
    ref = torch.tensor([1.5, 0.5, 1.0])
    print("excess:", excess_losses(proxy, ref).tolist())
    assert excess_losses(proxy, ref).tolist() == [0.0, 0.0, 1.0]
    print("per_sample_loss self-test ok")
