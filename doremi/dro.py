"""The Group DRO weight optimiser from DoReMi (Xie et al., 2023), Algorithm 1.

Per step: measure each domain's *excess* loss against a reference model, push the
weights up the exponentiated gradient of that, renormalise, smooth, and keep a
running average. The average -- not the final iterate -- is what DoReMi reports as
the learned mixture, because the per-step weights oscillate.

    lambda_j = mean_j max(0, L_proxy - L_ref)
    alpha   <- alpha * exp(eta * lambda)          then normalise
    alpha   <- (1 - c) * alpha + c * prior        then average into alpha_bar

Two adaptations, both forced by the domain count here being 50 over ~2300 scenes
rather than 22 over billions of tokens:

**Absent domains use their last estimate.** DoReMi's every minibatch contains every
domain, so `lambda` is fully observed each step. A step here sees `batch_size *
grad_accum` scenes across 50 domains, so most domains are absent from most steps.
Leaving them at zero excess would decay their weight simply for not being sampled --
an absent domain would lose weight to a present one on no evidence. Instead each
domain keeps an EMA of its own excess loss (`--dro-ema`), updated only on the steps
where it appears, and the update uses the full EMA vector. A domain that has never
been seen contributes zero, which leaves its weight untouched up to renormalisation.

**Smoothing pulls toward a prior, not toward uniform.** Passing `prior=baseline`
(what `train.py` does) means the floor under each domain is its natural share, so
the two 2-scene clusters cannot be handed 1/50th of the sample stream by the
smoothing term alone. Pass `prior=domains.uniform()` for the paper's behaviour.

`max_ratio` is a third, optional departure. The update compounds -- after T steps
`alpha_j ~ prior_j * exp(eta * sum_t lambda_j(t))` -- and what keeps that from
collapsing onto one domain is the game: oversampling a domain fits it, which lowers
its excess loss, which lowers its weight. That feedback needs the proxy to actually
learn from the reweighting, so if the proxy run is short (or `eta` is large) the
weights can run away before it does. `max_ratio` bounds `alpha / prior` so a
25-scene cluster cannot end up claiming 20x its share of the sample stream, which is
oversampling well past where it stops being informative. Watch `ess` in the log: if
it is falling monotonically the game has not equilibrated.

    python3 doremi/dro.py        # synthetic checks: equilibrium, and runaway
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DomainReweighter:
    """Exponentiated-gradient ascent on domain weights, with EMA-smoothed losses.

    `alpha` is the live iterate used to weight the training objective; `alpha_bar`
    is the running average and the thing to write out at the end.
    """

    n_domains: int
    eta: float = 1.0            # DoReMi's domain weight step size
    smoothing: float = 1e-3     # DoReMi's c
    ema: float = 0.9            # decay for the per-domain excess-loss estimate
    prior: np.ndarray | None = None   # smoothing target; defaults to uniform
    init: np.ndarray | None = None    # starting weights; defaults to `prior`
    max_ratio: float | None = None    # cap on alpha / prior (None = DoReMi's, uncapped)

    alpha: np.ndarray = field(init=False)
    alpha_bar: np.ndarray = field(init=False)
    excess: np.ndarray = field(init=False)      # EMA of per-domain excess loss
    seen: np.ndarray = field(init=False)        # samples contributed, per domain
    steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.prior is None:
            self.prior = np.full(self.n_domains, 1.0 / self.n_domains)
        self.prior = np.asarray(self.prior, dtype=np.float64)
        if self.prior.shape != (self.n_domains,):
            raise ValueError(f"prior must be ({self.n_domains},), got {self.prior.shape}")
        start = self.prior if self.init is None else np.asarray(self.init, dtype=np.float64)
        self.alpha = start / start.sum()
        # alpha_bar starts empty rather than at alpha, so a 1-step run reports the
        # step it actually took.
        self.alpha_bar = np.zeros(self.n_domains)
        self.excess = np.zeros(self.n_domains)
        self.seen = np.zeros(self.n_domains, dtype=np.int64)

    # ---- the update ----

    def observe(self, excess_sum: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Fold one step's per-domain excess-loss sums into the EMA. Returns the EMA.

        `excess_sum[j]` is the sum of `max(0, L_proxy - L_ref)` over the samples of
        domain j in this step, `counts[j]` how many there were. Domains with no
        samples are left alone.
        """
        counts = np.asarray(counts, dtype=np.float64)
        excess_sum = np.asarray(excess_sum, dtype=np.float64)
        present = counts > 0
        if present.any():
            observed = np.zeros(self.n_domains)
            observed[present] = excess_sum[present] / counts[present]
            fresh = self.seen == 0
            # First observation replaces rather than blends: blending into a zero
            # init would halve every domain's first estimate and make early steps
            # depend on arrival order.
            replace = present & fresh
            blend = present & ~fresh
            self.excess[replace] = observed[replace]
            self.excess[blend] = self.ema * self.excess[blend] + (1 - self.ema) * observed[blend]
            self.seen += counts.astype(np.int64)
        return self.excess

    def step(self, excess_sum: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """`observe`, then one exponentiated-gradient update. Returns the new alpha."""
        excess = self.observe(excess_sum, counts)
        # Shift by the max before exponentiating: exp(eta * lambda) overflows for a
        # diverged proxy, and subtracting a constant from every exponent is exactly
        # the renormalisation on the next line.
        logits = self.eta * (excess - excess.max())
        alpha = self.alpha * np.exp(logits)
        total = alpha.sum()
        alpha = self.prior.copy() if total <= 0 or not np.isfinite(total) else alpha / total
        self.alpha = (1 - self.smoothing) * alpha + self.smoothing * self.prior
        self.alpha /= self.alpha.sum()
        if self.max_ratio is not None:
            self.alpha = _clip_to_ratio(self.alpha, self.prior, self.max_ratio)
        self.steps += 1
        self.alpha_bar += (self.alpha - self.alpha_bar) / self.steps
        return self.alpha

    # ---- reporting ----

    def state(self) -> dict:
        return {
            "steps": self.steps,
            "eta": self.eta,
            "smoothing": self.smoothing,
            "ema": self.ema,
            "alpha": self.alpha.tolist(),
            "alpha_bar": self.alpha_bar.tolist(),
            "excess": self.excess.tolist(),
            "seen": self.seen.tolist(),
        }

    def load_state(self, state: dict) -> None:
        self.alpha = np.asarray(state["alpha"], dtype=np.float64)
        self.alpha_bar = np.asarray(state["alpha_bar"], dtype=np.float64)
        self.excess = np.asarray(state["excess"], dtype=np.float64)
        self.seen = np.asarray(state["seen"], dtype=np.int64)
        self.steps = int(state["steps"])

    def summary(self, ratio_to: np.ndarray | None = None) -> str:
        ref = self.prior if ratio_to is None else np.asarray(ratio_to, dtype=np.float64)
        ratio = self.alpha_bar / np.maximum(ref, 1e-12)
        unseen = int((self.seen == 0).sum())
        return (f"alpha_bar ratio: min={ratio.min():.2f} max={ratio.max():.2f} "
                f"ess={effective_sample_size(self.alpha_bar):.1f}/{self.n_domains}"
                + (f"  unseen_domains={unseen}" if unseen else ""))


def auto_eta(excess: np.ndarray, seen: np.ndarray, *, total_logit: float, planned_steps: int) -> float:
    """A step size in units of the excess loss actually being measured.

    `eta` is not dimensionless: it multiplies an excess loss, and DoReMi's eta=1.0 is
    calibrated to per-token cross-entropy, where a domain sits ~0.01-0.5 nats from
    the reference. This objective is `5*L_cam + L_depth + 0.5*L_point` and its excess
    runs 1-6 early in a run, so eta=1.0 exponentiates a spread of ~3 every step and
    saturates the mixture within a handful of steps -- observed, not hypothetical.

    So instead of fixing eta, fix what the *run* is allowed to do: a domain that
    stays one cross-domain standard deviation above average for the whole proxy run
    ends at `exp(total_logit)` times its prior share. That makes the knob
    `total_logit` (`--dro-budget`), which is in ratio units and comparable across
    loss weightings, model sizes and step counts.

    Returns 0 -- no movement -- until at least two domains have been measured, since
    a spread over one domain is not a spread.
    """
    observed = np.asarray(excess)[np.asarray(seen) > 0]
    if observed.size < 2 or planned_steps <= 0:
        return 0.0
    spread = float(observed.std())
    if not np.isfinite(spread) or spread <= 0:
        return 0.0
    return total_logit / (planned_steps * spread)


def _clip_to_ratio(alpha: np.ndarray, prior: np.ndarray, max_ratio: float,
                   iterations: int = 32) -> np.ndarray:
    """Hold alpha inside `prior / max_ratio <= alpha <= prior * max_ratio`, summing to 1.

    Clipping breaks the normalisation and renormalising breaks the clip, so the mass
    a clip removes is water-filled back onto the domains that are still off their
    bounds, in proportion to their current weight. The loop ends on a clip, so the
    ratio bound holds exactly and the sum is the quantity left approximate -- the
    right way round, since the bound is the thing being promised.

    Always feasible for `max_ratio >= 1`: the floors sum to `1/max_ratio <= 1` and
    the ceilings to `max_ratio >= 1`.
    """
    if max_ratio < 1.0:
        raise ValueError(f"max_ratio must be >= 1, got {max_ratio}")
    lo, hi = prior / max_ratio, prior * max_ratio
    alpha = np.clip(alpha, lo, hi)
    for _ in range(iterations):
        residual = 1.0 - alpha.sum()
        if abs(residual) < 1e-12:
            break
        free = alpha < hi - 1e-15 if residual > 0 else alpha > lo + 1e-15
        share = alpha[free].sum()
        if not free.any() or share <= 0:
            break
        alpha = alpha.copy()
        alpha[free] += residual * alpha[free] / share
        alpha = np.clip(alpha, lo, hi)
    return alpha


def effective_sample_size(alpha: np.ndarray) -> float:
    """1 / sum(alpha^2): how many domains the mixture effectively spreads over.

    Equal to `n_domains` for a uniform mixture and 1 for a mixture that collapsed
    onto one domain, so it is the one number that says whether the weights are still
    a mixture or have turned into a selection.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    total = (alpha ** 2).sum()
    return float(1.0 / total) if total > 0 else 0.0


if __name__ == "__main__":
    D, steps, batch = 6, 600, 8
    # Cluster sizes in the shape this repo's clusters json has: a few big ones, one
    # small, one tiny -- so small domains genuinely appear in only a few steps.
    sizes = np.array([60, 60, 60, 60, 20, 2], dtype=np.float64)
    prior = sizes / sizes.sum()
    # Domain 3 is hard, domain 0 is fitted better than the reference (negative
    # excess, clamped to 0 by the objective), the rest are noise.
    base_excess = np.array([-0.20, 0.0, 0.0, 0.30, 0.05, 0.0])

    def run(*, adaptive: bool, seed: int = 0, **kwargs) -> DomainReweighter:
        """`adaptive` closes DoReMi's loop: oversampling a domain fits it.

        Standing in for the proxy model, a domain's excess loss decays with the
        weight it has been given. That feedback is what makes the weights settle at
        a mixture; with `adaptive=False` the excess never responds and the
        exponentiated gradient compounds without limit, which is the failure mode
        `max_ratio` and the `ess` log line exist to catch.
        """
        rng = np.random.default_rng(seed)
        dro = DomainReweighter(D, eta=1.0, smoothing=1e-3, ema=0.9, prior=prior, **kwargs)
        for _ in range(steps):
            drawn = rng.choice(D, size=batch, p=prior)  # batches come from the baseline
            excess_sum, counts = np.zeros(D), np.zeros(D)
            fitted = (0.15 * np.log(np.maximum(dro.alpha / prior, 1e-6)) if adaptive
                      else np.zeros(D))
            for d in drawn:
                counts[d] += 1
                excess_sum[d] += max(0.0, base_excess[d] - fitted[d] + rng.normal(0, 0.05))
            dro.step(excess_sum, counts)
        return dro

    equilibrium = run(adaptive=True)
    print("equilibrium (proxy responds to the reweighting)")
    print("  base excess:", base_excess)
    print("  EMA excess :", np.round(equilibrium.excess, 3))
    print("  ratio      :", np.round(equilibrium.alpha_bar / prior, 3))
    print("  seen       :", equilibrium.seen, "|", equilibrium.summary())

    assert abs(equilibrium.alpha_bar.sum() - 1.0) < 1e-9, "alpha_bar must be a distribution"
    ratio = equilibrium.alpha_bar / prior
    well_sampled = equilibrium.seen >= 100
    assert ratio[3] == ratio[well_sampled].max(), "the hard domain should gain the most weight"
    assert ratio[0] == ratio.min(), "the easy domain should lose weight"
    assert equilibrium.alpha.min() > 0, "smoothing must keep every domain alive"
    assert effective_sample_size(equilibrium.alpha_bar) > 2.0, "should stay a mixture"

    # Domain 5 (2 scenes) outruns the genuinely hard domain 3 even though its true
    # excess is zero, and that is not a bug in the optimiser -- it is the two ways a
    # small domain misleads it, both of which get worse as k grows:
    #   * max(0, .) rectifies symmetric noise into positive mean excess, and the
    #     fewer samples a domain contributes the larger that bias is;
    #   * the feedback that pulls a weight back down is the proxy fitting the
    #     domain, and a domain seen 37 times cannot be fitted, so nothing pulls.
    # Hence `--min-domain-size` and a default `--max-ratio` in train.py.
    print(f"  tiny domain (n={sizes[5]:.0f}, true excess 0) reached "
          f"{ratio[5]:.1f}x on {equilibrium.seen[5]} samples -- see the comment below")
    assert ratio[5] > ratio[3], "the demo is meant to exhibit the small-domain runaway"

    # eta in units of the excess loss: the same weights, learned on a loss scaled up
    # 20x (this repo's total loss against per-token cross-entropy), should land in
    # the same place -- which is what `auto_eta` is for and a fixed eta is not.
    scale = 20.0
    scaled = DomainReweighter(D, eta=1.0, smoothing=1e-3, ema=0.9, prior=prior)
    plain = DomainReweighter(D, eta=1.0, smoothing=1e-3, ema=0.9, prior=prior)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        drawn = rng.choice(D, size=batch, p=prior)
        excess_sum, counts = np.zeros(D), np.zeros(D)
        for d in drawn:
            counts[d] += 1
            excess_sum[d] += max(0.0, base_excess[d] + rng.normal(0, 0.05))
        for dro, factor in ((plain, 1.0), (scaled, scale)):
            dro.eta = auto_eta(dro.excess, dro.seen, total_logit=np.log(4.0), planned_steps=steps)
            dro.step(excess_sum * factor, counts)
    drift = np.abs(plain.alpha_bar - scaled.alpha_bar).max()
    print(f"auto_eta: same weights at 1x and {scale:.0f}x loss scale, max drift {drift:.2e} "
          f"(eta {plain.eta:.4g} vs {scaled.eta:.4g})")
    assert drift < 1e-6, "auto_eta should make the mixture invariant to the loss scale"
    assert np.max(plain.alpha_bar / prior) < 4.0, "the budget should bound the run's total move"

    runaway = run(adaptive=False)
    capped = run(adaptive=False, max_ratio=3.0)
    print(f"runaway  (excess fixed):        ess={effective_sample_size(runaway.alpha):.2f}"
          f"  max ratio={np.max(runaway.alpha / prior):.1f}x")
    print(f"runaway with max_ratio=3.0:     ess={effective_sample_size(capped.alpha):.2f}"
          f"  max ratio={np.max(capped.alpha / prior):.1f}x")
    assert effective_sample_size(runaway.alpha) < 1.5, "fixed excess should collapse the mixture"
    assert np.max(capped.alpha / prior) <= 3.0 + 1e-6, "max_ratio must bound the ratio"
    assert np.min(capped.alpha / prior) >= 1 / 3.0 - 1e-6, "max_ratio must floor the ratio too"

    # A domain never sampled keeps its share up to renormalisation, rather than
    # being decayed for its absence.
    quiet = DomainReweighter(3, prior=np.array([0.5, 0.3, 0.2]), max_ratio=4.0)
    for _ in range(50):
        quiet.step(np.array([0.0, 0.1, 0.0]), np.array([1, 1, 0]))
    print(f"never-sampled domain: {quiet.alpha[2] / 0.2:.2f}x its prior share "
          f"(vs {quiet.alpha[1] / 0.3:.2f}x for the domain with excess loss)")
    assert quiet.alpha[1] / 0.3 > quiet.alpha[2] / 0.2, "evidence should beat absence"
    print("dro self-test ok")
