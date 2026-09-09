"""Two questions about a DoReMi run: is there anything to find, and did it find it.

**Before the three runs** -- `--eval-jsonl`. Join a per-scene eval (an
`evaluate.py --out` windows jsonl, over the *training* scenes) against the clusters
and report how much of the variance in per-scene loss the cluster assignment
explains. Reweighting domains can only help if the domains differ; if eta^2 is ~0
then per-scene loss is unrelated to which visual mode a scene belongs to, and the
mixture DoReMi learns will be fitting sampling noise. This costs one eval pass
against a checkpoint you already have, so it is worth doing first.

    python3 doremi/report.py --eval-jsonl runs/mixed/eval/windows-mixed-final.jsonl

**After the proxy run** -- `--weights` (and `--history` for the trajectory). Prints
the learned mixture, the biggest movers, and the two confounds worth ruling out
before believing it: whether the weights are tracking cluster *size* (a small domain
drifts up on rectified noise) and whether they are tracking the DL3DV/ScanNet split
(the two datasets differ in depth-GT quality, so a "hard" cluster may just be a
cluster with worse ground truth -- the reference model is supposed to subtract that
floor, and this is the check that it did).

    python3 doremi/report.py --weights runs/doremi-proxy/domain_weights.json \\
        --history runs/doremi-proxy/domain_weights.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doremi.domains import format_table, load_domains, load_weights  # noqa: E402
from doremi.dro import effective_sample_size  # noqa: E402


def read_windows(path: Path, key: str) -> dict[str, float]:
    """`scene_id` -> mean of `key` over that scene's windows, from an eval jsonl."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            value = row.get(key)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            scene = row["scene_id"]
            sums[scene] = sums.get(scene, 0.0) + float(value)
            counts[scene] = counts.get(scene, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def spread_report(domains, per_scene: dict[str, float], key: str, top: int = 8) -> None:
    """Per-cluster mean loss, and how much of the variance the clusters explain."""
    by_name: dict[str, tuple[int, str]] = {}
    for key_ in domains.scene_to_domain:
        scene = key_.split("/", 1)[1]
        by_name.setdefault(scene, (domains.scene_to_domain[key_],
                                   domains.scene_to_dataset.get(key_, "?")))

    rows = [(v, *by_name[scene]) for scene, v in per_scene.items() if scene in by_name]
    if len(rows) < 2:
        raise SystemExit(f"only {len(rows)} scenes matched the clusters json; "
                         "is this eval over the training scenes?")
    values = np.array([r[0] for r in rows], dtype=np.float64)
    doms = np.array([r[1] for r in rows])
    dsets = np.array([r[2] for r in rows])
    populated = np.unique(doms)

    print(f"{key} over {len(values)} scenes in {len(populated)}/{len(domains)} clusters "
          f"(grand mean {values.mean():.4f}, sd {values.std():.4f})")
    for name in sorted(set(dsets)):
        m = dsets == name
        print(f"  {name:>8s}: {int(m.sum()):>5d} scenes, mean {values[m].mean():.4f}")

    raw = _eta2(values, doms)
    raw_null = _eta2_null(values, doms)
    by_dataset = _eta2(values, dsets)
    # Residualise on dataset -- subtract each dataset's own mean -- and re-measure.
    # 46 of the 48 domains are >95% one dataset, and the two differ systematically in
    # depth-GT quality (ScanNet is a metric sensor, DL3DV's is estimated), so the raw
    # eta^2 partly counts "which dataset" rather than "which visual mode". Only the
    # residual part is something a *mixture over clusters* can act on that a
    # --dl3dv-weight / --scannet-weight pair could not; and the null has to shuffle
    # within dataset to match.
    resid = values.copy()
    for name in np.unique(dsets):
        m = dsets == name
        resid[m] -= resid[m].mean()
    within = _eta2(resid, doms)
    within_null = _eta2_null(resid, doms, blocks=dsets)

    print(f"\nvariance explained:")
    print(f"  by cluster (raw)                 eta^2 = {raw:.3f}"
          f"   null: mean {raw_null[0]:.3f}, p95 {raw_null[1]:.3f}")
    share = f"   ({by_dataset / raw:.0%} of the raw cluster effect)" if raw > 0 else ""
    print(f"  by dataset alone                 eta^2 = {by_dataset:.3f}{share}")
    print(f"  by cluster, dataset removed      eta^2 = {within:.3f}"
          f"   null: mean {within_null[0]:.3f}, p95 {within_null[1]:.3f}")

    if within > within_null[1]:
        print(f"  -> above the null within dataset: clusters separate loss for reasons beyond "
              f"provenance, so a mixture over clusters has something to act on that "
              f"--dl3dv-weight / --scannet-weight does not")
    elif raw > raw_null[1]:
        print(f"  -> the cluster effect is essentially the dataset effect. Tune "
              f"--dl3dv-weight / --scannet-weight instead; 50 domains would buy nothing")
    else:
        print(f"  -> inside the null: per-scene loss looks unrelated to the clusters, so a "
              f"learned mixture would be fitting noise")

    means = sorted(
        ((values[doms == d].mean(), values[doms == d].std(), int((doms == d).sum()), int(d))
         for d in populated), reverse=True)
    print(f"\nhardest and easiest clusters by mean {key}:")
    print(f"{'cluster':>8s} {'n':>4s} {'mean':>9s} {'sd':>8s}  datasets")
    shown = (means[:top] + [(None, 0.0, 0, None)] + means[-top:]) if len(means) > 2 * top else means
    for mean, sd, n, d in shown:
        if d is None:
            print(f"{'...':>8s}")
            continue
        mix = domains.meta[d].get("datasets") or {}
        mix_s = " ".join(f"{k}={v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1]))
        print(f"{domains.ids[d]:>8d} {n:>4d} {mean:>9.4f} {sd:>8.4f}  {mix_s}")


def _eta2(y: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance in `y` that lies between the levels of `groups`."""
    grand = y.mean()
    total = ((y - grand) ** 2).sum()
    if total <= 0:
        return 0.0
    between = sum(int((groups == g).sum()) * (y[groups == g].mean() - grand) ** 2
                  for g in np.unique(groups))
    return float(between / total)


def _eta2_null(y: np.ndarray, groups: np.ndarray, *, blocks: np.ndarray | None = None,
               draws: int = 200, seed: int = 0) -> tuple[float, float]:
    """(mean, p95) of eta^2 under random reassignment, keeping the group sizes.

    eta^2 is biased upward by the number of groups -- 48 clusters over 2341 scenes
    explain ~2% of any noise -- so the statistic is only readable against this null.
    `blocks` shuffles within each block (pass the dataset labels when `y` has already
    been residualised on them), so the null keeps whatever structure the residual
    already removed.
    """
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(draws):
        shuffled = y.copy()
        if blocks is None:
            rng.shuffle(shuffled)
        else:
            for b in np.unique(blocks):
                m = blocks == b
                block = shuffled[m]
                rng.shuffle(block)
                shuffled[m] = block
        nulls.append(_eta2(shuffled, groups))
    return float(np.mean(nulls)), float(np.percentile(nulls, 95))


def weights_report(domains, alpha: np.ndarray, payload: dict, top: int = 10) -> None:
    baseline = domains.baseline()
    ratio = alpha / np.maximum(baseline, 1e-12)
    sizes = np.array(domains.sizes, dtype=np.float64)
    print(f"learned mixture: {payload.get('dro_steps', '?')} weight updates, "
          f"{payload.get('scenes_per_update', '?')} scenes each, eta={payload.get('eta', '?')} "
          f"({payload.get('eta_mode', '?')}), prior={payload.get('prior', '?')}, "
          f"max_ratio={payload.get('max_ratio')}")
    print(f"ratio: min={ratio.min():.2f} median={np.median(ratio):.2f} max={ratio.max():.2f}  "
          f"ess={effective_sample_size(alpha):.1f}/{len(domains)}")
    at_cap = payload.get("max_ratio")
    if at_cap:
        pinned = int((ratio > 0.99 * at_cap).sum() + (ratio < 1.01 / at_cap).sum())
        if pinned:
            print(f"  {pinned} domains are pinned at +-max_ratio: the budget bound, not the data, "
                  f"is setting their weight")
    print()
    print(format_table(domains, alpha, excess=np.array(payload.get("excess_loss", [np.nan] * len(domains))),
                       limit=top))

    print("\nconfounds:")
    print(f"  corr(log ratio, log cluster size) = {_corr(np.log(ratio), np.log(sizes)):+.2f}"
          "   (strongly negative = the weights are tracking domain size, i.e. noise)")
    dl3dv_share = np.array([
        (m.get("datasets") or {}).get("dl3dv", 0) / max(s, 1)
        for m, s in zip(domains.meta, domains.sizes)
    ])
    print(f"  corr(log ratio, DL3DV share)      = {_corr(np.log(ratio), dl3dv_share):+.2f}"
          "   (large either way = the split the reference model was supposed to cancel)")
    seen = np.array(payload.get("seen", [0] * len(domains)))
    if seen.any():
        unseen = int((seen == 0).sum())
        print(f"  samples per domain: min={seen.min()} median={int(np.median(seen))} "
              f"max={seen.max()}" + (f", {unseen} never sampled" if unseen else ""))


def history_report(path: Path, domains, top: int = 6) -> None:
    """Whether the mixture settled or was still moving when the run ended."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    steps = [r["step"] for r in rows]
    alphas = np.array([r["alpha_bar"] if any(r["alpha_bar"]) else r["alpha"] for r in rows])
    ess = [effective_sample_size(a) for a in alphas]
    print(f"\ntrajectory over {len(rows)} snapshots (steps {steps[0]}..{steps[-1]}):")
    print(f"{'step':>8s} {'ess':>7s} {'ratio_max':>10s} {'ratio_min':>10s} {'move':>8s}")
    baseline = domains.baseline()
    for i in range(0, len(rows), max(len(rows) // 8, 1)):
        ratio = alphas[i] / baseline
        move = float(np.abs(alphas[i] - alphas[i - 1]).sum()) if i else float("nan")
        print(f"{steps[i]:>8d} {ess[i]:>7.1f} {ratio.max():>10.2f} {ratio.min():>10.2f} {move:>8.4f}")
    drift = float(np.abs(alphas[-1] - alphas[max(len(alphas) - 2, 0)]).sum())
    trend = ess[-1] - ess[len(ess) // 2]
    print(f"  last snapshot moved {drift:.4f} in total variation; ess changed by {trend:+.1f} "
          f"over the second half")
    if trend < -0.05 * len(domains):
        print("  -> ess still falling at the end: the weights had not equilibrated, so they are "
              "partly a function of how long the proxy ran. Lower --dro-budget or run longer")

    final = alphas[-1] / baseline
    order = np.argsort(-np.abs(np.log(final)))
    print(f"\nbiggest movers:")
    for d in order[:top]:
        series = " ".join(f"{alphas[i][d] / baseline[d]:.2f}"
                          for i in range(0, len(rows), max(len(rows) // 6, 1)))
        print(f"  {domains.label(d):<34s} {series}")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clusters", type=Path, default=None)
    p.add_argument("--min-domain-size", type=int, default=10,
                   help="match the trainer's, so the domain set is the same one")
    p.add_argument("--eval-jsonl", type=Path, default=None,
                   help="evaluate.py windows jsonl over the training scenes (the go/no-go check)")
    p.add_argument("--metric", default="loss", help="which jsonl field to spread-test")
    p.add_argument("--weights", type=Path, default=None, help="a domain_weights.json")
    p.add_argument("--history", type=Path, default=None, help="a domain_weights.jsonl")
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    if not any((args.eval_jsonl, args.weights, args.history)):
        p.error("pass at least one of --eval-jsonl / --weights / --history")

    domains = load_domains(args.clusters, min_size=args.min_domain_size)
    print(f"{domains.path.name}: {len(domains)} domains over {domains.n_scenes} scenes\n")

    if args.eval_jsonl:
        spread_report(domains, read_windows(args.eval_jsonl, args.metric), args.metric, top=args.top)
    if args.weights:
        payload = json.loads(args.weights.read_text())
        weights_report(domains, load_weights(args.weights, domains), payload, top=args.top)
    if args.history:
        history_report(args.history, domains, top=args.top)
