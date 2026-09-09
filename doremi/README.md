# DoReMi over the patch-average clusters

Domain reweighting in the sense of [DoReMi](https://github.com/sangmichaelxie/doremi)
(Xie et al., 2023), with the 50 clusters of `patch_avg_clustering/cluster.py` as the
domains: learn how much of the training stream each *visual mode* of the pooled
DL3DV + ScanNet set deserves, by training a small proxy model whose sampling weights
follow the Group DRO update on excess loss against a reference model.

This generalises what `--dl3dv-weight` / `--scannet-weight` already do by hand for two
domains. The clusters are ~48 usable domains over ~2340 scenes, so the reweighting is
over visual modes rather than over dataset provenance.

| file | what it is |
| --- | --- |
| `domains.py` | clusters json -> domain map, baseline mixture, weights file, per-scene factors |
| `dro.py` | the exponentiated-gradient weight update, EMA-smoothed, with a ratio cap |
| `per_sample_loss.py` | per-scene losses out of `VGGTOmegaLoss`, and the alpha-weighted objective |
| `sampling.py` | drawing an epoch from a learned mixture, DDP-sharded |
| `train.py` | the reference / proxy / final runs |
| `report.py` | the go/no-go check beforehand, the confound checks afterwards |
| `excess_report.py` | the second gate: is the *excess* loss about hardness or about cluster size? |
| `run_raw_clusters.sh` | the short recipe end to end, over the unpruned clusters, one stage at a time |

Every module runs standalone as its own self-test (`python3 doremi/dro.py`, etc.);
none of them needs a GPU except `train.py`.

## 0. Is there anything to reweight?

Reweighting domains can only help if the domains differ in loss. That is one eval
pass with a checkpoint you already have, over the *training* scenes:

Pick a checkpoint that was *trained on this pool*: the per-scene losses are then
training losses, which is also what DoReMi's excess loss is measured on.
`patch_avg_clustering/subset_pct100.txt` is the pool the `runs/80pct`, `runs/sota` and
`runs/patch_avg*` runs used, and it is `pool.txt` plus the 4 scenes `--min-size 10`
drops.

`evaluate.py` takes one dataset root at a time, so the pool needs two passes and a
`cat`. Do not read the DL3DV-only half as the answer: 42 of the 50 clusters hold
scenes from one dataset only, so half the pool is half the domains.

```bash
# the clustered pool, at the same --min-size the trainer will use
python3 doremi/domains.py --min-size 10 --scene-list doremi/tmp/pool.txt

python training/evaluate.py --checkpoint runs/80pct/final.pt --split train \
    --scene-list doremi/tmp/pool.txt \
    --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only \
    --repeats 3 --out runs/80pct/eval-train-dl3dv

python training/evaluate.py --checkpoint runs/80pct/final.pt --split train \
    --scene-list doremi/tmp/pool.txt \
    --data-root ~/scannet-train --depth-root ~/scannet-train/depth --no-dense-only \
    --repeats 3 --out runs/80pct/eval-train-scannet

# the jsonl is named windows-<run>-<checkpoint>.jsonl, not windows-final.jsonl
cat runs/80pct/eval-train-{dl3dv,scannet}/windows-80pct-final.jsonl \
    > runs/80pct/eval-train.jsonl

python3 doremi/report.py --eval-jsonl runs/80pct/eval-train.jsonl
```

About 20 minutes for the pair at ~10 windows/s on one 4090; `--repeats 1` gives a
rough read in a third of that.

`report.py` prints the share of per-scene loss variance the cluster assignment
explains, against the null of the same cluster sizes over shuffled scenes. If eta^2
sits inside the null, stop here: the learned mixture would be fitting sampling noise,
and the three runs below are not worth the GPU time.

### 0b. Is the *excess* loss about hardness?

Clearing that gate is necessary and not sufficient, and the difference cost 32 GPU
hours once. `report.py` measures the raw loss; the weight update never sees the raw
loss, it sees `max(0, L_proxy - L_ref)`. Those can be structured around completely
different variables. Once a reference and a proxy both exist, check the quantity that
actually drives the update -- 40 minutes of eval, no training:

```bash
# both checkpoints in one pass, so they see identical frame windows per scene
for pass in dl3dv scannet; do
    case $pass in
      dl3dv)   roots="--data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only" ;;
      scannet) roots="--data-root ~/scannet-train --depth-root ~/scannet-train/depth --no-dense-only" ;;
    esac
    python training/evaluate.py \
        --checkpoint runs/doremi-ref/final.pt runs/doremi-proxy/final.pt \
        --split train --scene-list doremi/tmp/pool.txt $roots \
        --repeats 3 --out runs/doremi-excess/$pass
done
for m in doremi-ref doremi-proxy; do
    cat runs/doremi-excess/{dl3dv,scannet}/windows-$m-final.jsonl \
        > runs/doremi-excess/$m.jsonl
done

python3 doremi/excess_report.py
```

It answers three questions in the order that decides whether to rerun: does the
cluster assignment explain the proxy-reference difference at all (eta^2 against the
same shuffle null); is the answer cluster *size* rather than cluster hardness; and is
the cross-domain spread bigger than the standard error of one domain's mean. Read the
size correlation first. `runs/doremi-proxy` cleared the eta^2 gate at 0.137 against a
null p95 of 0.026 -- and correlated **-0.92 with log cluster size** and **+0.09 with
cluster hardness**, because of the objective bug described under `--dro-objective`
below. A mixture learned from that is a mixture over cluster sizes.

**Match the `--clusters` to the checkpoint.** With the default (pruned) clusters json
this is clean, because the checkpoints above trained on exactly its scenes. Running
the check against `..._full.json` instead means 587 of the 2932 scenes were never
trained on, their loss is higher for that reason alone, and pruning removed them
unevenly across clusters -- so eta^2 comes out optimistic. To vet the unpruned pool,
use the `--mode reference` run from step 1 as the checkpoint, since it trains on all
of it.

## 1. Reference run

The model the excess loss is measured against. Trains on the baseline mixture -- the
natural one, every scene once per epoch -- restricted to the clustered pool.

```bash
python doremi/train.py --mode reference \
    --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only \
    --scannet-root ~/scannet-train \
    --preset small --dinov3 checkpoints/dinov3_vits16.pt \
    --num-frames 16 --batch-size 4 --grad-accum 8 --max-steps 30000 \
    --out runs/doremi-ref
```

Reference and proxy stay on this 4x8 recipe even though the final run below does
not: `--grad-accum` here is for the DRO update (32 scenes per weight step), not
for matching `runs/patch_avg`. They also stay on `small`, so the excess loss is
a headroom estimate between two models of the same capacity.

## 2. Proxy run

```bash
python doremi/train.py --mode proxy --reference runs/doremi-ref/final.pt \
    --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only \
    --scannet-root ~/scannet-train \
    --preset small --dinov3 checkpoints/dinov3_vits16.pt \
    --num-frames 16 --batch-size 4 --grad-accum 8 --max-steps 30000 \
    --min-domain-size 10 --dro-budget 4 --max-ratio 4 \
    --out runs/doremi-proxy
```

> **`runs/doremi-proxy` (30k steps, 32.7 h) is not usable.** It predates
> `--dro-objective` and ran under `domain-mean`, so its mixture (ratio 0.95-1.13,
> `ess` 44/48) tracks cluster size rather than hardness. Do not pass its
> `domain_weights.json` to `--mode final`. Its checkpoint is still a converged
> baseline-mixture model, so it is a fine thing to resume a short DRO-only phase
> from -- see step 0b and `--dro-start-step`.

Writes `runs/doremi-proxy/domain_weights.json` (the averaged mixture, which is what
DoReMi reports) and `domain_weights.jsonl` (the trajectory). Costs one extra forward
pass per step for the frozen reference; `--reference-device cpu` if it will not fit
alongside the proxy, or `--reference-losses <windows.jsonl>` to compare against cached
per-scene losses instead of a live model -- cheaper, and noisier because the cached
loss is for a different frame window than the proxy just saw.

**`--grad-accum` is doing double duty here.** One weight update happens per optimiser
step, over `batch_size * grad_accum * world_size` scenes spread across ~48 domains.
At 4x8 that is 32 scenes, so a domain that appears contributes ~1 scene and its excess
loss for that step is a single noisy sample; the EMA in `dro.py` is what makes that
usable, and more scenes per update is strictly better for the weights.

### Resuming a finished proxy for a fresh DRO phase

The mixture only means something once the proxy has converged, but `--dro-budget` is
spread over `--max-steps - --dro-start-step` updates, so a from-scratch run spends
most of its budget on steps where the excess loss is dominated by the proxy's own
convergence transient -- a common offset across domains, so no weight moves. A
finished proxy checkpoint is already a converged baseline-mixture model, so resume it
and spend the whole budget on the converged phase. Three flags make that safe:

* `--reset-dro`, or `alpha_bar`'s 30k-update running mean pins the mixture where it
  was (the trainer warns if you forget).
* `--constant-lr`, or the cosine is re-derived from the new `--max-steps` and step
  30000 of a 33000-step schedule gets ~4x the LR the 30000-step run ended at.
* `--dro-start-step` in *absolute* steps: resumed step + the EMA warmup you want.
  The 15-scene cluster appears in ~19% of 32-scene steps, so ~300 steps gives it
  ~55 observations against `--dro-ema 0.9`'s ~10-step window.

Then check the result before using it:

```bash
python3 doremi/report.py --weights runs/doremi-proxy/domain_weights.json \
    --history runs/doremi-proxy/domain_weights.jsonl
```

Three things to look for, all printed:

* **domains pinned at `--max-ratio`.** Their weight is being set by the cap, not by
  the data. Lower `--dro-budget` (or raise the cap and re-check `ess`).
* **`corr(log ratio, log cluster size)` strongly negative.** The weights are tracking
  domain *size*, which is the failure mode `dro.py`'s self-test reproduces: `max(0, .)`
  rectifies noise into positive mean excess, and a small domain has noisier estimates
  *and* too few scenes for the proxy to fit, so nothing pulls its weight back down.
  Raise `--min-domain-size`.
* **`ess` still falling at the end of the trajectory.** The min-max game had not
  equilibrated, so the mixture is partly a function of how long the proxy ran.

`runs/doremi-proxy-v2` is that resume: 3000 steps off `runs/doremi-proxy/final.pt`
at `--constant-lr 2e-5`, `--dro-start-step 30300`, `--dro-objective scene-ratio`,
3.3 h. It settles the objective question and leaves the size question open:

* the fix works as a fix. The mixture moves -- `alpha_bar / baseline` spans
  0.77-3.19, against 0.95-1.13 under `domain-mean` -- and it tracks its own signal,
  `corr(ratio, excess) = +0.79`.
* it still lands on size. `corr(excess, log cluster size) = -0.75` at the first DRO
  log and -0.75 at the last, so the pruned pool's size structure is in the excess
  loss with the objective bias gone. Top weights are the three smallest clusters
  again (47, 45, 44).
* 4x oversampling those clusters for 1400 steps did not move their excess at all
  (0.108 mean for n<=30, flat). Either 1400 steps at 2e-5 is nowhere near enough to
  undo 30k biased steps, or the correlation has a cause outside the objective.
* `--max-ratio 4` bound at DRO step 1300 and held to the end, so the tail of the
  mixture is the cap, not the data. Hence `--dro-budget 2` in the short recipe below.

The one cheap thing it does establish: the mixture is readable in ~1300 DRO steps.
Nothing needs 30k steps of DRO.

## Short recipe

`doremi/run_raw_clusters.sh` runs the pair for ~8 h instead of 56 h, over the
unpruned `..._full.json` clusters (48 domains / 2928 scenes at `--min-size 10`,
sizes 15-141). Four changes, none of which touch what the mixture *means*:

| change | why it is safe | saving |
| --- | --- | --- |
| `--max-steps 10000` | 30k x 32 scenes is 410 epochs of a 2.3k-scene pool. The mixture needs the ref-proxy *gap*, not either model's convergence, and both run the same schedule. | 3x |
| `--num-frames 8` | changes the covisibility windows for both models equally. The assumption is that the mixture transfers to a 16-frame final run, which is the same assumption that lets a `small` proxy speak for a `base` run. | ~2x |
| `--reference-losses` off one eval pass | the reference forward is 28% of proxy step time (3.92 vs 2.83 s/step); the pass costs 6 min per checkpoint at ~19 windows/s. Noisier: the cached loss is a different frame window. | 1.4x on the proxy |
| `--dro-start-step 7000` | v2 saturated in 1300 DRO updates. | -- |

```bash
bash doremi/run_raw_clusters.sh dryrun    # seconds: pool, domain table, cache coverage
bash doremi/run_raw_clusters.sh ref       # ~4 h
bash doremi/run_raw_clusters.sh cache     # ~15 min, and it is also the step-0 gate
bash doremi/run_raw_clusters.sh gate      # eta^2 vs the shuffle null -- read this
bash doremi/run_raw_clusters.sh proxy     # ~4 h
bash doremi/run_raw_clusters.sh report    # the three checks above
bash doremi/run_raw_clusters.sh excess    # ~15 min: the size correlation, out of sample
```

`FRAMES=16 STEPS=30000 WARMUP=2000 bash ...` is the long recipe, and
`MIN_SIZE=20 bash ...` is the lever to pull if `report` still shows the weights
tracking cluster size (46 domains / 2894 scenes).

Two things the short recipe changes that are worth watching. `--warmup-steps` scales
with the run -- 800, not 2000, or a fifth of training is warmup. And the reference
here is trained on the *superset*: every scene of the unpruned pool, so the same
`runs/doremi-ref-raw` is a valid reference for any subset of these clusters, pruned
or not, and step 0's "match the `--clusters` to the checkpoint" rule is satisfied for
both pools by one run.

The `cache` stage does double duty on purpose: the per-scene reference losses the
proxy reads are exactly the jsonl step 0's gate wants, so the gate costs nothing
extra and lands before the proxy's 4 h rather than after.

## 3. Final run

The flags here match `runs/patch_avg` / `runs/sota` (`small`, batch 4, accum 1,
lr 1e-4, warmup 3k, 15k steps), so a gain is not "because we trained a `base`
model for 100k steps with a 32-scene batch". `--grad-accum 8` stays on the
reference and proxy only.

```bash
python doremi/train.py --mode final \
    --domain-weights runs/doremi-proxy/domain_weights.json \
    --data-root ~/dl3dv-train --depth-root ~/dl3dv-depth --dense-only \
    --scannet-root ~/scannet-train \
    --preset small --dinov3 checkpoints/dinov3_vits16.pt \
    --num-frames 16 --batch-size 4 --grad-accum 1 \
    --lr 1e-4 --warmup-steps 3000 --max-steps 15000 \
    --out runs/doremi-final
```

`runs/80pct` used 24k steps and `runs/80pct_origsteps` used 30k; swap
`--max-steps` if that is the column you want. Defaults in `training/train.py`
are lr 2e-4 / warmup 2k / 100k steps -- passing them is load-bearing.

The epoch is drawn with replacement from per-scene weights `alpha_j / |C_j|`, so
"epoch" stops meaning "every scene once" -- the trainer counts steps only. The
comparison run is the same command with `--domain-weights` swapped for a baseline
weights file, *not* `training/train.py` and not the existing `runs/patch_avg*`
checkpoints: sampling with replacement is itself a change, and it should not be
inside the measurement. Those checkpoints are a sanity check that the baseline
arm lands nearby. Write a baseline file with

```bash
python3 -c "import sys; sys.path.insert(0,'.'); from pathlib import Path; \
from doremi.domains import load_domains, save_weights; d = load_domains(min_size=10); \
save_weights(Path('runs/doremi-final/baseline_weights.json'), d, d.baseline())"
```

`--val-every` runs the DL3DV/ScanNet val splits during the run, as in
`training/train.py`. The ETH3D numbers that decide whether the mixture was worth
anything come afterwards, the same way every other run in this repo gets them:

```bash
python training/evaluate.py --checkpoint runs/doremi-final/final.pt \
    --data-root ~/eth3d-eval --depth-root ~/eth3d-eval/depth \
    --repeats 3 --out runs/doremi-final/eval-eth3d
```

This trainer has no in-loop ETH3D probe -- that lives in `train_with_eval/train.py`,
which extends `training/train.py`'s parser the same way this one does, so the two sets
of flags could be merged if the probe turns out to be wanted mid-run.

## What is adapted from DoReMi, and why

Each of these is argued where it is implemented; the short version:

* **The baseline mixture is proportional to cluster size, not uniform** (`domains.py`).
  A plain epoch lists every scene once, so that is the mixture the reference is
  trained on and the only meaningful denominator for `ratio`.
* **Smoothing pulls toward that baseline** rather than toward uniform. Uniform over 48
  domains would put a floor of ~23x oversampling under a 15-scene cluster.
* **Domains absent from a step keep their last excess-loss estimate** (`dro.py`).
  DoReMi's every minibatch contains every domain; ~32 scenes over 48 domains does not,
  and treating "not sampled" as "no excess loss" would decay a domain's weight on no
  evidence.
* **`eta` is chosen from a ratio budget, not fixed at 1.0** (`dro.auto_eta`). `eta`
  multiplies an excess loss, and DoReMi's 1.0 is calibrated to per-token
  cross-entropy; this objective's excess runs 1-6, which saturated the mixture within
  4 steps when measured. `--dro-budget` says instead how far a persistently hard
  domain may move over the whole run.
* **`--max-ratio` caps `alpha / baseline`** both ways. What stops the exponentiated
  gradient from compounding is the proxy fitting the domains it oversamples; with
  ~50 scenes per domain that feedback is weak, so there is a backstop.
* **Per-scene, not per-token, normalisation** (`per_sample_loss.py`). The analogue of
  "per token" here is "per valid pixel", which is what `criterion(batch)` already
  does; the per-domain mean of per-scene losses weights scenes equally instead. They
  differ whenever depth coverage differs across scenes, which for DL3DV's sparse
  COLMAP depth is often. `--per-sample-objective` puts the reference and final runs on
  the proxy's normalisation if you want the three to match exactly.
* **The objective is a per-scene reweighting, not a weighted mean of per-domain
  means** (`--dro-objective scene-ratio`, the default). DoReMi's
  `sum_j alpha_j * mean_j(losses)` needs every minibatch to contain every domain in
  proportion to its share. A step here holds ~32 scenes over 48 domains, so a domain
  that appears at all almost always appears exactly once, and scene i ends up
  carrying `alpha_j / (count_j * mass)` -- a weight **proportional to its cluster's
  size**, 3.2x from the 15-scene cluster to the 65-scene one. `runs/doremi-proxy` ran
  that way: it undertrained its smallest clusters by ~3x, scored worse on them than
  the reference for that reason alone, and the reweighter read the gap it had created
  as excess loss and upweighted exactly those clusters. Weighting each scene by
  `alpha_j / baseline_j` instead is flat in cluster size and, at the baseline
  mixture, is *exactly* the reference run's objective -- which is the premise the
  excess loss rests on. `--dro-objective domain-mean` reproduces the old behaviour;
  `python3 doremi/per_sample_loss.py` measures both.

## Known limits

* **The domains are small.** ~48 domains over ~2340 scenes, median 46 scenes each.
  Upweighting a domain means revisiting the same ~46 scenes more often, which past a
  modest ratio trades coverage for repetition. This is the main reason to be
  suspicious of a large `ratio`, and the reason `--max-ratio` defaults to 4.
* **Cluster and dataset are nearly the same variable.** 42 of the 50 clusters contain
  scenes from only one of DL3DV and ScanNet, and 48 are >=95% one of them. The two
  differ in depth-GT quality, so a cluster can look hard for a reason no amount of
  resampling fixes. Subtracting a reference model's loss is exactly the correction for
  that, which is the strongest argument for doing this properly rather than
  reweighting on raw per-cluster loss -- and
  `report.py`'s `corr(log ratio, DL3DV share)` is the check that it worked.
* **The clusters json is already pruned.** The default one keeps 2345 of 2930 scenes by
  complexity-based density pruning, which has already thinned some of the clusters
  DoReMi may want to upweight. For the cleaner experiment, re-cluster without
  `--target`/`--target-pct` and pass the result via `--clusters`.
* **The proxy objective and the reference objective are not identical** unless the
  reference run is given `--per-sample-objective`; see above.
