#!/usr/bin/env bash
# DoReMi over the *unpruned* clusters, on the short recipe: ~8 h for the pair
# instead of the 56 h `runs/doremi-ref` + `runs/doremi-proxy` cost.
#
# Four things buy that back (see doremi/README.md "Short recipe"):
#   * 10k steps instead of 30k -- 137 epochs of a 2928-scene pool, not 410.
#   * 8 frames instead of 16.
#   * the proxy reads cached per-scene reference losses instead of running the
#     reference model every step (-28% per step).
#   * DRO runs only over the last 3k steps, which is where v2's mixture moved.
#
# Stages are separate on purpose: `gate` decides whether `proxy` is worth running.
#
#   bash doremi/run_raw_clusters.sh dryrun    # seconds, no training
#   bash doremi/run_raw_clusters.sh ref       # ~4 h
#   bash doremi/run_raw_clusters.sh cache     # ~15 min (two dataset passes)
#   bash doremi/run_raw_clusters.sh gate      # seconds -- read before `proxy`
#   bash doremi/run_raw_clusters.sh proxy     # ~4 h
#   bash doremi/run_raw_clusters.sh report    # seconds
#   bash doremi/run_raw_clusters.sh excess    # ~15 min + seconds
#
# FRAMES=16 STEPS=30000 bash ... reproduces the long recipe.
#
# Capacity-deficit rerun (`runs/doremi-proxy-tiny`)
# -------------------------------------------------
# `runs/doremi-proxy-raw` ran the proxy at the reference's own preset, and its
# per-cluster excess loss came out inside the shuffle null (eta^2 0.017 vs p95
# 0.023): two identically-sized models on the same pool differ by a near-constant
# offset plus window noise, not by domain. The clamp then read that noise as
# signal -- the clamped domain excess was 90% predicted by the *variance* of the
# per-window difference (corr +0.95), which tracks the loss level and hence the
# dataset, so the mixture correlated -0.77 with DL3DV share and +0.08 with the
# signed excess. Both halves are fixed below: a proxy with 2.6x less non-trunk
# capacity, and no rectification.
#
# It reuses `runs/doremi-ref-raw` and its cached losses, so only the proxy and the
# paired eval need to run -- ~4 h, not ~8 h:
#
#   PROXY=runs/doremi-proxy-tiny EVAL=runs/doremi-excess-tiny \
#   PROXY_PRESET=tiny EXCESS_CLAMP=none \
#       bash doremi/run_raw_clusters.sh dryrun proxy report excess
#
# Read `excess`'s section 1 first: eta^2 above the null is the whole point of the
# rerun, and a mixture learned when it is inside the null is not worth reading.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTERS=patch_avg_clustering/clusters_dl3dv_scannet_patchavg_k50_none_full.json
POOL=doremi/tmp/pool_full.txt
MIN_SIZE=${MIN_SIZE:-10}
FRAMES=${FRAMES:-8}
STEPS=${STEPS:-10000}
WARMUP=${WARMUP:-800}
DRO_START=${DRO_START:-7000}
BUDGET=${BUDGET:-2}

REF=${REF:-runs/doremi-ref-raw}
PROXY=${PROXY:-runs/doremi-proxy-raw}
# Two separate eval dirs on purpose. REF_EVAL holds the reference-only pass whose
# jsonl becomes the proxy's loss cache, and belongs to the *reference*, so a rerun
# that swaps only the proxy must not move it -- otherwise CACHE points at a file
# nothing wrote. EVAL holds that proxy's own paired pass ("$EVAL-paired").
REF_EVAL=${REF_EVAL:-runs/doremi-excess-raw}
EVAL=${EVAL:-runs/doremi-excess-raw}
CACHE=${CACHE:-$REF_EVAL/$(basename "$REF").jsonl}

# The reference is always `small`. The proxy's preset and excess estimator are the
# two knobs the `tiny` rerun turns; see "Capacity-deficit rerun" above.
REF_PRESET=small
PROXY_PRESET=${PROXY_PRESET:-small}
EXCESS_CLAMP=${EXCESS_CLAMP:-scene}

COMMON=(
    --clusters "$CLUSTERS" --min-domain-size "$MIN_SIZE"
    --data-root "$HOME/dl3dv-train" --depth-root "$HOME/dl3dv-depth" --dense-only
    --scannet-root "$HOME/scannet-train"
    --dinov3 checkpoints/dinov3_vits16.pt
    --num-frames "$FRAMES" --batch-size 4 --grad-accum 8
    --lr 2e-4 --warmup-steps "$WARMUP" --max-steps "$STEPS"
    --val-every 1000 --save-every 1000 --log-every 50 --workers 8
)
DRO=(
    --dro-start-step "$DRO_START" --dro-budget "$BUDGET" --max-ratio 4
    --dro-objective scene-ratio --dro-log-every 100
    --excess-clamp "$EXCESS_CLAMP"
)

# `evaluate.py` names its jsonl after the run directory, not the checkpoint file.
windows() { echo "windows-$(basename "$1")-final.jsonl"; }

# One eval pass per dataset root, over the pool the trainer will use.
eval_pool() {  # eval_pool <out-dir> <checkpoint>...
    local out=$1; shift
    python training/evaluate.py --checkpoint "$@" \
        --split train --scene-list "$POOL" --num-frames "$FRAMES" \
        --data-root "$HOME/dl3dv-train" --depth-root "$HOME/dl3dv-depth" --dense-only \
        --repeats 3 --out "$out/dl3dv"
    python training/evaluate.py --checkpoint "$@" \
        --split train --scene-list "$POOL" --num-frames "$FRAMES" \
        --data-root "$HOME/scannet-train" --depth-root "$HOME/scannet-train/depth" --no-dense-only \
        --repeats 3 --out "$out/scannet"
}

stage_dryrun() {
    python3 doremi/domains.py --clusters "$CLUSTERS" --min-size "$MIN_SIZE" --scene-list "$POOL"
    python doremi/train.py --mode reference "${COMMON[@]}" --preset "$REF_PRESET" \
        --out "$REF" --dry-run
    if [[ -f $CACHE ]]; then
        python doremi/train.py --mode proxy --reference-losses "$CACHE" \
            "${COMMON[@]}" "${DRO[@]}" --preset "$PROXY_PRESET" --out "$PROXY" --dry-run
    else
        echo "[dryrun] skipping the proxy dry-run: $CACHE does not exist yet (stage 'cache')"
    fi
}

stage_ref() {
    python doremi/train.py --mode reference "${COMMON[@]}" --preset "$REF_PRESET" --out "$REF"
}

stage_cache() {
    python3 doremi/domains.py --clusters "$CLUSTERS" --min-size "$MIN_SIZE" --scene-list "$POOL"
    eval_pool "$REF_EVAL" "$REF/final.pt"
    cat "$REF_EVAL"/{dl3dv,scannet}/"$(windows $REF)" > "$CACHE"
    wc -l "$CACHE"
}

# Does the cluster assignment explain per-scene loss on *this* pool at all? The
# reference is the only checkpoint trained on all of it, so it is the only honest
# one to ask. If eta^2 sits inside the shuffle null, skip the proxy.
stage_gate() {
    python3 doremi/report.py --clusters "$CLUSTERS" --min-domain-size "$MIN_SIZE" \
        --eval-jsonl "$CACHE"
}

stage_proxy() {
    python doremi/train.py --mode proxy --reference-losses "$CACHE" \
        "${COMMON[@]}" "${DRO[@]}" --preset "$PROXY_PRESET" --out "$PROXY"
}

stage_report() {
    python3 doremi/report.py --clusters "$CLUSTERS" --min-domain-size "$MIN_SIZE" \
        --weights "$PROXY/domain_weights.json" --history "$PROXY/domain_weights.jsonl"
}

# The mixture's own signal, out of sample: both checkpoints in one pass so they
# see identical frame windows per scene.
stage_excess() {
    eval_pool "$EVAL-paired" "$REF/final.pt" "$PROXY/final.pt"
    for run in $REF $PROXY; do
        cat "$EVAL-paired"/{dl3dv,scannet}/"$(windows $run)" > "$EVAL-paired/$(basename $run).jsonl"
    done
    python3 doremi/excess_report.py --clusters "$CLUSTERS" --min-domain-size "$MIN_SIZE" \
        --reference-jsonl "$EVAL-paired/$(basename $REF).jsonl" \
        --proxy-jsonl "$EVAL-paired/$(basename $PROXY).jsonl"
}

for stage in "${@:-dryrun}"; do
    echo "=== $stage ==="
    "stage_$stage"
done
echo "=== done ==="
