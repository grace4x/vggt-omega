#!/usr/bin/env bash
# Score one checkpoint on the fixed eval suite, writing <out-dir>/eval-<set>/.
#
#   bash grad_cluster_train/eval_suite.sh runs/uniform-s0/avg.pt runs/uniform-s0
#
# Fixed protocol so every run is scored on identical windows: 8 frames, eval
# seed 1234, sampling/window from the checkpoint. ETH3D keeps the 256x256 crop
# the earlier runs were scored with; the 224x384 sets are scored at full frame.
# Every set gets 3 windows per scene (the whole suite takes about a minute).
set -euo pipefail

ckpt=$1
out=$2

ev() {
    local name=$1
    shift
    echo "== $name"
    python training/evaluate.py --checkpoint "$ckpt" --num-frames 8 --seed 1234 --progress 0 \
        --out "$out/eval-$name" "$@"
}

ev eth3d     --data-root ~/eth3d-eval     --depth-root ~/eth3d-eval/depth     --split all --repeats 3 --eval-crop 256 256
ev dl3dv     --data-root ~/dl3dv-eval     --depth-root ~/dl3dv-eval-depth     --split all --repeats 3
ev scannet   --data-root ~/scannet-train   --depth-root ~/scannet-train/depth   --split val --repeats 3
ev hypersim  --data-root ~/hypersim-train  --depth-root ~/hypersim-train/depth  --split val --repeats 3
ev tartanair --data-root ~/tartanair-train --depth-root ~/tartanair-train/depth --split val --repeats 3
