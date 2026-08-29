#!/usr/bin/env bash
# Stream the ETH3D high-res multi-view training set into a VGGT-Omega eval set:
# for each scene, download its two archives, extract, preprocess, then delete the
# raw files before moving on.
#
# Why streaming. The download is only ~10 GB, but the ground truth depth maps are
# uncompressed float32 dumps at 6048x4032 -- 97 MB *per frame* -- so `facade`'s 76
# frames expand to ~7.4 GB on disk and the whole set to ~50 GB, all of it pure
# intermediate. Preprocessing per scene and deleting as it goes keeps peak disk at
# (jobs x ~9 GB) of staging plus ~100 MB of output.
#
# Resumable and idempotent: a scene with a marker under <out>/.done is skipped, so
# re-running after an interruption picks up where it left off. Scenes that failed
# with a transient error get no marker and are retried; scenes the preprocessor
# legitimately rejected are marked so they are not downloaded again.
#
# Usage:
#   training/download_eth3d.sh                      # all 13 scenes
#   training/download_eth3d.sh --limit 1            # one scene, to try it out
#   training/download_eth3d.sh --jobs 3             # more parallelism
#   training/download_eth3d.sh --scenes "pipes office"
#
# Sizing. Every ETH3D DSLR frame is 6048x4032, so --resolution 384 stores the whole
# set at one shape, 256x384. To make it stackable in the same batch as a DL3DV set
# built at --resolution 384 (stored 224x384) -- only needed if you mix, not to
# evaluate -- force the shape:
#   training/download_eth3d.sh --target-hw 224 384 --fit crop
#
# This fetches the *_dslr_jpg (distorted) archives rather than *_dslr_undistorted,
# because ETH3D's ground truth depth maps are registered to the distorted images
# and the model that undoes that distortion ships only with the distorted
# calibration. See the preprocess_eth3d.py docstring.
#
# By running this you are accepting ETH3D's terms: the data is free for research
# use with attribution -- see https://www.eth3d.net/

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE="${ETH3D_STAGE:-$HOME/eth3d-raw}"
OUT="${ETH3D_OUT:-$HOME/eth3d-eval}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
BASE=https://www.eth3d.net/data
JOBS=2
RESOLUTION=384
TARGET_HW=""
FIT=crop
SUPERSAMPLE=8
MIN_FRAMES=8
MIN_DEPTH_FRAC=0.05
LIMIT=0
RETRIES=3
DL_TIMEOUT=3600
SCENES=""
KEEP_RAW=0
CHECK=1

# The 13 scenes of the high-res multi-view *training* split. The test split ships
# no laser scan and no depth, so there is nothing to score against offline.
ALL_SCENES="courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains"

usage() { sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---- helpers --------------------------------------------------------------- #

# Content-Length is the only way to tell a finished download from a truncated one,
# and it matters more than usual here: the depth files carry no header and no
# checksum, so `preprocess_eth3d.py` would read a truncated archive's short frame
# as a size mismatch at best and silently-wrong geometry at worst.
fetch_verified() {
  local url="$1" dest="$2" name="$3" expected="" got=""
  for attempt in $(seq 1 "$RETRIES"); do
    expected=$(curl -sIL --max-time 60 "$url" | tr -d '\r' | grep -i '^content-length' | tail -1 | awk '{print $2}')
    [[ -n "$expected" && "$expected" -gt 0 ]] 2>/dev/null && break
    sleep $((attempt * 5))
  done
  [[ -z "$expected" ]] && { echo "[FAIL ] $name: could not read Content-Length for $url"; return 1; }

  for attempt in $(seq 1 "$RETRIES"); do
    got=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    [[ "$got" == "$expected" ]] && return 0
    # `-C -` resumes from whatever is on disk, so a dropped transfer costs only the
    # remaining bytes rather than starting the 700 MB archive over.
    timeout "$DL_TIMEOUT" curl -sL -C - --retry 5 --retry-delay 5 --retry-all-errors \
        --speed-limit 10240 --speed-time 120 -o "$dest" "$url" >/dev/null 2>&1
    got=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    [[ "$got" == "$expected" ]] && return 0
    echo "[retry] $name attempt $attempt/$RETRIES: $got/$expected bytes"
    sleep $((attempt * 10))
  done
  echo "[FAIL ] $name: download failed after $RETRIES attempts"
  return 1
}

# ---- internal per-scene worker (re-entry point for xargs) ------------------ #
if [[ "${1:-}" == "--worker" ]]; then
  scene="$2"; shift 2
  STAGE="$ETH3D_W_STAGE"; OUT="$ETH3D_W_OUT"; PYTHON="$ETH3D_W_PYTHON"; REPO_DIR="$ETH3D_W_REPO"
  BASE="$ETH3D_W_BASE"; RESOLUTION="$ETH3D_W_RES"; TARGET_HW="$ETH3D_W_TARGETHW"; FIT="$ETH3D_W_FIT"
  SUPERSAMPLE="$ETH3D_W_SS"; MIN_FRAMES="$ETH3D_W_MINF"; MIN_DEPTH_FRAC="$ETH3D_W_MINDF"
  RETRIES="$ETH3D_W_RETRIES"; DL_TIMEOUT="$ETH3D_W_TIMEOUT"; KEEP_RAW="$ETH3D_W_KEEP"; CHECK="$ETH3D_W_CHECK"

  marker="$OUT/.done/$scene"
  [[ -f "$marker" ]] && { echo "[skip ] $scene (already done)"; exit 0; }

  mkdir -p "$STAGE"
  for kind in jpg depth; do
    fetch_verified "$BASE/${scene}_dslr_${kind}.7z" "$STAGE/${scene}_dslr_${kind}.7z" "$scene/$kind" || exit 0
  done

  # py7zr rather than 7z: p7zip is not in the base image and installing it needs
  # root, while py7zr is a pip dependency of this repo's venv. Extracting into the
  # stage root is what puts both archives' <scene>/ trees on top of each other,
  # which is the layout preprocess_eth3d.py expects.
  if ! "$PYTHON" - "$STAGE" "$scene" <<'PY' 2>&1
import sys, py7zr
from pathlib import Path
stage, scene = Path(sys.argv[1]), sys.argv[2]
for kind in ("jpg", "depth"):
    with py7zr.SevenZipFile(stage / f"{scene}_dslr_{kind}.7z") as z:
        z.extractall(stage)
PY
  then echo "[ERROR] $scene: extraction failed"; [[ "$KEEP_RAW" == 1 ]] || rm -rf "$STAGE/$scene" "$STAGE/${scene}"_dslr_*.7z; exit 0; fi

  raw_size=$(du -sh "$STAGE/$scene" 2>/dev/null | cut -f1)
  # Deliberately unquoted: --target-hw takes two words, and is empty when unset.
  # shellcheck disable=SC2086
  result=$("$PYTHON" "$REPO_DIR/training/preprocess_eth3d.py" \
      --raw-root "$STAGE" --out "$OUT" --scenes "$scene" \
      --resolution "$RESOLUTION" ${TARGET_HW:+--target-hw $TARGET_HW} --fit "$FIT" \
      --supersample "$SUPERSAMPLE" --min-frames "$MIN_FRAMES" \
      --min-depth-frac "$MIN_DEPTH_FRAC" ${CHECK:+--check} \
      --workers 1 --quiet --no-index 2>&1 | tail -3)

  status=$(echo "$result" | awk -v s="$scene" '$1==s {print $2}' | tail -1)
  case "$status" in
    ok|cached|skip) mkdir -p "$OUT/.done" && touch "$marker" ;;
    *) echo "[ERROR] $scene: ${result:-no output from preprocessor}" ;;
  esac
  # Both the archives and the extracted tree are pure intermediate; nothing reads
  # them again once meta.npz and the depth PNGs exist.
  if [[ "$KEEP_RAW" != 1 ]]; then rm -rf "$STAGE/$scene" "$STAGE/${scene}"_dslr_*.7z; fi
  echo "[$(printf '%-5s' "${status:-err}")] $scene (raw $raw_size) $(echo "$result" | tail -1 | cut -d' ' -f3-)"
  exit 0
fi

# ---- argument parsing ------------------------------------------------------ #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)          STAGE="$2"; shift 2 ;;
    --out)            OUT="$2"; shift 2 ;;
    --python)         PYTHON="$2"; shift 2 ;;
    --jobs|-j)        JOBS="$2"; shift 2 ;;
    --resolution)     RESOLUTION="$2"; shift 2 ;;
    --target-hw)      TARGET_HW="$2 $3"; shift 3 ;;
    --fit)            FIT="$2"; shift 2 ;;
    --supersample)    SUPERSAMPLE="$2"; shift 2 ;;
    --min-frames)     MIN_FRAMES="$2"; shift 2 ;;
    --min-depth-frac) MIN_DEPTH_FRAC="$2"; shift 2 ;;
    --limit)          LIMIT="$2"; shift 2 ;;
    --retries)        RETRIES="$2"; shift 2 ;;
    --timeout)        DL_TIMEOUT="$2"; shift 2 ;;
    --scenes)         SCENES="$2"; shift 2 ;;
    --keep-raw)       KEEP_RAW=1; shift ;;
    --no-check)       CHECK=""; shift ;;
    -h|--help)        usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ -x "$PYTHON" ]] || { echo "python not found at $PYTHON (pass --python)" >&2; exit 1; }
"$PYTHON" -c "import py7zr" 2>/dev/null || {
  echo "py7zr is needed to unpack ETH3D's .7z archives. Install it with:" >&2
  echo "    $PYTHON -m pip install py7zr" >&2
  exit 1
}

mkdir -p "$STAGE" "$OUT/.done"

todo="$OUT/.todo"
printf '%s\n' ${SCENES:-$ALL_SCENES} > "$todo"
[[ "$LIMIT" -gt 0 ]] && head -n "$LIMIT" "$todo" > "$todo.n" && mv "$todo.n" "$todo"

total=$(wc -l < "$todo")
done_already=$(find "$OUT/.done" -type f 2>/dev/null | wc -l)
echo "ETH3D high-res multi-view (training split) -> $OUT"
echo "  $total scenes queued, $done_already already done, $JOBS parallel jobs"
echo "  ~10 GB to download; staging in $STAGE (deleted per scene, peak ~$((JOBS * 9)) GB)"
echo "  accepting the ETH3D terms of use: https://www.eth3d.net/"
echo

export ETH3D_W_STAGE="$STAGE" ETH3D_W_OUT="$OUT" ETH3D_W_PYTHON="$PYTHON" ETH3D_W_REPO="$REPO_DIR"
export ETH3D_W_BASE="$BASE" ETH3D_W_RES="$RESOLUTION" ETH3D_W_TARGETHW="$TARGET_HW" ETH3D_W_FIT="$FIT"
export ETH3D_W_SS="$SUPERSAMPLE" ETH3D_W_MINF="$MIN_FRAMES" ETH3D_W_MINDF="$MIN_DEPTH_FRAC"
export ETH3D_W_RETRIES="$RETRIES" ETH3D_W_TIMEOUT="$DL_TIMEOUT" ETH3D_W_KEEP="$KEEP_RAW"
export ETH3D_W_CHECK="$CHECK"

started=$(date +%s)
# Workers write only per-scene files, never index.json -- see --no-index in
# preprocess_eth3d.py for why the index is rebuilt afterwards instead.
xargs -a "$todo" -P "$JOBS" -I{} "${BASH_SOURCE[0]}" --worker {}

echo
echo "rebuilding index..."
# shellcheck disable=SC2086
"$PYTHON" "$REPO_DIR/training/preprocess_eth3d.py" \
    --raw-root "$STAGE" --out "$OUT" --index-only \
    --resolution "$RESOLUTION" ${TARGET_HW:+--target-hw $TARGET_HW} --fit "$FIT" \
    --supersample "$SUPERSAMPLE"

elapsed=$(( $(date +%s) - started ))
echo "done in $((elapsed / 60))m. output $(du -sh "$OUT" 2>/dev/null | cut -f1) in $OUT"
echo
echo "evaluate with:"
echo "  $PYTHON training/evaluate.py --checkpoint <run>/latest.pt \\"
echo "      --data-root $OUT --depth-root $OUT/depth --split all"
