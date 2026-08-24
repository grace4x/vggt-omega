#!/usr/bin/env bash
# Stream ScanNet v2 into a VGGT-Omega training set: for each scan, download its
# .sens, preprocess it, then delete the .sens before moving on.
#
# Why streaming. The .sens files are the only ones worth downloading (they carry
# RGB, metric depth, poses and intrinsics all together) but they total ~820 GB
# across the 1513 scans, which will not fit alongside an existing DL3DV set. They
# are also pure intermediate: once preprocess_scannet.py has read one, nothing
# ever needs it again. So peak disk here is (jobs x ~1 GB) of staging plus the
# ~20 GB of output, instead of 820 GB.
#
# Resumable and idempotent: a scan with a marker under <out>/.done is skipped, so
# re-running after an interruption picks up where it left off. Scans that failed
# with a transient error get no marker and are retried on the next run; scans
# legitimately rejected by the preprocessor (too few valid poses, etc.) are
# marked so they are not downloaded again.
#
# Usage:
#   training/download_scannet.sh                       # all 1513 scans
#   training/download_scannet.sh --limit 20            # first 20, to try it out
#   training/download_scannet.sh --jobs 6              # more parallelism
#   training/download_scannet.sh --scenes-file my.txt  # a specific list
#
# Sizing. By default frames are stored at --resolution 384, which for ScanNet's
# 4:3 source means 288x384. To make ScanNet stackable in the same batch as a
# DL3DV set built at --resolution 384 (stored 224x384), force the shape:
#   training/download_scannet.sh --target-hw 224 384 --fit crop
# `--fit crop` costs ~22% of the vertical field of view; `--fit squash` keeps it
# all but stretches the frames. See plan_output() in preprocess_scannet.py.
#
# By running this you are accepting the ScanNet terms of use:
#   http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOWNLOADER="${SCANNET_DOWNLOADER:-$HOME/download-scannet.py}"
STAGE="${SCANNET_STAGE:-$HOME/scannet-sens}"
OUT="${SCANNET_OUT:-$HOME/scannet-train}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
JOBS=3
RESOLUTION=384
TARGET_HW=""
FIT=crop
MAX_FRAMES=150
MIN_FRAMES=24
MIN_SHARPNESS=0
LIMIT=0
RETRIES=3
DL_TIMEOUT=2400
SCENES_FILE=""
USE_DOWNLOADER=0
BASE=http://kaldir.vc.cit.tum.de/scannet
KEEP_SENS=0

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---- internal per-scan worker (re-entry point for xargs) ------------------- #
if [[ "${1:-}" == "--worker" ]]; then
  scene="$2"; shift 2
  STAGE="$SCANNET_W_STAGE"; OUT="$SCANNET_W_OUT"; PYTHON="$SCANNET_W_PYTHON"
  DOWNLOADER="$SCANNET_W_DOWNLOADER"; REPO_DIR="$SCANNET_W_REPO"
  RESOLUTION="$SCANNET_W_RES"; MAX_FRAMES="$SCANNET_W_MAXF"; MIN_FRAMES="$SCANNET_W_MINF"
  MIN_SHARPNESS="$SCANNET_W_SHARP"; RETRIES="$SCANNET_W_RETRIES"
  TARGET_HW="$SCANNET_W_TARGETHW"; FIT="$SCANNET_W_FIT"
  USE_DOWNLOADER="$SCANNET_W_USEDL"; BASE_URL="$SCANNET_W_BASE"
  DL_TIMEOUT="$SCANNET_W_TIMEOUT"; KEEP_SENS="$SCANNET_W_KEEP"

  marker="$OUT/.done/$scene"
  [[ -f "$marker" ]] && { echo "[skip ] $scene (already done)"; exit 0; }

  sens="$STAGE/scans/$scene/$scene.sens"
  mkdir -p "$(dirname "$sens")"
  url="$BASE_URL/v1/scans/$scene/$scene.sens"   # v2 reuses v1's .sens; v2/ 404s

  # Content-Length is the only way to tell a finished download from a truncated
  # one, and truncation is common here (the server drops ~500 MB transfers
  # midway). It matters more than usual because sens_reader tolerates a short
  # file by returning fewer frames -- so an unverified partial download would
  # quietly become a partial scene rather than an error.
  expected=""
  for attempt in $(seq 1 "$RETRIES"); do
    expected=$(curl -sIL --max-time 60 "$url" | tr -d '\r' | grep -i '^content-length' | tail -1 | awk '{print $2}')
    [[ -n "$expected" && "$expected" -gt 0 ]] 2>/dev/null && break
    sleep $((attempt * 5))
  done
  if [[ -z "$expected" ]]; then echo "[FAIL ] $scene: could not read Content-Length"; exit 0; fi

  ok=0
  for attempt in $(seq 1 "$RETRIES"); do
    got=$(stat -c%s "$sens" 2>/dev/null || echo 0)
    if [[ "$got" == "$expected" ]]; then ok=1; break; fi

    if [[ "$USE_DOWNLOADER" == 1 ]]; then
      # download-scannet.py prompts twice on this path: once for the TOS, once to
      # confirm .sens. Two blank lines accepts both. It cannot resume -- it uses
      # urlretrieve and starts from byte 0 -- so a truncated transfer throws away
      # everything and the whole scan is re-fetched. That is why --use-downloader
      # is opt-in rather than the default.
      rm -f "$sens" "$STAGE/scans/$scene"/tmp* 2>/dev/null
      printf '\n\n' | timeout "$DL_TIMEOUT" "$PYTHON" "$DOWNLOADER" \
          -o "$STAGE" --id "$scene" --type .sens >/dev/null 2>&1
      rm -f "$STAGE/scans/$scene"/tmp* 2>/dev/null   # urlretrieve leaves these behind on failure
    else
      # `-C -` resumes from whatever is already on disk, so a dropped transfer
      # costs only the remaining bytes instead of the whole file.
      timeout "$DL_TIMEOUT" curl -sL -C - --retry 5 --retry-delay 5 --retry-all-errors \
          --speed-limit 10240 --speed-time 120 -o "$sens" "$url" >/dev/null 2>&1
    fi

    got=$(stat -c%s "$sens" 2>/dev/null || echo 0)
    if [[ "$got" == "$expected" ]]; then ok=1; break; fi
    echo "[retry] $scene attempt $attempt/$RETRIES: $got/$expected bytes"
    sleep $((attempt * 10))
  done
  if [[ "$ok" != 1 ]]; then echo "[FAIL ] $scene: download failed after $RETRIES attempts"; exit 0; fi

  size=$(du -h "$sens" | cut -f1)
  # Deliberately unquoted: --target-hw takes two words, and is empty when unset.
  # shellcheck disable=SC2086
  result=$("$PYTHON" "$REPO_DIR/training/preprocess_scannet.py" \
      --sens-root "$STAGE/scans" --out "$OUT" --scenes "$scene" \
      --resolution "$RESOLUTION" ${TARGET_HW:+--target-hw $TARGET_HW} --fit "$FIT" \
      --max-frames "$MAX_FRAMES" --min-frames "$MIN_FRAMES" \
      --min-sharpness "$MIN_SHARPNESS" --splits-dir "$OUT/splits" \
      --workers 1 --quiet --no-index 2>&1 | tail -3)

  status=$(echo "$result" | awk -v s="$scene" '$1==s {print $2}' | tail -1)
  case "$status" in
    ok|cached|skip) mkdir -p "$OUT/.done" && touch "$marker" ;;
    *) echo "[ERROR] $scene: ${result:-no output from preprocessor}" ;;
  esac
  [[ "$KEEP_SENS" == 1 ]] || rm -rf "$STAGE/scans/$scene"
  echo "[$(printf '%-5s' "${status:-err}")] $scene ($size) $(echo "$result" | tail -1 | cut -d' ' -f3-)"
  exit 0
fi

# ---- argument parsing ------------------------------------------------------ #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --downloader)    DOWNLOADER="$2"; shift 2 ;;
    --stage)         STAGE="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --python)        PYTHON="$2"; shift 2 ;;
    --jobs|-j)       JOBS="$2"; shift 2 ;;
    --resolution)    RESOLUTION="$2"; shift 2 ;;
    --target-hw)     TARGET_HW="$2 $3"; shift 3 ;;
    --fit)           FIT="$2"; shift 2 ;;
    --max-frames)    MAX_FRAMES="$2"; shift 2 ;;
    --min-frames)    MIN_FRAMES="$2"; shift 2 ;;
    --min-sharpness) MIN_SHARPNESS="$2"; shift 2 ;;
    --limit)         LIMIT="$2"; shift 2 ;;
    --retries)       RETRIES="$2"; shift 2 ;;
    --timeout)       DL_TIMEOUT="$2"; shift 2 ;;
    --scenes-file)   SCENES_FILE="$2"; shift 2 ;;
    --keep-sens)     KEEP_SENS=1; shift ;;
    --use-downloader) USE_DOWNLOADER=1; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ "$USE_DOWNLOADER" == 1 && ! -f "$DOWNLOADER" ]] && { echo "download-scannet.py not found at $DOWNLOADER (pass --downloader)" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "python not found at $PYTHON (pass --python)" >&2; exit 1; }

mkdir -p "$STAGE" "$OUT/splits" "$OUT/.done"

# ---- scan list and official splits ---------------------------------------- #
if [[ ! -s "$OUT/splits/scans.txt" ]]; then
  echo "fetching scan list..."
  curl -sfL --retry 3 "$BASE/v2/scans.txt" -o "$OUT/splits/scans.txt" \
    || { echo "could not fetch the scan list" >&2; exit 1; }
fi
# The benchmark split keeps both scans of a room on the same side, which a random
# holdout would not; preprocess_scannet.py picks it up from --splits-dir.
for f in scannetv2_train.txt scannetv2_val.txt; do
  [[ -s "$OUT/splits/$f" ]] || curl -sfL --retry 3 \
    "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/$f" \
    -o "$OUT/splits/$f" || echo "warning: could not fetch $f; falling back to a random val split" >&2
done

if [[ -n "$SCENES_FILE" ]]; then
  cp "$SCENES_FILE" "$OUT/splits/.todo"
else
  cp "$OUT/splits/scans.txt" "$OUT/splits/.todo"
fi
[[ "$LIMIT" -gt 0 ]] && head -n "$LIMIT" "$OUT/splits/.todo" > "$OUT/splits/.todo.n" && mv "$OUT/splits/.todo.n" "$OUT/splits/.todo"

total=$(wc -l < "$OUT/splits/.todo")
done_already=$(find "$OUT/.done" -type f 2>/dev/null | wc -l)
echo "ScanNet v2 -> $OUT"
echo "  $total scans queued, $done_already already done, $JOBS parallel jobs"
echo "  staging .sens in $STAGE (deleted after each scan; peak ~$((JOBS + 1)) GB)"
echo "  accepting the ScanNet TOS: $BASE/ScanNet_TOS.pdf"
echo

export SCANNET_W_STAGE="$STAGE" SCANNET_W_OUT="$OUT" SCANNET_W_PYTHON="$PYTHON"
export SCANNET_W_DOWNLOADER="$DOWNLOADER" SCANNET_W_REPO="$REPO_DIR"
export SCANNET_W_RES="$RESOLUTION" SCANNET_W_MAXF="$MAX_FRAMES" SCANNET_W_MINF="$MIN_FRAMES"
export SCANNET_W_SHARP="$MIN_SHARPNESS" SCANNET_W_RETRIES="$RETRIES"
export SCANNET_W_TIMEOUT="$DL_TIMEOUT" SCANNET_W_KEEP="$KEEP_SENS"
export SCANNET_W_USEDL="$USE_DOWNLOADER" SCANNET_W_BASE="$BASE"
export SCANNET_W_TARGETHW="$TARGET_HW" SCANNET_W_FIT="$FIT"

started=$(date +%s)
# Workers write only per-scene files, never index.json -- see --no-index in
# preprocess_scannet.py for why the index is rebuilt afterwards instead.
xargs -a "$OUT/splits/.todo" -P "$JOBS" -I{} "${BASH_SOURCE[0]}" --worker {}

echo
echo "rebuilding index..."
# shellcheck disable=SC2086
"$PYTHON" "$REPO_DIR/training/preprocess_scannet.py" \
    --sens-root "$STAGE/scans" --out "$OUT" --splits-dir "$OUT/splits" \
    --resolution "$RESOLUTION" ${TARGET_HW:+--target-hw $TARGET_HW} --fit "$FIT" \
    --max-frames "$MAX_FRAMES" --index-only

elapsed=$(( $(date +%s) - started ))
echo "done in $((elapsed / 60))m. output $(du -sh "$OUT" 2>/dev/null | cut -f1) in $OUT"
echo
echo "train with:"
echo "  --data-root $OUT --depth-root $OUT/depth"
