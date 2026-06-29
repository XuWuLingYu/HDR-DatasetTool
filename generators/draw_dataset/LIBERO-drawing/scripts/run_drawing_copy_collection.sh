#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python}"
OUT="${1:-outputs/drawing_copy_episode}"
if [[ $# -gt 0 ]]; then
  shift
fi
SEED="${SEED:-0}"
CAMERA_SIZE="${CAMERA_SIZE:-256}"
STRIDE="${STRIDE:-3}"
FPS="${FPS:-20}"
RECORD_EVERY="${RECORD_EVERY:-3}"
SPEED_SCALE="${SPEED_SCALE:-2.0}"
HORIZON="${HORIZON:-5000}"
USE_WELD="${USE_WELD:-0}"

EXTRA_ARGS=()
if [[ "$USE_WELD" == "1" || "$USE_WELD" == "true" || "$USE_WELD" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-weld)
fi

"$PYTHON_BIN" scripts/collect_drawing_copy_dataset.py \
  --out "$OUT" \
  --seed "$SEED" \
  --camera-size "$CAMERA_SIZE" \
  --stride "$STRIDE" \
  --record-every "$RECORD_EVERY" \
  --speed-scale "$SPEED_SCALE" \
  --horizon "$HORIZON" \
  --fps "$FPS" \
  "${EXTRA_ARGS[@]}" \
  "$@"

"$PYTHON_BIN" scripts/verify_drawing_copy_episode.py "$OUT"
