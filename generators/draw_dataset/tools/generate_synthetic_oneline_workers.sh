#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-outputs/draw_dataset/synthetic_oneline_copy_10k}"
WORKERS="${WORKERS:-16}"
TOTAL="${TOTAL:-10000}"
SEED="${SEED:-23}"
CONFIG="${CONFIG:-configs/one_stroke_copy.yaml}"

rm -rf "$OUT"
mkdir -p "$OUT"

for wid in $(seq 0 $((WORKERS - 1))); do
  python tools/generate_synthetic_oneline.py \
    --config "$CONFIG" \
    --out "$OUT" \
    --total "$TOTAL" \
    --seed "$SEED" \
    --num-workers "$WORKERS" \
    --worker-id "$wid" \
    --quiet > "$OUT/worker_${wid}.log" 2>&1 &
done

wait
cat "$OUT"/metadata.worker*.jsonl > "$OUT"/metadata.jsonl
wc -l "$OUT"/metadata.jsonl
