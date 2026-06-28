#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-quickdraw_onestroke_copy_100k}"
WORKERS="${WORKERS:-16}"
TOTAL="${TOTAL:-100000}"
SEED="${SEED:-17}"

rm -rf "$OUT"
mkdir -p "$OUT"

for wid in $(seq 0 $((WORKERS - 1))); do
  python tools/generate_from_quickdraw.py \
    --config configs/one_stroke_copy.yaml \
    --quickdraw-dir data/quickdraw_ndjson \
    --categories-file data/quickdraw_categories.txt \
    --out "$OUT" \
    --total "$TOTAL" \
    --seed "$SEED" \
    --single-stroke-only \
    --num-workers "$WORKERS" \
    --worker-id "$wid" \
    --quiet > "$OUT/worker_${wid}.log" 2>&1 &
done

wait

cat "$OUT"/metadata.worker*.jsonl > "$OUT"/metadata.jsonl
wc -l "$OUT"/metadata.jsonl
