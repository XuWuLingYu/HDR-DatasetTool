#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-quickdraw_text2video_100k}"
WORKERS="${WORKERS:-16}"
TOTAL="${TOTAL:-100000}"
SEED="${SEED:-7}"

rm -rf "$OUT"
mkdir -p "$OUT"

for wid in $(seq 0 $((WORKERS - 1))); do
  python tools/generate_from_quickdraw.py \
    --config configs/default.yaml \
    --quickdraw-dir data/quickdraw_ndjson \
    --categories-file data/quickdraw_categories.txt \
    --out "$OUT" \
    --total "$TOTAL" \
    --seed "$SEED" \
    --num-workers "$WORKERS" \
    --worker-id "$wid" \
    --quiet > "$OUT/worker_${wid}.log" 2>&1 &
done

wait

cat "$OUT"/metadata.worker*.jsonl > "$OUT/metadata.jsonl"
wc -l "$OUT/metadata.jsonl"
