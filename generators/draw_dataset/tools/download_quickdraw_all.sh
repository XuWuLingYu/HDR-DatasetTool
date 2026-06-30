#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/data/quickdraw_ndjson}"
JOBS="${JOBS:-8}"
CATEGORIES_FILE="$ROOT/data/quickdraw_categories.txt"

source /mnt/zezhong/scripts/source_download_proxy.sh
mkdir -p "$OUT_DIR" "$(dirname "$CATEGORIES_FILE")"

python - <<'PY'
from pathlib import Path
import urllib.request

url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
out = Path("data/quickdraw_categories.txt")
text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
cats = [line.strip() for line in text.splitlines() if line.strip()]
out.write_text("\n".join(cats) + "\n", encoding="utf-8")
print(f"wrote {len(cats)} categories to {out}")
PY

download_one() {
  local cat="$1"
  local encoded="${cat// /%20}"
  local out="$OUT_DIR/${cat}.ndjson"
  local tmp="$out.part"
  if [[ -s "$out" ]]; then
    echo "exists $cat"
    return 0
  fi
  echo "download $cat"
  wget -q -c -O "$tmp" "https://storage.googleapis.com/quickdraw_dataset/full/simplified/${encoded}.ndjson"
  mv "$tmp" "$out"
}

export OUT_DIR
export -f download_one

xargs -a "$CATEGORIES_FILE" -I{} -P "$JOBS" bash -lc 'download_one "$@"' _ {}

echo "done: $(find "$OUT_DIR" -maxdepth 1 -name '*.ndjson' | wc -l) files in $OUT_DIR"
