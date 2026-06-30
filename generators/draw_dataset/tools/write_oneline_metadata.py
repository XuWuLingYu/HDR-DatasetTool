#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_draw_dataset.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Write prompt/video JSONL from generated one-line mp4 files.")
    parser.add_argument("--config", type=Path, default=Path("configs/one_stroke_interwoven.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--metadata-name", default="metadata.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    shard_size = int(cfg.get("output", {}).get("shard_size", 1000))
    video_prefix = str(cfg.get("output", {}).get("video_prefix", "oneline"))
    prompt = str(cfg.get("prompt", {}).get("fixed") or cfg.get("prompt", {}).get("template") or "")
    missing = []
    metadata_path = args.out / args.metadata_name
    with metadata_path.open("w", encoding="utf-8") as f:
        for idx in range(args.total):
            video_path = args.out / f"shard_{idx // shard_size:04d}" / f"{video_prefix}{idx:06d}.mp4"
            if not video_path.exists() or video_path.stat().st_size <= 0:
                missing.append(idx)
                continue
            f.write(json.dumps({"prompt": prompt, "video": str(video_path)}, ensure_ascii=False, sort_keys=True) + "\n")
    if missing:
        missing_path = args.out / "missing_indices.txt"
        missing_path.write_text("\n".join(str(i) for i in missing) + "\n", encoding="utf-8")
        raise SystemExit(f"missing {len(missing)} videos; wrote {missing_path}")
    stale_missing = args.out / "missing_indices.txt"
    if stale_missing.exists():
        stale_missing.unlink()
    print(metadata_path.resolve())


if __name__ == "__main__":
    main()
