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
from sim_draw_dataset.dataset import generate_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ornamental one-line copy videos.")
    parser.add_argument("--config", type=Path, default=Path("configs/one_stroke_copy.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/draw_dataset/synthetic_oneline_copy_10k"))
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not (0 <= args.worker_id < args.num_workers):
        raise ValueError("--worker-id must satisfy 0 <= worker-id < num-workers")

    cfg = load_config(args.config)
    if args.out.exists() and args.overwrite and args.worker_id == 0:
        import shutil

        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    shard_size = int(cfg.get("output", {}).get("shard_size", 1000))
    video_prefix = str(cfg.get("output", {}).get("video_prefix", "oneline"))
    manifest_name = "metadata.jsonl" if args.num_workers == 1 else f"metadata.worker{args.worker_id:02d}.jsonl"
    manifest = (args.out / manifest_name).open("w", encoding="utf-8")
    written = 0
    try:
        for idx in range(args.worker_id, args.total, args.num_workers):
            shard = args.out / f"shard_{idx // shard_size:04d}"
            shard.mkdir(parents=True, exist_ok=True)
            video_path = shard / f"{video_prefix}{idx:06d}.mp4"
            meta = generate_episode(cfg, video_path, idx, args.seed + idx * 9973)
            manifest.write(json.dumps({"prompt": meta["prompt"], "video": str(video_path)}, ensure_ascii=False, sort_keys=True) + "\n")
            if not args.quiet:
                print(f"wrote {video_path} frames={meta['num_frames']}")
            elif written > 0 and written % 1000 == 0:
                print(f"worker {args.worker_id}: wrote {written}")
            written += 1
    finally:
        manifest.close()
    print(f"done worker={args.worker_id}/{args.num_workers} videos={written}")


if __name__ == "__main__":
    main()
