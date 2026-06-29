#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_draw_dataset.config import load_config
from sim_draw_dataset.dataset import generate_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume synthetic one-line video generation without overwriting existing mp4 files.")
    parser.add_argument("--config", type=Path, default=Path("configs/one_stroke_interwoven.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--seed", type=int, default=190)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    shard_size = int(cfg.get("output", {}).get("shard_size", 1000))
    video_prefix = str(cfg.get("output", {}).get("video_prefix", "oneline"))
    written = 0
    skipped = 0
    for idx in range(args.worker_id, args.total, args.num_workers):
        shard = args.out / f"shard_{idx // shard_size:04d}"
        video_path = shard / f"{video_prefix}{idx:06d}.mp4"
        if video_path.exists() and video_path.stat().st_size > 0:
            skipped += 1
            continue
        shard.mkdir(parents=True, exist_ok=True)
        generate_episode(cfg, video_path, idx, args.seed + idx * 9973)
        written += 1
        if not args.quiet and written % 100 == 0:
            print(f"worker {args.worker_id}: written={written} skipped={skipped}", flush=True)
    print(f"done worker={args.worker_id}/{args.num_workers} written={written} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
