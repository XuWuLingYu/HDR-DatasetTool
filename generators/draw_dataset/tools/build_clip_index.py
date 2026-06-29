#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-length video clip records from generated episodes.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--clip-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_len <= 0:
        raise ValueError("--clip-len must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    dataset_dir = args.dataset_dir
    out_path = args.out or dataset_dir / "clip_index.jsonl"
    episode_dirs = sorted(p for p in dataset_dir.glob("episode_*") if p.is_dir())

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ep in episode_dirs:
            with (ep / "meta.json").open("r", encoding="utf-8") as mf:
                meta = json.load(mf)
            n = int(meta["num_frames"])
            if n < args.clip_len:
                continue
            for start in range(0, n - args.clip_len + 1, args.stride):
                rec = {
                    "episode_id": meta["episode_id"],
                    "episode_dir": str(ep),
                    "video": str(ep / meta["video"]),
                    "target": str(ep / meta["target"]),
                    "actions": str(ep / meta["actions"]),
                    "states": str(ep / meta["states"]),
                    "start_frame": start,
                    "end_frame": start + args.clip_len,
                    "clip_len": args.clip_len,
                    "fps": meta["fps"],
                    "sketch_type": meta["sketch_type"],
                }
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                count += 1

    print(f"wrote {count} clips to {out_path}")


if __name__ == "__main__":
    main()

