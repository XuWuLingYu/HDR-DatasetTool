#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def complete_stems(directory: Path) -> list[str]:
    stems = []
    for json_path in sorted(directory.glob("irregular_push_maze_*.json")):
        mp4_path = json_path.with_suffix(".mp4")
        if mp4_path.exists():
            stems.append(json_path.stem)
    return stems


def wait_for(directory: Path, target_count: int, poll_seconds: int, label: str) -> list[str]:
    while True:
        stems = complete_stems(directory)
        if len(stems) >= target_count:
            return stems[:target_count]
        print(f"{label}: {len(stems)}/{target_count}", flush=True)
        time.sleep(poll_seconds)


def copy_pair(src_dir: Path, stem: str, dst_dir: Path, index: int) -> None:
    dst_stem = f"irregular_push_maze_{index:04d}"
    shutil.copy2(src_dir / f"{stem}.json", dst_dir / f"{dst_stem}.json")
    shutil.copy2(src_dir / f"{stem}.mp4", dst_dir / f"{dst_stem}.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge final irregular push maze buckets into one 5k directory.")
    parser.add_argument("--long-dir", type=Path, default=Path("irregular_push_maze_final_long_side_1k"))
    parser.add_argument("--multi-dir", type=Path, default=Path("irregular_push_maze_final_multi_push_4k"))
    parser.add_argument("--output-dir", type=Path, default=Path("irregular_push_maze_final_5k"))
    parser.add_argument("--long-count", type=int, default=1000)
    parser.add_argument("--multi-count", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    long_stems = wait_for(args.long_dir, args.long_count, args.poll_seconds, "long-side")
    multi_stems = wait_for(args.multi_dir, args.multi_count, args.poll_seconds, "multi-push")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = args.long_count + args.multi_count
    for i, stem in enumerate(long_stems):
        copy_pair(args.long_dir, stem, args.output_dir, i)
    for offset, stem in enumerate(multi_stems):
        copy_pair(args.multi_dir, stem, args.output_dir, args.long_count + offset)

    print(f"merged {total} samples into {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
