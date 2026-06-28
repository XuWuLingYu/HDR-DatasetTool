#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from pathlib import Path

import generate_irregular_push_maze as maze


DIRECTIONS = ("up", "down", "left", "right")


def worker(args: tuple[int, int, int, int, str | None, int, int, str]) -> maze.MazeSample | None:
    seed, block_count, min_moved_blocks, max_attempts, target_direction, worker_attempts, salt, push_side = args
    rng = random.Random(seed + salt * 1_000_003)
    for _ in range(worker_attempts):
        trial_seed = rng.randrange(2**31)
        try:
            return maze.generate_required_blocks_one(
                trial_seed,
                block_count=block_count,
                min_moved_blocks=min_moved_blocks,
                max_attempts=max_attempts,
                target_direction=target_direction,
                push_side=push_side,
            )
        except RuntimeError:
            continue
    return None


def save_sample(sample: maze.MazeSample, output_dir: Path, index: int) -> None:
    maze.save_sample(sample, output_dir, index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool required-block irregular push mazes.")
    parser.add_argument("--output-dir", type=Path, default=Path("irregular_push_maze_required_pool"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-count", type=int, default=3)
    parser.add_argument("--min-moved-blocks", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--worker-attempts", type=int, default=8)
    parser.add_argument("--target-direction", type=str, default=None, choices=[None, "up", "down", "left", "right"])
    parser.add_argument("--balanced-directions", action="store_true")
    parser.add_argument("--push-side", type=str, default="any", choices=["any", "long_only", "short_only"])
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_id = 0
    saved = 0
    counts = {direction: 0 for direction in DIRECTIONS}
    per_direction = args.count // len(DIRECTIONS) if args.balanced_directions else None

    with mp.Pool(processes=args.workers) as pool:
        pending: list[mp.pool.ApplyResult] = []
        while saved < args.count:
            while len(pending) < args.workers * 4:
                if args.balanced_directions:
                    direction = next((d for d in DIRECTIONS if counts[d] < per_direction), None)
                    if direction is None:
                        break
                else:
                    direction = args.target_direction
                pending.append(
                    pool.apply_async(
                        worker,
                        (
                            (
                                args.seed,
                                args.block_count,
                                args.min_moved_blocks,
                                args.max_attempts,
                                direction,
                                args.worker_attempts,
                                task_id,
                                args.push_side,
                            ),
                        ),
                    )
                )
                task_id += 1
            ready_index = next((index for index, item in enumerate(pending) if item.ready()), None)
            if ready_index is None:
                time.sleep(1.0)
                continue
            result = pending.pop(ready_index).get()
            if result is None:
                continue
            if args.balanced_directions:
                dirs = maze.solution_moved_directions(result.solution_block_paths)
                direction = next((d for d in DIRECTIONS if d in dirs and counts[d] < per_direction), None)
                if direction is None:
                    continue
                index = DIRECTIONS.index(direction) * per_direction + counts[direction]
                counts[direction] += 1
            else:
                index = saved
            save_sample(result, args.output_dir, index)
            saved += 1
            if saved == 1 or saved % args.progress_every == 0 or saved == args.count:
                moved = len(maze.moved_block_indices(result.solution_block_paths))
                print(f"{saved}/{args.count} moved_blocks={moved} directions={maze.solution_moved_directions(result.solution_block_paths)} counts={counts}", flush=True)


if __name__ == "__main__":
    main()
