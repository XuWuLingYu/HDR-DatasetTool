#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import generate_irregular_push_maze as maze


def worker(args: tuple[int, int, int, int, int]) -> maze.MazeSample | None:
    seed, block_count, min_moved_blocks, max_attempts, salt = args
    rng = random.Random(seed + salt * 1_000_003)
    for _ in range(max_attempts):
        trial_seed = rng.randrange(2**31)
        try:
            sample = maze.generate_one(
                trial_seed,
                block_count=block_count,
                max_attempts=80,
                push_side="any",
                min_moved_blocks=min_moved_blocks,
            )
        except RuntimeError:
            continue
        if len(maze.moved_block_indices(sample.solution_block_paths)) >= min_moved_blocks:
            return sample
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool irregular push mazes that require pushing multiple different blocks.")
    parser.add_argument("--output-dir", type=Path, default=Path("irregular_push_maze_multi_block"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-count", type=int, default=3)
    parser.add_argument("--min-moved-blocks", type=int, default=2)
    parser.add_argument("--worker-attempts", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    task_id = 0
    with mp.Pool(processes=args.workers) as pool:
        pending: list[mp.pool.ApplyResult] = []
        while saved < args.count:
            while len(pending) < args.workers * 4:
                pending.append(
                    pool.apply_async(
                        worker,
                        ((args.seed, args.block_count, args.min_moved_blocks, args.worker_attempts, task_id),),
                    )
                )
                task_id += 1
            sample = pending.pop(0).get()
            if sample is None:
                continue
            maze.save_sample(sample, args.output_dir, saved)
            saved += 1
            if saved == 1 or saved % args.progress_every == 0 or saved == args.count:
                moved = len(maze.moved_block_indices(sample.solution_block_paths))
                print(f"{saved}/{args.count} moved_blocks={moved} raw_push_steps={sample.raw_push_steps}", flush=True)


if __name__ == "__main__":
    main()
