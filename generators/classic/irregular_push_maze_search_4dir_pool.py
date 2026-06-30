#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from pathlib import Path

import generate_irregular_push_maze as maze


DIRECTION_ORDER = ("up", "down", "left", "right")
DIRECTION_VECTORS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def sample_direction(sample: maze.MazeSample) -> str | None:
    directions = maze.pushed_directions(sample.solution_block_paths)
    if not directions:
        return None
    totals = (sum(dx for dx, _dy in directions), sum(dy for _dx, dy in directions))
    if totals[0] > 0 and totals[1] == 0:
        return "right"
    if totals[0] < 0 and totals[1] == 0:
        return "left"
    if totals[1] > 0 and totals[0] == 0:
        return "down"
    if totals[1] < 0 and totals[0] == 0:
        return "up"
    return None


def long_direction_ok(sample: maze.MazeSample, direction: str) -> bool:
    expected = DIRECTION_VECTORS[direction]
    directions = maze.pushed_directions(sample.solution_block_paths)
    return bool(directions) and all(step == expected for step in directions)


def worker(args: tuple[int, int, int]) -> tuple[str, maze.MazeSample] | None:
    seed, max_attempts, salt = args
    rng = random.Random(seed + salt * 1_000_003)
    for _ in range(max_attempts):
        trial_seed = rng.randrange(2**31)
        try:
            sample = maze.generate_one(trial_seed, block_count=1, max_attempts=40, push_side="long_only")
        except RuntimeError:
            continue
        direction = sample_direction(sample)
        if direction is None:
            continue
        if long_direction_ok(sample, direction):
            return direction, sample
    return None


def output_index(direction: str, offset: int, count_per_direction: int) -> int:
    return DIRECTION_ORDER.index(direction) * count_per_direction + offset


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool irregular long-side push mazes into balanced up/down/left/right directions.")
    parser.add_argument("--output-dir", type=Path, default=Path("irregular_push_maze_search_long_4dir_4k"))
    parser.add_argument("--count-per-direction", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--worker-attempts", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {args.output_dir.resolve()}", flush=True)
    counts = {direction: 0 for direction in DIRECTION_ORDER}
    total_needed = len(DIRECTION_ORDER) * args.count_per_direction
    task_id = 0

    with mp.Pool(processes=args.workers) as pool:
        pending: list[mp.pool.ApplyResult] = []
        while sum(counts.values()) < total_needed:
            while len(pending) < args.workers * 4:
                pending.append(pool.apply_async(worker, ((args.seed, args.worker_attempts, task_id),)))
                task_id += 1
            ready_idx = next((i for i, task in enumerate(pending) if task.ready()), None)
            if ready_idx is None:
                time.sleep(0.1)
                continue
            result = pending.pop(ready_idx).get()
            if result is None:
                continue
            direction, sample = result
            if counts[direction] >= args.count_per_direction:
                continue
            index = output_index(direction, counts[direction], args.count_per_direction)
            maze.save_sample(sample, args.output_dir, index)
            counts[direction] += 1
            done = sum(counts.values())
            if done == 1 or done % args.progress_every == 0 or done == total_needed:
                print(f"{done}/{total_needed} counts={counts}", flush=True)


if __name__ == "__main__":
    main()
