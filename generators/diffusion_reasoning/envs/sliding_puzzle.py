from __future__ import annotations

import heapq
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base_env import BaseEnv

PROMPT = (
    "The scene shows a grid containing sliding tiles and one empty space. The puzzle rule is: "
    "identify the current configuration of the tiles, then determine the valid sequence of moves "
    "where an adjacent tile slides horizontally or vertically into the empty space. Observe the "
    "initial state and calculate the sequence of sliding actions required to reach the target ordered "
    "state. Animate each tile moving smoothly into the adjacent empty space one by one, ensuring all "
    "tiles end up in their exact final positions without overlapping or clipping through other tiles."
)

State = tuple[int, ...]
Move = tuple[int, int]


@dataclass(frozen=True)
class BucketSpec:
    name: str
    topology_split: str
    board_size: int
    quota: int
    family: str
    scramble_min: int
    scramble_max: int
    optimal_min: int
    optimal_max: int
    min_manhattan_sum: int
    min_misplaced_tiles: int
    max_correct_tiles: int
    min_state_distance: int
    max_retries: int


@dataclass(frozen=True)
class PuzzleSpec:
    sample_id: str
    difficulty_level: int
    split: str
    topology_split: str
    bucket_name: str
    family: str
    board_size: int
    initial_state: State
    goal_state: State
    solution_moves: tuple[Move, ...]
    frame_count: int
    fps: int
    resolution: tuple[int, int]
    metrics: dict[str, int | float]


class SlidingPuzzleEnv(BaseEnv):
    task_name = "sliding_puzzle"
    _distance_cache_3x3: dict[State, int] | None = None
    _blank_neighbors_cache: dict[int, tuple[tuple[int, ...], ...]] = {}
    _goal_positions_cache: dict[int, tuple[tuple[int, int], ...]] = {}

    def __init__(self, env_config: dict[str, Any]) -> None:
        super().__init__(env_config)
        global_config = env_config.get("global", {})
        render = env_config.get("render", {})
        generation = env_config.get("generation", {})
        solver = env_config.get("solver", {})

        self.output_root = Path(env_config.get("output_root", global_config.get("default_output_root", "data")))
        self.levels = env_config.get("levels", {})
        self.global_seed = int(env_config.get("seed", 0))
        self.default_workers = int(generation.get("workers", max(1, min(6, os.cpu_count() or 1))))

        self.resolution = tuple(render.get("resolution", [208, 208]))
        self.fps = int(render.get("fps", 10))
        duration = render.get("duration_seconds", {})
        self.min_frames = int(duration.get("min", 3) * self.fps)
        self.max_frames = int(duration.get("max", 7) * self.fps)
        self.base_frames_per_move = int(render.get("base_frames_per_move", 4))
        self.min_frames_per_move = int(render.get("min_frames_per_move", 2))
        self.max_frames_per_move = int(render.get("max_frames_per_move", 6))
        self.start_hold_frames = int(render.get("start_hold_frames", 2))
        self.end_hold_frames = int(render.get("end_hold_frames", 4))
        self.style = render.get("style", {})

        self.max_nodes_by_size = {
            int(size): int(limit) for size, limit in solver.get("max_nodes_by_size", {3: 50000, 4: 120000}).items()
        }

    def generate_dataset(
        self,
        output_root: str | Path | None = None,
        levels: list[int] | None = None,
        workers: int | None = None,
    ) -> None:
        resolved_root = Path(output_root) if output_root is not None else self.output_root
        target_levels = levels or sorted(int(level) for level in self.levels.keys())
        requested_workers = self.default_workers if workers is None else max(1, int(workers))
        manifests: dict[int, dict[str, list[dict[str, Any]]]] = {
            int(level): {"train": [], "eval": []} for level in target_levels
        }
        for level in target_levels:
            for split in ("train", "eval"):
                manifests[int(level)][split] = self.generate_split(
                    output_root=resolved_root,
                    level=int(level),
                    split=split,
                    workers=requested_workers,
                )

        for level in target_levels:
            difficulty_dir = resolved_root / f"sliding_puzzle_{level}level"
            for split, entries in manifests[int(level)].items():
                manifest_path = difficulty_dir / f"{split}_manifest.json"
                with manifest_path.open("w", encoding="utf-8") as handle:
                    json.dump(entries, handle, indent=2, ensure_ascii=False)

    def generate_split(
        self,
        output_root: str | Path,
        level: int,
        split: str,
        workers: int | None = None,
    ) -> list[dict[str, Any]]:
        resolved_root = Path(output_root)
        level_key = str(level) if str(level) in self.levels else level
        level_config = self.levels[level_key]
        difficulty_dir = resolved_root / f"sliding_puzzle_{level}level"
        split_dir = self.ensure_split_dir(difficulty_dir, split)
        bucket_specs = self._build_bucket_schedule(level_config, split)
        self._validate_bucket_search_space(bucket_specs)
        worker_count = self.default_workers if workers is None else max(1, int(workers))

        if bucket_specs and all(bucket.board_size == 4 for bucket in bucket_specs) and worker_count > 1:
            puzzle_specs = self._generate_parallel_specs(level, split, bucket_specs, worker_count)
        else:
            puzzle_specs = self._generate_sequential_specs(level, split, bucket_specs)

        entries: list[dict[str, Any]] = []
        total_specs = len(puzzle_specs)
        render_interval = max(1, total_specs // 20) if total_specs else 1
        for render_index, puzzle in enumerate(puzzle_specs, start=1):
            artifacts = self._render_sample(split_dir, puzzle)
            entries.append(
                {
                    "sample_id": puzzle.sample_id,
                    "video": artifacts["video"],
                    "metadata": artifacts["metadata"],
                    "topology_split": puzzle.topology_split,
                    "bucket": puzzle.bucket_name,
                }
            )
            if render_index == 1 or render_index == total_specs or render_index % render_interval == 0:
                print(
                    f"[sliding_puzzle] level {level} {split}: rendered {render_index}/{total_specs} samples",
                    file=sys.stderr,
                    flush=True,
                )
        return entries

    def _generate_sequential_specs(
        self,
        level: int,
        split: str,
        bucket_specs: list[BucketSpec],
    ) -> list[PuzzleSpec]:
        seen_states: set[State] = set()
        kept_states: list[State] = []
        puzzle_specs: list[PuzzleSpec] = []

        for index, bucket in enumerate(bucket_specs):
            puzzle = self._make_puzzle_spec(level, split, index, bucket, seen_states, kept_states)
            kept_states.append(puzzle.initial_state)
            puzzle_specs.append(puzzle)
        return puzzle_specs

    def _generate_parallel_specs(
        self,
        level: int,
        split: str,
        bucket_specs: list[BucketSpec],
        worker_count: int,
    ) -> list[PuzzleSpec]:
        tasks = [
            {
                "env_config": self.config,
                "level": level,
                "split": split,
                "index": index,
                "bucket": bucket.__dict__,
                "seed_offset": 0,
            }
            for index, bucket in enumerate(bucket_specs)
        ]
        worker_count = min(worker_count, len(tasks))
        if worker_count == 1:
            raw_specs = [_generate_independent_sample_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_generate_independent_sample_task, task): task["index"]
                    for task in tasks
                }
                raw_specs = [None] * len(tasks)
                completed = 0
                total = len(tasks)
                progress_interval = max(1, total // 20)
                for future in as_completed(futures):
                    index = futures[future]
                    raw_specs[index] = future.result()
                    completed += 1
                    if completed == 1 or completed == total or completed % progress_interval == 0:
                        print(
                            f"[sliding_puzzle] level {level} {split}: generated {completed}/{total} candidate specs",
                            file=sys.stderr,
                            flush=True,
                        )

        seen_states: set[State] = set()
        kept_states: list[State] = []
        puzzle_specs: list[PuzzleSpec] = []
        total = len(raw_specs)
        merge_interval = max(1, total // 20) if total else 1
        for index, (bucket, raw_spec) in enumerate(zip(bucket_specs, raw_specs)):
            puzzle = self._deserialize_puzzle_spec(raw_spec)
            if puzzle.initial_state in seen_states or not self._far_enough(puzzle.initial_state, kept_states, bucket.min_state_distance):
                puzzle = self._regenerate_unique_parallel_spec(level, split, index, bucket, seen_states, kept_states)
            seen_states.add(puzzle.initial_state)
            kept_states.append(puzzle.initial_state)
            puzzle_specs.append(puzzle)
            merged = index + 1
            if merged == 1 or merged == total or merged % merge_interval == 0:
                print(
                    f"[sliding_puzzle] level {level} {split}: finalized {merged}/{total} samples",
                    file=sys.stderr,
                    flush=True,
                )
        return puzzle_specs

    def _regenerate_unique_parallel_spec(
        self,
        level: int,
        split: str,
        index: int,
        bucket: BucketSpec,
        seen_states: set[State],
        kept_states: list[State],
    ) -> PuzzleSpec:
        for seed_offset in range(1, 512):
            puzzle = self._make_independent_puzzle_spec(level, split, index, bucket, seed_offset=seed_offset)
            if puzzle.initial_state in seen_states:
                continue
            if not self._far_enough(puzzle.initial_state, kept_states, bucket.min_state_distance):
                continue
            return puzzle
        raise RuntimeError(f"Failed to regenerate a unique parallel sliding puzzle for bucket {bucket.name}")

    def _make_independent_puzzle_spec(
        self,
        level: int,
        split: str,
        index: int,
        bucket: BucketSpec,
        seed_offset: int = 0,
    ) -> PuzzleSpec:
        return self._make_puzzle_spec(
            level=level,
            split=split,
            index=index,
            bucket=bucket,
            seen_states=set(),
            kept_states=[],
            seed_offset=seed_offset,
        )

    def _build_bucket_schedule(self, level_config: dict[str, Any], split: str) -> list[BucketSpec]:
        if split == "train":
            bucket_dicts = self._scale_bucket_dicts(
                list(level_config.get("train_buckets", [])),
                int(level_config["counts"]["train"]),
            )
            return self._expand_bucket_schedule(bucket_dicts)

        total = int(level_config["counts"]["eval"])
        ood_ratio = float(level_config.get("eval_ood_ratio", 0.4))
        ood_total = min(total, round(total * ood_ratio))
        id_total = total - ood_total
        bucket_dicts = self._scale_bucket_dicts(list(level_config.get("eval_id_buckets", [])), id_total)
        bucket_dicts.extend(self._scale_bucket_dicts(list(level_config.get("eval_ood_buckets", [])), ood_total))
        return self._expand_bucket_schedule(bucket_dicts)

    def _expand_bucket_schedule(self, bucket_dicts: list[dict[str, Any]]) -> list[BucketSpec]:
        schedule: list[BucketSpec] = []
        for bucket in bucket_dicts:
            spec = self._bucket_from_dict(bucket)
            schedule.extend([spec] * spec.quota)
        return schedule

    def _scale_bucket_dicts(self, bucket_dicts: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
        if total <= 0 or not bucket_dicts:
            return []
        raw_weights = [float(bucket.get("quota", bucket.get("weight", 1))) for bucket in bucket_dicts]
        weight_sum = sum(raw_weights)
        provisional = [int(math.floor(total * weight / max(weight_sum, 1.0))) for weight in raw_weights]
        remainder = total - sum(provisional)
        order = sorted(range(len(bucket_dicts)), key=lambda idx: (raw_weights[idx], -idx), reverse=True)
        for offset in range(remainder):
            provisional[order[offset % len(order)]] += 1

        scaled: list[dict[str, Any]] = []
        for bucket, quota in zip(bucket_dicts, provisional):
            if quota <= 0:
                continue
            payload = dict(bucket)
            payload["quota"] = quota
            scaled.append(payload)
        return scaled

    def _bucket_from_dict(self, payload: dict[str, Any]) -> BucketSpec:
        return BucketSpec(
            name=str(payload["name"]),
            topology_split=str(payload.get("topology_split", "id")),
            board_size=int(payload["board_size"]),
            quota=int(payload["quota"]),
            family=str(payload.get("family", "random_walk")),
            scramble_min=int(payload.get("scramble_steps_range", [8, 20])[0]),
            scramble_max=int(payload.get("scramble_steps_range", [8, 20])[1]),
            optimal_min=int(payload.get("optimal_len_range", [1, 10])[0]),
            optimal_max=int(payload.get("optimal_len_range", [1, 10])[1]),
            min_manhattan_sum=int(payload.get("min_manhattan_sum", 0)),
            min_misplaced_tiles=int(payload.get("min_misplaced_tiles", 0)),
            max_correct_tiles=int(payload.get("max_correct_tiles", 999)),
            min_state_distance=int(payload.get("min_state_distance", 2)),
            max_retries=int(payload.get("max_retries", 600)),
        )

    def _validate_bucket_search_space(self, buckets: list[BucketSpec]) -> None:
        grouped: dict[tuple[int, str], int] = {}
        for bucket in buckets:
            grouped[(bucket.board_size, bucket.topology_split)] = grouped.get((bucket.board_size, bucket.topology_split), 0) + 1

        for (board_size, _), quota in grouped.items():
            total_tiles = board_size * board_size
            estimated = math.factorial(total_tiles) // 2
            if estimated < quota * 4:
                raise RuntimeError(
                    f"Sliding puzzle search space appears undersized for {board_size}x{board_size}: "
                    f"estimate={estimated}, requested_quota={quota}"
                )

    def _make_puzzle_spec(
        self,
        level: int,
        split: str,
        index: int,
        bucket: BucketSpec,
        seen_states: set[State],
        kept_states: list[State],
        seed_offset: int = 0,
    ) -> PuzzleSpec:
        board_size = bucket.board_size
        goal_state = self._goal_state(board_size)
        base_seed = self._sample_seed(level, split, index) + seed_offset * 1_000_000

        if board_size == 3:
            candidate = self._sample_from_exact_pool(base_seed, bucket, goal_state, seen_states, kept_states)
            if candidate is None:
                raise RuntimeError(f"Failed to generate a unique sliding puzzle for bucket {bucket.name}")

            solution_moves = self._solve_optimal(candidate, goal_state, board_size)
            if solution_moves is None:
                raise RuntimeError(f"Failed to solve sampled sliding puzzle for bucket {bucket.name}")

            metrics = self._state_metrics(candidate, goal_state, solution_moves, board_size)
            frame_count = self._frame_count(len(solution_moves))
            seen_states.add(candidate)
            return PuzzleSpec(
                sample_id=f"sliding_puzzle_l{level}_{split}_{index:04d}",
                difficulty_level=level,
                split=split,
                topology_split=bucket.topology_split,
                bucket_name=bucket.name,
                family=bucket.family,
                board_size=board_size,
                initial_state=candidate,
                goal_state=goal_state,
                solution_moves=tuple(solution_moves),
                frame_count=frame_count,
                fps=self.fps,
                resolution=self.resolution,
                metrics=metrics,
            )

        for attempt in range(bucket.max_retries):
            rng = np.random.default_rng(base_seed + attempt * 9_973)
            scramble_steps = int(rng.integers(bucket.scramble_min, bucket.scramble_max + 1))
            candidate = self._sample_scrambled_state(goal_state, board_size, scramble_steps, bucket.family, rng)
            if candidate in seen_states:
                continue
            if not self._far_enough(candidate, kept_states, bucket.min_state_distance):
                continue
            cheap_metrics = self._state_metrics(candidate, goal_state, None, board_size)
            if not self._accept_non_solution_metrics(cheap_metrics, bucket):
                continue

            solution_moves = self._solve_optimal(candidate, goal_state, board_size)
            if solution_moves is None:
                continue

            metrics = dict(cheap_metrics)
            metrics["optimal_solution_length"] = len(solution_moves)
            if not self._accept_metrics(metrics, bucket):
                continue

            seen_states.add(candidate)
            frame_count = self._frame_count(len(solution_moves))
            return PuzzleSpec(
                sample_id=f"sliding_puzzle_l{level}_{split}_{index:04d}",
                difficulty_level=level,
                split=split,
                topology_split=bucket.topology_split,
                bucket_name=bucket.name,
                family=bucket.family,
                board_size=board_size,
                initial_state=candidate,
                goal_state=goal_state,
                solution_moves=tuple(solution_moves),
                frame_count=frame_count,
                fps=self.fps,
                resolution=self.resolution,
                metrics=metrics,
            )

        raise RuntimeError(f"Failed to generate a unique sliding puzzle for bucket {bucket.name}")

    def _deserialize_puzzle_spec(self, payload: dict[str, Any]) -> PuzzleSpec:
        return PuzzleSpec(
            sample_id=str(payload["sample_id"]),
            difficulty_level=int(payload["difficulty_level"]),
            split=str(payload["split"]),
            topology_split=str(payload["topology_split"]),
            bucket_name=str(payload["bucket_name"]),
            family=str(payload["family"]),
            board_size=int(payload["board_size"]),
            initial_state=tuple(int(value) for value in payload["initial_state"]),
            goal_state=tuple(int(value) for value in payload["goal_state"]),
            solution_moves=tuple((int(move[0]), int(move[1])) for move in payload["solution_moves"]),
            frame_count=int(payload["frame_count"]),
            fps=int(payload["fps"]),
            resolution=(int(payload["resolution"][0]), int(payload["resolution"][1])),
            metrics=dict(payload["metrics"]),
        )

    def _sample_from_exact_pool(
        self,
        base_seed: int,
        bucket: BucketSpec,
        goal_state: State,
        seen_states: set[State],
        kept_states: list[State],
    ) -> State | None:
        distance_map = self._distance_map_3x3(goal_state)
        eligible: list[State] = []
        relaxed: list[State] = []
        for state, distance in distance_map.items():
            if distance < bucket.optimal_min or distance > bucket.optimal_max:
                continue
            metrics = self._state_metrics(state, goal_state, [(-1, -1)] * distance, 3)
            if not self._accept_metrics(metrics, bucket):
                continue
            if self._family_matches_state(state, bucket.family, 3):
                eligible.append(state)
            else:
                relaxed.append(state)

        rng = np.random.default_rng(base_seed)
        rng.shuffle(eligible)
        rng.shuffle(relaxed)
        min_distances = list(range(bucket.min_state_distance, 0, -1)) or [0]
        if 0 not in min_distances:
            min_distances.append(0)
        for min_distance in min_distances:
            for pool in (eligible, relaxed):
                for state in pool:
                    if state in seen_states:
                        continue
                    if not self._far_enough(state, kept_states, min_distance):
                        continue
                    return state
        return None

    @classmethod
    def _distance_map_3x3(cls, goal_state: State) -> dict[State, int]:
        if cls._distance_cache_3x3 is not None:
            return cls._distance_cache_3x3

        queue: list[State] = [goal_state]
        distance_map: dict[State, int] = {goal_state: 0}
        index = 0
        temp_env = cls({"levels": {}, "render": {}, "generation": {}, "solver": {}})
        while index < len(queue):
            state = queue[index]
            index += 1
            current_distance = distance_map[state]
            for next_state, _ in temp_env._neighbors(state, 3):
                if next_state in distance_map:
                    continue
                distance_map[next_state] = current_distance + 1
                queue.append(next_state)
        cls._distance_cache_3x3 = distance_map
        return distance_map

    def _family_matches_state(self, state: State, family: str, board_size: int) -> bool:
        if family == "random_walk":
            return True
        blank_index = state.index(0)
        row, col = divmod(blank_index, board_size)
        row_switch = sum(1 for idx, value in enumerate(state) if value != 0 and idx // board_size != (value - 1) // board_size)
        col_switch = sum(1 for idx, value in enumerate(state) if value != 0 and idx % board_size != (value - 1) % board_size)
        if family == "row_bias":
            return row_switch >= col_switch
        if family == "column_bias":
            return col_switch >= row_switch
        if family == "edge_bias":
            return row in (0, board_size - 1) or col in (0, board_size - 1)
        if family == "corner_bias":
            return (row, col) in ((0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1))
        if family == "alternating":
            return (row + col) % 2 == 0
        return True

    def _sample_seed(self, level: int, split: str, index: int) -> int:
        split_offset = 0 if split == "train" else 50_000
        return self.global_seed + level * 10_000 + split_offset + index

    def _goal_state(self, board_size: int) -> State:
        tile_count = board_size * board_size
        return tuple(list(range(1, tile_count)) + [0])

    def _sample_scrambled_state(
        self,
        start_state: State,
        board_size: int,
        steps: int,
        family: str,
        rng: np.random.Generator,
    ) -> State:
        current = start_state
        previous: State | None = None
        for _ in range(steps):
            candidates = self._neighbors(current, board_size)
            if previous is not None and len(candidates) > 1:
                filtered = [candidate for candidate in candidates if candidate[0] != previous]
                if filtered:
                    candidates = filtered
            scored = [
                (self._family_score(candidate_state, board_size, family, rng), candidate_state)
                for candidate_state, _ in candidates
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            top_k = min(len(scored), 2 if len(scored) > 1 else 1)
            viable = [state for _, state in scored[:top_k]]
            next_state = viable[int(rng.integers(0, len(viable)))]
            previous, current = current, next_state
        return current

    def _family_score(
        self,
        state: State,
        board_size: int,
        family: str,
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        blank_index = state.index(0)
        row, col = divmod(blank_index, board_size)
        edge_distance = min(row, col, board_size - 1 - row, board_size - 1 - col)
        center_distance = abs(row - (board_size - 1) / 2) + abs(col - (board_size - 1) / 2)
        row_switch = sum(1 for idx, value in enumerate(state) if value != 0 and idx // board_size != (value - 1) // board_size)
        col_switch = sum(1 for idx, value in enumerate(state) if value != 0 and idx % board_size != (value - 1) % board_size)
        jitter = float(rng.random())

        if family == "row_bias":
            return (float(row_switch), -float(col_switch), jitter)
        if family == "column_bias":
            return (float(col_switch), -float(row_switch), jitter)
        if family == "edge_bias":
            return (center_distance, -float(edge_distance), jitter)
        if family == "corner_bias":
            corner_distance = min(
                abs(row - 0) + abs(col - 0),
                abs(row - 0) + abs(col - (board_size - 1)),
                abs(row - (board_size - 1)) + abs(col - 0),
                abs(row - (board_size - 1)) + abs(col - (board_size - 1)),
            )
            return (-float(corner_distance), center_distance, jitter)
        if family == "alternating":
            parity = (row + col) % 2
            return (1.0 if parity == 0 else 0.0, float(row_switch + col_switch), jitter)
        return (float(row_switch + col_switch), center_distance, jitter)

    def _neighbors(self, state: State, board_size: int) -> list[tuple[State, Move]]:
        blank_index = state.index(0)
        neighbors: list[tuple[State, Move]] = []
        for tile_index in self._blank_neighbors(board_size)[blank_index]:
            board = list(state)
            board[blank_index], board[tile_index] = board[tile_index], board[blank_index]
            neighbors.append((tuple(board), (tile_index, blank_index)))
        return neighbors

    def _solve_optimal(self, start: State, goal: State, board_size: int) -> list[Move] | None:
        if start == goal:
            return []

        max_nodes = self.max_nodes_by_size.get(board_size, 120000)
        open_heap: list[tuple[int, int, State]] = []
        parent: dict[State, tuple[State | None, Move | None]] = {start: (None, None)}
        g_score: dict[State, int] = {start: 0}
        heuristic = self._heuristic(start, board_size)
        heapq.heappush(open_heap, (heuristic, 0, start))
        closed: set[State] = set()

        while open_heap:
            _, current_g, state = heapq.heappop(open_heap)
            if state in closed:
                continue
            if state == goal:
                return self._reconstruct_moves(state, parent)
            closed.add(state)
            if len(closed) > max_nodes:
                return None

            for next_state, move in self._neighbors(state, board_size):
                tentative_g = current_g + 1
                if tentative_g >= g_score.get(next_state, 10**9):
                    continue
                g_score[next_state] = tentative_g
                parent[next_state] = (state, move)
                priority = tentative_g + self._heuristic(next_state, board_size)
                heapq.heappush(open_heap, (priority, tentative_g, next_state))
        return None

    def _reconstruct_moves(
        self,
        goal_state: State,
        parent: dict[State, tuple[State | None, Move | None]],
    ) -> list[Move]:
        moves: list[Move] = []
        state = goal_state
        while True:
            prev_state, move = parent[state]
            if prev_state is None or move is None:
                break
            moves.append(move)
            state = prev_state
        moves.reverse()
        return moves

    def _manhattan_sum(self, state: State, board_size: int) -> int:
        goal_positions = self._goal_positions(board_size)
        total = 0
        for index, tile in enumerate(state):
            if tile == 0:
                continue
            row, col = divmod(index, board_size)
            goal_row, goal_col = goal_positions[tile]
            total += abs(row - goal_row) + abs(col - goal_col)
        return total

    def _heuristic(self, state: State, board_size: int) -> int:
        manhattan = self._manhattan_sum(state, board_size)
        if board_size < 4:
            return manhattan
        return manhattan + self._linear_conflict(state, board_size)

    def _linear_conflict(self, state: State, board_size: int) -> int:
        conflicts = 0
        goal_positions = self._goal_positions(board_size)
        for row in range(board_size):
            row_tiles: list[tuple[int, int]] = []
            for col in range(board_size):
                tile = state[row * board_size + col]
                if tile == 0:
                    continue
                goal_row, goal_col = goal_positions[tile]
                if goal_row == row:
                    row_tiles.append((col, goal_col))
            for idx in range(len(row_tiles)):
                for jdx in range(idx + 1, len(row_tiles)):
                    if row_tiles[idx][1] > row_tiles[jdx][1]:
                        conflicts += 2

        for col in range(board_size):
            col_tiles: list[tuple[int, int]] = []
            for row in range(board_size):
                tile = state[row * board_size + col]
                if tile == 0:
                    continue
                goal_row, goal_col = goal_positions[tile]
                if goal_col == col:
                    col_tiles.append((row, goal_row))
            for idx in range(len(col_tiles)):
                for jdx in range(idx + 1, len(col_tiles)):
                    if col_tiles[idx][1] > col_tiles[jdx][1]:
                        conflicts += 2
        return conflicts

    def _state_metrics(
        self,
        state: State,
        goal: State,
        solution_moves: list[Move] | None,
        board_size: int,
    ) -> dict[str, int | float]:
        misplaced_tiles = sum(1 for idx, tile in enumerate(state) if tile != 0 and tile != goal[idx])
        correct_tiles = sum(1 for idx, tile in enumerate(state) if tile != 0 and tile == goal[idx])
        blank_index = state.index(0)
        blank_row, blank_col = divmod(blank_index, board_size)
        return {
            "optimal_solution_length": 0 if solution_moves is None else len(solution_moves),
            "manhattan_distance_sum": self._manhattan_sum(state, board_size),
            "misplaced_tile_count": misplaced_tiles,
            "tiles_already_correct": correct_tiles,
            "blank_row": blank_row,
            "blank_col": blank_col,
        }

    def _accept_non_solution_metrics(self, metrics: dict[str, int | float], bucket: BucketSpec) -> bool:
        if int(metrics["manhattan_distance_sum"]) < bucket.min_manhattan_sum:
            return False
        if int(metrics["misplaced_tile_count"]) < bucket.min_misplaced_tiles:
            return False
        if int(metrics["tiles_already_correct"]) > bucket.max_correct_tiles:
            return False
        return True

    def _accept_metrics(self, metrics: dict[str, int | float], bucket: BucketSpec) -> bool:
        optimal_length = int(metrics["optimal_solution_length"])
        if optimal_length < bucket.optimal_min or optimal_length > bucket.optimal_max:
            return False
        if not self._accept_non_solution_metrics(metrics, bucket):
            return False
        return True

    @classmethod
    def _blank_neighbors(cls, board_size: int) -> tuple[tuple[int, ...], ...]:
        cached = cls._blank_neighbors_cache.get(board_size)
        if cached is not None:
            return cached
        neighbors: list[tuple[int, ...]] = []
        for blank_index in range(board_size * board_size):
            row, col = divmod(blank_index, board_size)
            items: list[int] = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    items.append(nr * board_size + nc)
            neighbors.append(tuple(items))
        cls._blank_neighbors_cache[board_size] = tuple(neighbors)
        return cls._blank_neighbors_cache[board_size]

    @classmethod
    def _goal_positions(cls, board_size: int) -> tuple[tuple[int, int], ...]:
        cached = cls._goal_positions_cache.get(board_size)
        if cached is not None:
            return cached
        total = board_size * board_size
        positions = [(board_size - 1, board_size - 1)] * total
        for tile in range(1, total):
            positions[tile] = divmod(tile - 1, board_size)
        cls._goal_positions_cache[board_size] = tuple(positions)
        return cls._goal_positions_cache[board_size]

    def _far_enough(self, state: State, kept_states: list[State], min_state_distance: int) -> bool:
        if min_state_distance <= 0:
            return True
        for other in kept_states:
            if self._state_distance(state, other) < min_state_distance:
                return False
        return True

    def _state_distance(self, state_a: State, state_b: State) -> int:
        return sum(1 for left, right in zip(state_a, state_b) if left != right)

    def _frame_count(self, move_count: int) -> int:
        if move_count <= 0:
            return self.start_hold_frames + self.end_hold_frames + 1
        hold_total = self.start_hold_frames + self.end_hold_frames
        unclipped = move_count * self.base_frames_per_move + hold_total
        target_total = int(np.clip(unclipped, self.min_frames, self.max_frames))
        available_motion_frames = max(target_total - hold_total, move_count * self.min_frames_per_move)
        per_move = max(
            self.min_frames_per_move,
            min(self.max_frames_per_move, available_motion_frames // max(move_count, 1)),
        )
        return hold_total + move_count * per_move

    def _render_sample(self, split_dir: Path, sample: PuzzleSpec) -> dict[str, str]:
        video_path = split_dir / f"{sample.sample_id}.mp4"
        metadata_path = split_dir / f"{sample.sample_id}.json"

        frames = self._render_frames(sample)
        self._write_video(video_path, frames, sample.fps)

        metadata = self._build_metadata(sample, video_path, metadata_path)
        self.write_metadata(metadata_path, metadata)
        return {"video": video_path.name, "metadata": metadata_path.name}

    def _write_video(self, video_path: Path, frames: list[np.ndarray], fps: int) -> None:
        if importlib.util.find_spec("imageio_ffmpeg") is None:
            raise RuntimeError(
                "MP4 output requires the 'imageio-ffmpeg' package in the active environment. "
                "Install it with `python -m pip install imageio-ffmpeg` or "
                "`conda install -c conda-forge imageio-ffmpeg`, then rerun the dataset generator."
            )
        try:
            imageio.mimwrite(video_path, frames, fps=fps)
        except ValueError as exc:
            raise RuntimeError(
                f"Failed to write MP4 dataset artifact to {video_path}. "
                "Ensure 'imageio-ffmpeg' is installed in the active environment."
            ) from exc

    def _build_metadata(self, sample: PuzzleSpec, video_path: Path, metadata_path: Path) -> dict[str, Any]:
        blank_start = sample.initial_state.index(0)
        blank_goal = sample.goal_state.index(0)
        current_state = sample.initial_state
        move_payloads: list[dict[str, Any]] = []
        for move in sample.solution_moves:
            move_payloads.append(self._move_to_payload(current_state, move, sample.board_size))
            board = list(current_state)
            from_index, to_index = move
            board[from_index], board[to_index] = board[to_index], board[from_index]
            current_state = tuple(board)
        return {
            "task": self.task_name,
            "prompt": PROMPT,
            "sample_id": sample.sample_id,
            "difficulty_level": sample.difficulty_level,
            "split": sample.split,
            "topology_split": sample.topology_split,
            "bucket": sample.bucket_name,
            "family": sample.family,
            "puzzle": {
                "board_size": [sample.board_size, sample.board_size],
                "initial_board": self._matrix_from_state(sample.initial_state, sample.board_size),
                "goal_board": self._matrix_from_state(sample.goal_state, sample.board_size),
                "blank_start": [blank_start // sample.board_size, blank_start % sample.board_size],
                "blank_goal": [blank_goal // sample.board_size, blank_goal % sample.board_size],
            },
            "solution": {
                "moves": move_payloads,
                "move_count": len(sample.solution_moves),
                "shortest_solution": True,
            },
            "metrics": sample.metrics,
            "render": {
                "resolution": list(sample.resolution),
                "fps": sample.fps,
                "frame_count": sample.frame_count,
            },
            "artifacts": {
                "video": video_path.name,
                "metadata": metadata_path.name,
            },
        }

    def _matrix_from_state(self, state: State, board_size: int) -> list[list[int]]:
        return [list(state[row * board_size : (row + 1) * board_size]) for row in range(board_size)]

    def _move_to_payload(self, state: State, move: Move, board_size: int) -> dict[str, Any]:
        from_index, to_index = move
        return {
            "tile": int(state[from_index]),
            "from": [from_index // board_size, from_index % board_size],
            "to": [to_index // board_size, to_index % board_size],
        }

    def _render_frames(self, sample: PuzzleSpec) -> list[np.ndarray]:
        move_count = len(sample.solution_moves)
        hold_total = self.start_hold_frames + self.end_hold_frames
        per_move = (
            0
            if move_count == 0
            else max(
                self.min_frames_per_move,
                min(
                    self.max_frames_per_move,
                    max(sample.frame_count - hold_total, move_count * self.min_frames_per_move) // max(move_count, 1),
                ),
            )
        )
        geometry = self._board_geometry(sample.board_size)
        frames: list[np.ndarray] = []
        current_state = sample.initial_state

        initial_frame = self._render_board(sample, current_state, geometry)
        frames.extend([initial_frame] * self.start_hold_frames)
        for move in sample.solution_moves:
            step_frames, current_state = self._render_move_frames(sample, current_state, move, geometry, per_move)
            frames.extend(step_frames)
        final_frame = self._render_board(sample, current_state, geometry)
        frames.extend([final_frame] * self.end_hold_frames)
        return frames

    def _board_geometry(self, board_size: int) -> dict[str, float]:
        width, height = self.resolution
        outer_margin_default = 4 if board_size == 3 else 8
        panel_padding_default = 4 if board_size == 3 else 8
        slot_gap_default = 3 if board_size == 3 else 5

        outer_margin = float(self.style.get("outer_margin", outer_margin_default))
        panel_radius = float(self.style.get("panel_radius", 16))
        slot_gap = float(self.style.get("slot_gap", slot_gap_default))
        inner_padding = float(self.style.get("panel_padding", panel_padding_default))

        panel_w = width - 2 * outer_margin
        panel_h = height - 2 * outer_margin
        available = min(panel_w - 2 * inner_padding, panel_h - 2 * inner_padding)
        tile_size = (available - (board_size - 1) * slot_gap) / max(board_size, 1)
        board_extent = board_size * tile_size + (board_size - 1) * slot_gap
        origin_x = (width - board_extent) / 2
        origin_y = (height - board_extent) / 2

        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "tile_size": tile_size,
            "slot_gap": slot_gap,
            "panel_radius": panel_radius,
            "outer_margin": outer_margin,
        }

    def _render_move_frames(
        self,
        sample: PuzzleSpec,
        current_state: State,
        move: Move,
        geometry: dict[str, float],
        frames_per_move: int,
    ) -> tuple[list[np.ndarray], State]:
        from_index, to_index = move
        board = list(current_state)
        moving_tile = board[from_index]
        board[from_index], board[to_index] = board[to_index], board[from_index]
        next_state = tuple(board)

        from_rect = self._tile_rect(from_index, sample.board_size, geometry)
        to_rect = self._tile_rect(to_index, sample.board_size, geometry)
        frames: list[np.ndarray] = []
        for frame_idx in range(1, max(frames_per_move, 1) + 1):
            progress = frame_idx / max(frames_per_move, 1)
            moving_rect = (
                from_rect[0] + (to_rect[0] - from_rect[0]) * progress,
                from_rect[1] + (to_rect[1] - from_rect[1]) * progress,
                from_rect[2] + (to_rect[2] - from_rect[2]) * progress,
                from_rect[3] + (to_rect[3] - from_rect[3]) * progress,
            )
            frames.append(
                self._render_board(
                    sample,
                    current_state,
                    geometry,
                    moving_tile=(moving_tile, from_index, moving_rect),
                )
            )
        return frames, next_state

    def _render_board(
        self,
        sample: PuzzleSpec,
        state: State,
        geometry: dict[str, float],
        moving_tile: tuple[int, int, tuple[float, float, float, float]] | None = None,
    ) -> np.ndarray:
        width, height = sample.resolution
        colors = self._style_colors()
        frame = Image.new("RGBA", (width, height), colors["background"] + (255,))
        draw = ImageDraw.Draw(frame)

        outer_margin = geometry["outer_margin"]
        panel_rect = (outer_margin, outer_margin, width - outer_margin, height - outer_margin)
        draw.rounded_rectangle(panel_rect, radius=geometry["panel_radius"], fill=colors["panel"] + (255,))

        moving_from_index = moving_tile[1] if moving_tile is not None else None
        tile_size = geometry["tile_size"]
        slot_radius = max(6, int(tile_size * 0.12))
        tile_radius = max(7, int(tile_size * 0.12))

        for index, tile in enumerate(state):
            rect = self._tile_rect(index, sample.board_size, geometry)
            if tile == 0 or index == moving_from_index:
                draw.rounded_rectangle(rect, radius=slot_radius, fill=colors["slot"] + (255,))
                continue
            self._draw_tile(draw, tile, rect, tile_radius, colors, sample.board_size)

        if moving_tile is not None:
            tile, _, rect = moving_tile
            self._draw_tile(draw, tile, rect, tile_radius, colors, sample.board_size)

        return np.asarray(frame.convert("RGB"))

    def _draw_tile(
        self,
        draw: ImageDraw.ImageDraw,
        tile: int,
        rect: tuple[float, float, float, float],
        radius: int,
        colors: dict[str, tuple[int, int, int]],
        board_size: int,
    ) -> None:
        draw.rounded_rectangle(rect, radius=radius, fill=colors["tile"] + (255,), outline=colors["tile_border"] + (255,), width=2)
        glyph = self._render_scaled_glyph(str(tile), rect, colors["text"], board_size)
        glyph_x = int(round((rect[0] + rect[2] - glyph.width) / 2))
        glyph_y = int(round((rect[1] + rect[3] - glyph.height) / 2))
        draw._image.paste(glyph, (glyph_x, glyph_y), glyph)

    def _render_scaled_glyph(
        self,
        text: str,
        rect: tuple[float, float, float, float],
        fill_color: tuple[int, int, int],
        board_size: int,
    ) -> Image.Image:
        tile_w = max(1, int(round(rect[2] - rect[0])))
        tile_h = max(1, int(round(rect[3] - rect[1])))
        target_h = max(1, int(round(tile_h * 0.8)))
        target_w_limit = max(1, int(round(tile_w * 0.8)))

        font = self._pick_base_font(board_size, max(tile_w, tile_h) * 2)
        canvas_size = max(tile_w, tile_h) * 4
        mask = Image.new("L", (canvas_size, canvas_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        bbox = mask_draw.textbbox((0, 0), text, font=font, anchor="lt")
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        origin_x = (canvas_size - text_w) // 2 - bbox[0]
        origin_y = (canvas_size - text_h) // 2 - bbox[1]
        mask_draw.text((origin_x, origin_y), text, font=font, fill=255)

        crop_box = mask.getbbox()
        if crop_box is None:
            return Image.new("RGBA", (target_w_limit, target_h), fill_color + (0,))

        glyph_mask = mask.crop(crop_box)
        glyph_w, glyph_h = glyph_mask.size
        scaled_w = max(1, int(round(glyph_w * (target_h / max(glyph_h, 1)))))
        scaled_h = target_h
        if scaled_w > target_w_limit:
            scale = target_w_limit / max(scaled_w, 1)
            scaled_w = target_w_limit
            scaled_h = max(1, int(round(scaled_h * scale)))

        glyph_mask = glyph_mask.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        glyph_mask = glyph_mask.point(lambda value: 255 if value >= 96 else 0, mode="L")

        glyph = Image.new("RGBA", (scaled_w, scaled_h), fill_color + (0,))
        glyph.putalpha(glyph_mask)
        return glyph

    def _pick_base_font(
        self,
        board_size: int,
        size: int,
    ) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        font_candidates = (
            "DejaVuSans-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
            "DejaVuSans.ttf",
        )
        if board_size >= 4:
            font_candidates = (
                "DejaVuSansCondensed-Bold.ttf",
                "DejaVuSans-Bold.ttf",
                "DejaVuSans.ttf",
            )
        for font_name in font_candidates:
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _tile_rect(
        self,
        index: int,
        board_size: int,
        geometry: dict[str, float],
    ) -> tuple[float, float, float, float]:
        row, col = divmod(index, board_size)
        tile_size = geometry["tile_size"]
        slot_gap = geometry["slot_gap"]
        x0 = geometry["origin_x"] + col * (tile_size + slot_gap)
        y0 = geometry["origin_y"] + row * (tile_size + slot_gap)
        return (x0, y0, x0 + tile_size, y0 + tile_size)

    def _style_colors(self) -> dict[str, tuple[int, int, int]]:
        def hex_to_rgb(value: str, default: str) -> tuple[int, int, int]:
            raw = value if value else default
            raw = raw.lstrip("#")
            return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))

        return {
            "background": hex_to_rgb(str(self.style.get("background", "#0e1118")), "#0e1118"),
            "panel": hex_to_rgb(str(self.style.get("panel", "#25272b")), "#25272b"),
            "slot": hex_to_rgb(str(self.style.get("slot", "#43485c")), "#43485c"),
            "tile": hex_to_rgb(str(self.style.get("tile", "#30426d")), "#30426d"),
            "tile_border": hex_to_rgb(str(self.style.get("tile_border", "#b8d1ff")), "#b8d1ff"),
            "text": hex_to_rgb(str(self.style.get("text", "#f1f4fb")), "#f1f4fb"),
            "text_stroke": hex_to_rgb(str(self.style.get("text_stroke", "#1c2230")), "#1c2230"),
        }


def _generate_split_task(task: dict[str, Any]) -> dict[str, Any]:
    env = SlidingPuzzleEnv(task["env_config"])
    entries = env.generate_split(
        output_root=task["output_root"],
        level=int(task["level"]),
        split=str(task["split"]),
        workers=int(task.get("workers", 1)),
    )
    return {"level": int(task["level"]), "split": str(task["split"]), "entries": entries}


def _generate_independent_sample_task(task: dict[str, Any]) -> dict[str, Any]:
    env = SlidingPuzzleEnv(task["env_config"])
    bucket = BucketSpec(**task["bucket"])
    puzzle = env._make_independent_puzzle_spec(
        level=int(task["level"]),
        split=str(task["split"]),
        index=int(task["index"]),
        bucket=bucket,
        seed_offset=int(task.get("seed_offset", 0)),
    )
    return {
        "sample_id": puzzle.sample_id,
        "difficulty_level": puzzle.difficulty_level,
        "split": puzzle.split,
        "topology_split": puzzle.topology_split,
        "bucket_name": puzzle.bucket_name,
        "family": puzzle.family,
        "board_size": puzzle.board_size,
        "initial_state": list(puzzle.initial_state),
        "goal_state": list(puzzle.goal_state),
        "solution_moves": [list(move) for move in puzzle.solution_moves],
        "frame_count": puzzle.frame_count,
        "fps": puzzle.fps,
        "resolution": list(puzzle.resolution),
        "metrics": dict(puzzle.metrics),
    }
