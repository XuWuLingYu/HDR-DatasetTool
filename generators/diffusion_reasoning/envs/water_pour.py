from __future__ import annotations

import importlib.util
import json
import math
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .base_env import BaseEnv

PROMPT = (
    "The scene shows a set of vertical tubes, containing discrete blocks of various colors. "
    "The sorting rule is: select a source tube and pour the contiguous top blocks of the same color "
    "into another tube. These blocks can only be poured into an empty tube or onto a top block of the "
    "matching color, and only as many blocks move as the destination tube has available capacity for. "
    "Observe the initial distribution of colored blocks and determine the sequence of transfers. "
    "Animate each poured block following the same trajectory with a staggered motion, traveling through "
    "open space and dropping into the target tube, ensuring strict conservation of blocks and ending in a "
    "state where each tube contains only blocks of a single color."
)

ColorId = int
Tube = tuple[ColorId, ...]
State = tuple[Tube, ...]
Move = tuple[int, int, int]


@dataclass(frozen=True)
class BucketSpec:
    name: str
    topology_split: str
    tube_count: int
    capacity: int
    color_count: int
    quota: int
    min_solution_len: int
    max_solution_len: int
    max_start_moves: int
    max_branch_avg_times_ten: int
    max_sorted_tubes: int
    max_retries: int
    min_state_distance: int


@dataclass(frozen=True)
class PuzzleSpec:
    sample_id: str
    difficulty_level: int
    split: str
    topology_split: str
    bucket_name: str
    tube_count: int
    capacity: int
    color_count: int
    initial_state: State
    goal_state: State
    solution_moves: tuple[Move, ...]
    shortest_solution: bool
    frame_count: int
    fps: int
    resolution: tuple[int, int]
    palette: tuple[str, ...]
    layout_variant: str
    metrics: dict[str, int | float]


class WaterPourEnv(BaseEnv):
    task_name = "water_pour"

    def __init__(self, env_config: dict[str, Any]) -> None:
        super().__init__(env_config)
        global_config = env_config.get("global", {})
        render = env_config.get("render", {})
        generation = env_config.get("generation", {})
        self.output_root = Path(env_config.get("output_root", global_config.get("default_output_root", "data")))
        self.resolution = tuple(render.get("resolution", [208, 208]))
        self.fps = int(render.get("fps", 10))
        self.frames_per_lift = int(render.get("frames_per_lift", 3))
        self.frames_per_horizontal = int(render.get("frames_per_horizontal", 4))
        self.frames_per_drop = int(render.get("frames_per_drop", 3))
        self.hold_frames = int(render.get("hold_frames", 2))
        self.levels = env_config.get("levels", {})
        self.style = render.get("style", {})
        self.palettes = [tuple(palette) for palette in render.get("palettes", [])]
        self.global_seed = int(env_config.get("seed", 0))
        self.default_workers = int(generation.get("workers", max(1, min(6, os.cpu_count() or 1))))

    def generate_dataset(
        self,
        output_root: str | Path | None = None,
        levels: list[int] | None = None,
        workers: int | None = None,
    ) -> None:
        resolved_root = Path(output_root) if output_root is not None else self.output_root
        target_levels = levels or sorted(int(level) for level in self.levels.keys())
        requested_workers = self.default_workers if workers is None else max(1, int(workers))
        tasks = [
            {
                "env_config": self.config,
                "output_root": str(resolved_root),
                "level": int(level),
                "split": split,
            }
            for level in target_levels
            for split in ("train", "eval")
        ]
        worker_count = min(requested_workers, len(tasks))
        manifests: dict[int, dict[str, list[dict[str, Any]]]] = {
            int(level): {"train": [], "eval": []} for level in target_levels
        }

        if worker_count == 1:
            for task in tasks:
                result = _generate_split_task(task)
                manifests[result["level"]][result["split"]] = result["entries"]
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for result in executor.map(_generate_split_task, tasks):
                    manifests[result["level"]][result["split"]] = result["entries"]

        for level in target_levels:
            difficulty_dir = resolved_root / f"water_pour_{level}level"
            for split, entries in manifests[int(level)].items():
                manifest_path = difficulty_dir / f"{split}_manifest.json"
                with manifest_path.open("w", encoding="utf-8") as handle:
                    json.dump(entries, handle, indent=2, ensure_ascii=False)

    def generate_split(self, output_root: str | Path, level: int, split: str) -> list[dict[str, Any]]:
        resolved_root = Path(output_root)
        level_key = str(level) if str(level) in self.levels else level
        level_config = self.levels[level_key]
        difficulty_dir = resolved_root / f"water_pour_{level}level"
        split_dir = self.ensure_split_dir(difficulty_dir, split)
        bucket_specs = self._build_bucket_schedule(level_config, split)
        calibration = self._calibrate_buckets(bucket_specs)
        seen_states: set[State] = set()
        kept_samples: list[PuzzleSpec] = []
        kept_state_vectors: list[tuple[State, dict[str, int | float]]] = []
        entries: list[dict[str, Any]] = []

        for index, bucket in enumerate(bucket_specs):
            puzzle = self._make_puzzle_spec(
                level=level,
                split=split,
                index=index,
                bucket=bucket,
                calibration=calibration.get(bucket.name, {}),
                seen_states=seen_states,
                kept_state_vectors=kept_state_vectors,
            )
            kept_samples.append(puzzle)
            kept_state_vectors.append((puzzle.initial_state, puzzle.metrics))
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
        return entries

    def _build_bucket_schedule(self, level_config: dict[str, Any], split: str) -> list[BucketSpec]:
        if split == "train":
            bucket_dicts = list(level_config.get("train_buckets", []))
            total = int(level_config["counts"]["train"])
        else:
            total = int(level_config["counts"]["eval"])
            ood_ratio = float(level_config.get("eval_ood_ratio", 0.4))
            ood_total = min(total, round(total * ood_ratio))
            id_total = total - ood_total
            bucket_dicts = self._scale_bucket_dicts(level_config.get("eval_id_buckets", []), id_total)
            bucket_dicts.extend(self._scale_bucket_dicts(level_config.get("eval_ood_buckets", []), ood_total))
            return self._expand_bucket_schedule(bucket_dicts)

        if sum(int(bucket["quota"]) for bucket in bucket_dicts) != total:
            bucket_dicts = self._scale_bucket_dicts(bucket_dicts, total)
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
        weights = [float(bucket.get("quota", bucket.get("weight", 1))) for bucket in bucket_dicts]
        weight_sum = sum(weights)
        provisional = [int(math.floor(total * weight / max(weight_sum, 1.0))) for weight in weights]
        remainder = total - sum(provisional)
        order = sorted(range(len(bucket_dicts)), key=lambda idx: (weights[idx], -idx), reverse=True)
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
            tube_count=int(payload["tube_count"]),
            capacity=int(payload["capacity"]),
            color_count=int(payload["color_count"]),
            quota=int(payload["quota"]),
            min_solution_len=int(payload["solution_len_range"][0]),
            max_solution_len=int(payload["solution_len_range"][1]),
            max_start_moves=int(payload.get("max_start_moves", 12)),
            max_branch_avg_times_ten=int(payload.get("max_branch_avg_times_ten", 60)),
            max_sorted_tubes=int(payload.get("max_sorted_tubes", 1)),
            max_retries=int(payload.get("max_retries", 250)),
            min_state_distance=int(payload.get("min_state_distance", 2)),
        )

    def _calibrate_buckets(self, buckets: list[BucketSpec]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for bucket in buckets:
            key = f"{bucket.tube_count}:{bucket.capacity}:{bucket.color_count}"
            if key in grouped:
                continue
            grouped[key] = {
                "tube_count": bucket.tube_count,
                "capacity": bucket.capacity,
                "color_count": bucket.color_count,
                "estimate": self._estimate_bucket_space(bucket),
            }
        return {
            bucket.name: grouped[f"{bucket.tube_count}:{bucket.capacity}:{bucket.color_count}"] for bucket in buckets
        }

    def _estimate_bucket_space(self, bucket: BucketSpec) -> int:
        if bucket.tube_count <= 6 and bucket.capacity <= 4:
            goal_state = self._goal_state(bucket.tube_count, bucket.capacity, bucket.color_count)
            seen = {self._canonicalize_state(goal_state)}
            frontier = deque([goal_state])
            while frontier and len(seen) < 25000:
                state = frontier.popleft()
                for prev_state, _ in self._backward_moves(state, bucket.capacity):
                    canonical = self._canonicalize_state(prev_state)
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    frontier.append(prev_state)
            return len(seen)
        total_slots = bucket.tube_count * bucket.capacity
        empties = total_slots - bucket.color_count * bucket.capacity
        placements = bucket.tube_count ** max(bucket.color_count * bucket.capacity, 1)
        return max(1, placements // math.factorial(max(empties, 1)))

    def _make_puzzle_spec(
        self,
        level: int,
        split: str,
        index: int,
        bucket: BucketSpec,
        calibration: dict[str, Any],
        seen_states: set[State],
        kept_state_vectors: list[tuple[State, dict[str, int | float]]],
    ) -> PuzzleSpec:
        if int(calibration.get("estimate", 0)) < max(bucket.quota * 2, 20):
            raise RuntimeError(
                f"Bucket {bucket.name} appears undersized for requested quota {bucket.quota}: "
                f"estimated unique states={calibration.get('estimate', 0)}"
            )

        base_seed = self._sample_seed(level, split, index)
        for attempt in range(bucket.max_retries):
            rng = np.random.default_rng(base_seed + attempt * 9_973)
            goal_state = self._goal_state(bucket.tube_count, bucket.capacity, bucket.color_count)
            walk_length = int(rng.integers(bucket.min_solution_len, bucket.max_solution_len + 1))
            candidate, forward_hint = self._sample_backward_chain(goal_state, bucket, walk_length, rng)
            if candidate is None or forward_hint is None:
                continue
            if candidate == goal_state:
                continue

            canonical_state = self._canonicalize_state(candidate)
            if canonical_state in seen_states:
                continue

            solved = self._select_solution(candidate, bucket, forward_hint)
            if solved is None:
                continue
            solution_moves, goal_state, shortest_solution = solved

            metrics = self._state_metrics(candidate, solution_moves, bucket.capacity)
            if not self._accept_metrics(metrics, bucket):
                continue
            if not self._far_enough(candidate, metrics, kept_state_vectors, bucket.min_state_distance):
                continue

            layout_variant = "two_row" if bucket.tube_count >= 8 else "single_row"
            seen_states.add(canonical_state)
            frame_count = self._frame_count(solution_moves, layout_variant)
            palette = self.palettes[(base_seed + attempt) % len(self.palettes)] if self.palettes else ("#ff6f91", "#5d9dff")
            return PuzzleSpec(
                sample_id=f"water_pour_l{level}_{split}_{index:04d}",
                difficulty_level=level,
                split=split,
                topology_split=bucket.topology_split,
                bucket_name=bucket.name,
                tube_count=bucket.tube_count,
                capacity=bucket.capacity,
                color_count=bucket.color_count,
                initial_state=candidate,
                goal_state=goal_state,
                solution_moves=tuple(solution_moves),
                shortest_solution=shortest_solution,
                frame_count=frame_count,
                fps=self.fps,
                resolution=self.resolution,
                palette=palette,
                layout_variant=layout_variant,
                metrics=metrics,
            )

        raise RuntimeError(f"Failed to generate a unique water pour puzzle for bucket {bucket.name}")

    def _select_solution(
        self,
        candidate: State,
        bucket: BucketSpec,
        forward_hint: list[Move],
    ) -> tuple[list[Move], State, bool] | None:
        goal_state = self._goal_state(bucket.tube_count, bucket.capacity, bucket.color_count)
        if bucket.tube_count > 7 or bucket.capacity > 4:
            validated_goal = self._validate_solution_path(candidate, forward_hint, bucket.capacity)
            if validated_goal == goal_state:
                return list(forward_hint), goal_state, False
        solution_moves = self._solve_shortest(candidate, goal_state, bucket)
        if solution_moves is None:
            return None
        return solution_moves, goal_state, True

    def _sample_seed(self, level: int, split: str, index: int) -> int:
        split_offset = 0 if split == "train" else 50_000
        return self.global_seed + level * 10_000 + split_offset + index

    def _goal_state(self, tube_count: int, capacity: int, color_count: int) -> State:
        filled = [tuple([color] * capacity) for color in range(color_count)]
        empties = [tuple() for _ in range(tube_count - color_count)]
        return tuple(filled + empties)

    def _sample_backward_chain(
        self,
        goal_state: State,
        bucket: BucketSpec,
        walk_length: int,
        rng: np.random.Generator,
    ) -> tuple[State | None, list[Move] | None]:
        current = goal_state
        visited = {current}
        backward_chain: list[Move] = []
        for _ in range(walk_length):
            candidates = list(self._backward_moves(current, bucket.capacity))
            if not candidates:
                return None, None
            rng.shuffle(candidates)
            picked = None
            for prev_state, move in candidates:
                if prev_state in visited:
                    continue
                picked = (prev_state, move)
                break
            if picked is None:
                picked = candidates[0]
            current, move = picked
            visited.add(current)
            backward_chain.append(move)
        return current, list(reversed(backward_chain))

    def _solve_shortest(self, start: State, goal: State, bucket: BucketSpec) -> list[Move] | None:
        if start == goal:
            return []

        forward_queue = deque([start])
        backward_queue = deque([goal])
        forward_parent: dict[State, tuple[State | None, Move | None]] = {start: (None, None)}
        backward_parent: dict[State, tuple[State | None, Move | None]] = {goal: (None, None)}
        max_nodes = int(self.config.get("solver", {}).get("max_nodes", 60000))

        while forward_queue and backward_queue:
            if len(forward_parent) + len(backward_parent) > max_nodes:
                return None
            if len(forward_queue) <= len(backward_queue):
                meeting = self._expand_forward_layer(
                    queue=forward_queue,
                    own_parent=forward_parent,
                    other_parent=backward_parent,
                    capacity=bucket.capacity,
                )
            else:
                meeting = self._expand_backward_layer(
                    queue=backward_queue,
                    own_parent=backward_parent,
                    other_parent=forward_parent,
                    capacity=bucket.capacity,
                )
            if meeting is not None:
                return self._reconstruct_path(meeting, forward_parent, backward_parent)
        return None

    def _solve_shortest_to_solved(self, start: State, bucket: BucketSpec) -> tuple[list[Move], State] | None:
        if self._is_solved_state(start, bucket.capacity):
            return [], start

        queue = deque([start])
        parent: dict[State, tuple[State | None, Move | None]] = {start: (None, None)}
        max_nodes = int(self.config.get("solver", {}).get("max_nodes", 60000))

        while queue:
            if len(parent) > max_nodes:
                return None
            state = queue.popleft()
            for next_state, move in self._forward_moves(state, bucket.capacity):
                if next_state in parent:
                    continue
                parent[next_state] = (state, move)
                if self._is_solved_state(next_state, bucket.capacity):
                    return self._reconstruct_forward_path(next_state, parent), next_state
                queue.append(next_state)
        return None

    def _expand_forward_layer(
        self,
        queue: deque[State],
        own_parent: dict[State, tuple[State | None, Move | None]],
        other_parent: dict[State, tuple[State | None, Move | None]],
        capacity: int,
    ) -> State | None:
        layer_size = len(queue)
        for _ in range(layer_size):
            state = queue.popleft()
            for next_state, move in self._forward_moves(state, capacity):
                if next_state in own_parent:
                    continue
                own_parent[next_state] = (state, move)
                if next_state in other_parent:
                    return next_state
                queue.append(next_state)
        return None

    def _expand_backward_layer(
        self,
        queue: deque[State],
        own_parent: dict[State, tuple[State | None, Move | None]],
        other_parent: dict[State, tuple[State | None, Move | None]],
        capacity: int,
    ) -> State | None:
        layer_size = len(queue)
        for _ in range(layer_size):
            state = queue.popleft()
            for prev_state, move in self._backward_moves(state, capacity):
                if prev_state in own_parent:
                    continue
                own_parent[prev_state] = (state, move)
                if prev_state in other_parent:
                    return prev_state
                queue.append(prev_state)
        return None

    def _reconstruct_path(
        self,
        meeting: State,
        forward_parent: dict[State, tuple[State | None, Move | None]],
        backward_parent: dict[State, tuple[State | None, Move | None]],
    ) -> list[Move]:
        prefix: list[Move] = []
        state = meeting
        while True:
            parent_state, move = forward_parent[state]
            if parent_state is None or move is None:
                break
            prefix.append(move)
            state = parent_state
        prefix.reverse()

        suffix: list[Move] = []
        state = meeting
        while True:
            next_state, move = backward_parent[state]
            if next_state is None or move is None:
                break
            suffix.append(move)
            state = next_state
        return prefix + suffix

    def _reconstruct_forward_path(
        self,
        goal: State,
        parent: dict[State, tuple[State | None, Move | None]],
    ) -> list[Move]:
        path: list[Move] = []
        state = goal
        while True:
            prev_state, move = parent[state]
            if prev_state is None or move is None:
                break
            path.append(move)
            state = prev_state
        path.reverse()
        return path

    def _forward_moves(self, state: State, capacity: int) -> list[tuple[State, Move]]:
        moves: list[tuple[State, Move]] = []
        for source_idx, source in enumerate(state):
            if not source:
                continue
            block = source[-1]
            run_length = self._top_run_length(source)
            for dest_idx, dest in enumerate(state):
                if source_idx == dest_idx or len(dest) >= capacity:
                    continue
                if dest and dest[-1] != block:
                    continue
                moved_count = min(run_length, capacity - len(dest))
                if moved_count <= 0:
                    continue
                next_state = [list(tube) for tube in state]
                moved_blocks = next_state[source_idx][-moved_count:]
                del next_state[source_idx][-moved_count:]
                next_state[dest_idx].extend(moved_blocks)
                moves.append((tuple(tuple(tube) for tube in next_state), (source_idx, dest_idx, moved_count)))
        return moves

    def _is_legal_move(self, state: State, move: Move, capacity: int) -> bool:
        source_idx, dest_idx, moved_count = move
        if source_idx == dest_idx or moved_count <= 0:
            return False
        if source_idx < 0 or dest_idx < 0 or source_idx >= len(state) or dest_idx >= len(state):
            return False

        source = state[source_idx]
        dest = state[dest_idx]
        if not source or len(dest) >= capacity:
            return False

        movable_count = min(self._top_run_length(source), capacity - len(dest))
        if moved_count != movable_count or moved_count > len(source):
            return False

        moved_blocks = source[-moved_count:]
        moved_color = moved_blocks[0]
        if any(block != moved_color for block in moved_blocks):
            return False
        if dest and dest[-1] != moved_color:
            return False
        return True

    def _validate_solution_path(self, start: State, moves: list[Move] | tuple[Move, ...], capacity: int) -> State | None:
        state = start
        for move in moves:
            if not self._is_legal_move(state, move, capacity):
                return None
            state = self._apply_move(state, move, capacity)
        if not self._is_solved_state(state, capacity):
            return None
        return state

    def _backward_moves(self, state: State, capacity: int) -> list[tuple[State, Move]]:
        moves: list[tuple[State, Move]] = []
        for dest_idx, dest in enumerate(state):
            if not dest:
                continue
            block = dest[-1]
            dest_run_length = self._top_run_length(dest)
            for source_idx, source in enumerate(state):
                if source_idx == dest_idx or len(source) >= capacity:
                    continue
                max_moved = min(dest_run_length, capacity - len(source))
                for moved_count in range(1, max_moved + 1):
                    prev_state = [list(tube) for tube in state]
                    moved_blocks = prev_state[dest_idx][-moved_count:]
                    del prev_state[dest_idx][-moved_count:]
                    if prev_state[dest_idx] and prev_state[dest_idx][-1] != block:
                        continue
                    prev_state[source_idx].extend(moved_blocks)
                    previous = tuple(tuple(tube) for tube in prev_state)
                    move = (source_idx, dest_idx, moved_count)
                    if not self._is_legal_move(previous, move, capacity):
                        continue
                    moves.append((previous, move))
        return moves

    def _canonicalize_state(self, state: State) -> State:
        color_map: dict[ColorId, ColorId] = {}
        next_color = 0
        normalized_tubes: list[Tube] = []
        for tube in state:
            normalized: list[int] = []
            for color in tube:
                if color not in color_map:
                    color_map[color] = next_color
                    next_color += 1
                normalized.append(color_map[color])
            normalized_tubes.append(tuple(normalized))
        return tuple(sorted(normalized_tubes))

    def _state_metrics(self, state: State, solution_moves: list[Move], capacity: int) -> dict[str, int | float]:
        move_options = len(self._forward_moves(state, capacity))
        buried_blocks = 0
        color_runs = 0
        sorted_tubes = 0
        fill_counts: list[int] = []
        for tube in state:
            fill_counts.append(len(tube))
            if not tube:
                continue
            if len(set(tube)) == 1 and len(tube) == capacity:
                sorted_tubes += 1
            previous = None
            for block in tube:
                if previous != block:
                    color_runs += 1
                previous = block
            top = tube[-1]
            buried_blocks += sum(1 for block in tube[:-1] if block != top)

        path_branch_sum = 0
        state_cursor = state
        for move in solution_moves:
            path_branch_sum += len(self._forward_moves(state_cursor, capacity))
            state_cursor = self._apply_move(state_cursor, move, capacity)

        branch_avg = path_branch_sum / max(len(solution_moves), 1)
        fill_mean = sum(fill_counts) / max(len(fill_counts), 1)
        variance = sum((count - fill_mean) ** 2 for count in fill_counts) / max(len(fill_counts), 1)
        return {
            "solution_length": len(solution_moves),
            "start_legal_moves": move_options,
            "avg_branching": round(branch_avg, 2),
            "buried_blocks": buried_blocks,
            "color_runs": color_runs,
            "sorted_tubes": sorted_tubes,
            "fill_variance": round(variance, 2),
        }

    def _accept_metrics(self, metrics: dict[str, int | float], bucket: BucketSpec) -> bool:
        if int(metrics["solution_length"]) < bucket.min_solution_len:
            return False
        if int(metrics["solution_length"]) > bucket.max_solution_len:
            return False
        if int(metrics["start_legal_moves"]) > bucket.max_start_moves:
            return False
        if float(metrics["avg_branching"]) * 10 > bucket.max_branch_avg_times_ten:
            return False
        if int(metrics["sorted_tubes"]) > bucket.max_sorted_tubes:
            return False
        return True

    def _far_enough(
        self,
        candidate: State,
        metrics: dict[str, int | float],
        kept_state_vectors: list[tuple[State, dict[str, int | float]]],
        min_distance: int,
    ) -> bool:
        if not kept_state_vectors:
            return True
        window = kept_state_vectors[-96:]
        for existing_state, existing_metrics in window:
            if self._state_distance(candidate, existing_state, metrics, existing_metrics) < min_distance:
                return False
        return True

    def _state_distance(
        self,
        left: State,
        right: State,
        left_metrics: dict[str, int | float],
        right_metrics: dict[str, int | float],
    ) -> int:
        left_norm = self._canonicalize_state(left)
        right_norm = self._canonicalize_state(right)
        mismatch = 0
        for left_tube, right_tube in zip(left_norm, right_norm):
            pad = max(len(left_tube), len(right_tube))
            mismatch += sum(
                1
                for idx in range(pad)
                if (left_tube[idx] if idx < len(left_tube) else -1) != (right_tube[idx] if idx < len(right_tube) else -1)
            )
        metric_gap = abs(int(left_metrics["solution_length"]) - int(right_metrics["solution_length"]))
        metric_gap += abs(int(left_metrics["buried_blocks"]) - int(right_metrics["buried_blocks"]))
        return mismatch + metric_gap

    def _apply_move(self, state: State, move: Move, capacity: int) -> State:
        if not self._is_legal_move(state, move, capacity):
            raise ValueError(f"Illegal water-pour move {move} for state {state}")
        source_idx, dest_idx, moved_count = move
        next_state = [list(tube) for tube in state]
        moved_blocks = next_state[source_idx][-moved_count:]
        del next_state[source_idx][-moved_count:]
        next_state[dest_idx].extend(moved_blocks)
        return tuple(tuple(tube) for tube in next_state)

    def _apply_moves(self, state: State, moves: list[Move], capacity: int) -> State:
        current = state
        for move in moves:
            current = self._apply_move(current, move, capacity)
        return current

    def _frame_count(self, solution_moves: tuple[Move, ...] | list[Move], layout_variant: str) -> int:
        total = sum(self._move_frame_count(move[2], layout_variant) for move in solution_moves)
        return 1 + total + self.hold_frames

    def _move_frame_count(self, moved_count: int, layout_variant: str) -> int:
        if layout_variant == "two_row":
            base_frames = (
                self.frames_per_lift
                + int(self.style.get("frames_per_rise", 2))
                + self.frames_per_horizontal
                + int(self.style.get("frames_per_pre_drop", 2))
                + self.frames_per_drop
            )
        else:
            base_frames = self.frames_per_lift + self.frames_per_horizontal + self.frames_per_drop
        return base_frames + self._follow_delay_frames() * max(moved_count - 1, 0)

    def _follow_delay_frames(self) -> int:
        # A larger default delay keeps multi-block pours legible near the destination rim.
        return max(1, int(self.style.get("moving_follow_delay_frames", 2)))

    def _render_sample(self, split_dir: Path, sample: PuzzleSpec) -> dict[str, str]:
        video_path = split_dir / f"{sample.sample_id}.mp4"
        metadata_path = split_dir / f"{sample.sample_id}.json"
        validated_goal = self._validate_solution_path(sample.initial_state, sample.solution_moves, sample.capacity)
        if validated_goal is None:
            raise RuntimeError(f"Refusing to render invalid solution path for {sample.sample_id}")
        if validated_goal != sample.goal_state:
            raise RuntimeError(
                f"Validated goal state mismatch for {sample.sample_id}: expected {sample.goal_state}, got {validated_goal}"
            )
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
        return {
            "task": self.task_name,
            "prompt": PROMPT,
            "sample_id": sample.sample_id,
            "difficulty_level": sample.difficulty_level,
            "split": sample.split,
            "topology_split": sample.topology_split,
            "bucket": sample.bucket_name,
            "puzzle": {
                "tube_count": sample.tube_count,
                "capacity": sample.capacity,
                "color_count": sample.color_count,
                "initial_tubes": [list(tube) for tube in sample.initial_state],
                "goal_tubes": [list(tube) for tube in sample.goal_state],
            },
            "solution": {
                "moves": [list(move) for move in sample.solution_moves],
                "move_count": len(sample.solution_moves),
                "shortest_solution": sample.shortest_solution,
            },
            "metrics": sample.metrics,
            "render": {
                "resolution": list(sample.resolution),
                "fps": sample.fps,
                "frame_count": sample.frame_count,
                "layout_variant": sample.layout_variant,
                "palette": list(sample.palette),
            },
            "artifacts": {
                "video": video_path.name,
                "metadata": metadata_path.name,
            },
        }

    def _render_frames(self, sample: PuzzleSpec) -> list[np.ndarray]:
        tubes_geometry = self._tube_layout(sample.tube_count, sample.layout_variant)
        frames: list[np.ndarray] = []
        current_state = sample.initial_state
        frames.append(self._render_board(sample, current_state, tubes_geometry, moving_blocks=None))
        for move in sample.solution_moves:
            step_frames, current_state = self._render_move_frames(sample, current_state, move, tubes_geometry)
            frames.extend(step_frames)
        final_frame = self._render_board(sample, current_state, tubes_geometry, moving_blocks=None)
        frames.extend([final_frame] * self.hold_frames)
        return frames

    def _tube_layout(self, tube_count: int, layout_variant: str) -> list[dict[str, float]]:
        width, height = self.resolution
        margin_x = float(self.style.get("margin_x", 16))
        margin_y = float(self.style.get("margin_y", 20))
        if layout_variant == "two_row":
            cols = math.ceil(tube_count / 2)
            rows = 2
        else:
            cols = tube_count
            rows = 1

        gap_x = float(self.style.get("tube_gap_x", 14))
        gap_y = float(self.style.get("tube_gap_y", 22))
        tube_width = min(
            float(self.style.get("tube_width", 34)),
            (width - 2 * margin_x - (cols - 1) * gap_x) / max(cols, 1),
        )
        default_tube_height = (
            float(self.style.get("tube_height_two_row", 72))
            if rows == 2
            else float(self.style.get("tube_height", 126))
        )
        top_reserve = float(self.style.get("two_row_carry_reserve", 26)) if rows == 2 else 0.0
        tube_height = min(
            default_tube_height,
            (height - 2 * margin_y - (rows - 1) * gap_y - top_reserve) / max(rows, 1),
        )

        layout: list[dict[str, float]] = []
        total_width = cols * tube_width + (cols - 1) * gap_x
        total_height = rows * tube_height + (rows - 1) * gap_y + top_reserve
        origin_x = (width - total_width) / 2
        origin_y = (height - total_height) / 2 + top_reserve
        for index in range(tube_count):
            row = index // cols
            col = index % cols
            x = origin_x + col * (tube_width + gap_x)
            y = origin_y + row * (tube_height + gap_y)
            layout.append(
                {
                    "x": x,
                    "y": y,
                    "width": tube_width,
                    "height": tube_height,
                }
            )
        return layout

    def _render_move_frames(
        self,
        sample: PuzzleSpec,
        state: State,
        move: Move,
        tubes_geometry: list[dict[str, float]],
    ) -> tuple[list[np.ndarray], State]:
        source_idx, dest_idx, moved_count = move
        source_tube = state[source_idx]
        moving_blocks = list(reversed(source_tube[-moved_count:]))
        source_geom = tubes_geometry[source_idx]
        dest_geom = tubes_geometry[dest_idx]
        block_width = self._block_width(source_geom["width"])
        block_height = self._block_height(source_geom["height"], sample.capacity)
        dest_level = len(state[dest_idx])
        rim_clearance = float(self.style.get("carry_clearance", 14))
        carry_top_margin = float(self.style.get("carry_top_margin", 6))
        start_x = source_geom["x"] + source_geom["width"] / 2
        end_x = dest_geom["x"] + dest_geom["width"] / 2
        source_above_y = max(
            block_height / 2 + carry_top_margin,
            source_geom["y"] - block_height / 2 - rim_clearance,
        )
        dest_above_y = max(
            block_height / 2 + carry_top_margin,
            dest_geom["y"] - block_height / 2 - rim_clearance,
        )
        corridor_y = self._carry_corridor_y(tubes_geometry, block_height)
        lane_x = None
        if sample.layout_variant == "two_row":
            lane_x = self._two_row_lane_x(source_geom, dest_geom, tubes_geometry)
            self._validate_two_row_carry_path(
                sample,
                block_height,
                source_above_y,
                corridor_y,
                dest_above_y,
                lane_x,
                tubes_geometry,
            )

        base_frames = self._move_frame_count(1, sample.layout_variant)
        follow_delay = self._follow_delay_frames()
        total_frames = self._move_frame_count(moved_count, sample.layout_variant)
        frames: list[np.ndarray] = []

        for global_frame in range(total_frames):
            departed_count = 0
            landed_count = 0
            airborne_blocks: list[dict[str, Any]] = []
            for order_idx, color in enumerate(moving_blocks):
                local_frame = global_frame - order_idx * follow_delay
                if local_frame < 0:
                    continue
                departed_count += 1
                if local_frame >= base_frames - 1:
                    landed_count += 1
                    continue
                source_level = len(source_tube) - 1 - order_idx
                end_level = dest_level + order_idx
                center = self._trajectory_center_for_block(
                    sample=sample,
                    source_geom=source_geom,
                    dest_geom=dest_geom,
                    source_level=source_level,
                    dest_level=end_level,
                    local_frame=local_frame,
                    start_x=start_x,
                    end_x=end_x,
                    source_above_y=source_above_y,
                    dest_above_y=dest_above_y,
                    corridor_y=corridor_y,
                    lane_x=lane_x,
                )
                airborne_blocks.append(
                    {
                        "color": color,
                        "center": center,
                        "size": (block_width, block_height),
                    }
                )

            frame_state = [list(tube) for tube in state]
            if departed_count:
                del frame_state[source_idx][-departed_count:]
            if landed_count:
                frame_state[dest_idx].extend(moving_blocks[:landed_count])
            frame_state_tuple = tuple(tuple(tube) for tube in frame_state)
            frames.append(
                self._render_board(
                    sample,
                    frame_state_tuple,
                    tubes_geometry,
                    moving_blocks=airborne_blocks or None,
                )
            )
        return frames, frame_state_tuple

    def _carry_corridor_y(self, tubes_geometry: list[dict[str, float]], block_height: float) -> float:
        carry_corridor_clearance = float(self.style.get("carry_corridor_clearance", 18))
        carry_top_margin = float(self.style.get("carry_top_margin", 6))
        min_tube_y = min(geom["y"] for geom in tubes_geometry)
        desired = min_tube_y - block_height / 2 - carry_corridor_clearance
        return max(block_height / 2 + carry_top_margin, desired)

    def _render_board(
        self,
        sample: PuzzleSpec,
        state: State,
        tubes_geometry: list[dict[str, float]],
        moving_blocks: list[dict[str, Any]] | None,
    ) -> np.ndarray:
        colors = self._style_colors(sample.palette)
        frame = Image.new("RGBA", sample.resolution, colors["background"] + (255,))
        draw = ImageDraw.Draw(frame)

        for tube_idx, geom in enumerate(tubes_geometry):
            self._draw_tube(draw, geom, colors)
            for block_idx, color in enumerate(state[tube_idx]):
                center_y = self._block_center_y(geom, block_idx, sample.capacity)
                self._draw_block(
                    draw=draw,
                    center=(geom["x"] + geom["width"] / 2, center_y),
                    size=(self._block_width(geom["width"]), self._block_height(geom["height"], sample.capacity)),
                    fill=colors["blocks"][color % len(colors["blocks"])],
                    shadow=colors["shadow"],
                )

        if moving_blocks is not None:
            for moving_block in moving_blocks:
                color = colors["blocks"][int(moving_block["color"]) % len(colors["blocks"])]
                self._draw_block(
                    draw=draw,
                    center=moving_block["center"],
                    size=moving_block["size"],
                    fill=color,
                    shadow=colors["shadow"],
                )

        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    def _style_colors(self, palette: tuple[str, ...]) -> dict[str, Any]:
        base_blocks = [self._hex_to_rgb(value) for value in palette]
        if len(base_blocks) < 4:
            base_blocks = base_blocks + [
                (95, 205, 170),
                (255, 196, 86),
                (189, 140, 255),
                (255, 146, 96),
            ]
        return {
            "background": self._hex_to_rgb(self.style.get("background", "#0f1220")),
            "tube_outer": self._hex_to_rgb(self.style.get("tube_outer", "#666a74")),
            "tube_inner": self._hex_to_rgb(self.style.get("tube_inner", "#232733")),
            "shadow": self._hex_to_rgb(self.style.get("shadow", "#10131d")),
            "blocks": base_blocks,
        }

    def _draw_tube(self, draw: ImageDraw.ImageDraw, geom: dict[str, float], colors: dict[str, Any]) -> None:
        wall = float(self.style.get("tube_wall", 4))
        radius = int(self.style.get("tube_radius", 10))
        x1 = geom["x"]
        y1 = geom["y"]
        x2 = geom["x"] + geom["width"]
        y2 = geom["y"] + geom["height"]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=colors["tube_outer"] + (255,))
        draw.rounded_rectangle(
            (x1 + wall, y1, x2 - wall, y2 - wall),
            radius=max(2, radius - 2),
            fill=colors["tube_inner"] + (255,),
        )
        draw.rectangle((x1 + wall, y1, x2 - wall, y1 + wall + 2), fill=colors["background"] + (255,))

    def _draw_block(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[float, float],
        size: tuple[float, float],
        fill: tuple[int, int, int],
        shadow: tuple[int, int, int],
    ) -> None:
        width, height = size
        x1 = center[0] - width / 2
        y1 = center[1] - height / 2
        x2 = center[0] + width / 2
        y2 = center[1] + height / 2
        radius = max(4, int(min(width, height) * 0.28))
        draw.rounded_rectangle((x1 + 1, y1 + 2, x2 + 1, y2 + 2), radius=radius, fill=shadow + (70,))
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=fill + (255,))

    def _block_height(self, tube_height: float, capacity: int) -> float:
        padding = float(self.style.get("tube_inner_padding", 8))
        gap = float(self.style.get("block_gap", 6))
        usable = tube_height - padding * 2 - gap * (capacity - 1)
        return usable / capacity

    def _block_width(self, tube_width: float) -> float:
        side_margin = float(self.style.get("block_side_margin", 7))
        return max(10.0, tube_width - side_margin * 2)

    def _block_center_y(self, geom: dict[str, float], level: int, capacity: int) -> float:
        padding = float(self.style.get("tube_inner_padding", 8))
        gap = float(self.style.get("block_gap", 6))
        block_height = self._block_height(geom["height"], capacity)
        bottom = geom["y"] + geom["height"] - padding - block_height / 2
        return bottom - level * (block_height + gap)

    def _trajectory_center_for_block(
        self,
        sample: PuzzleSpec,
        source_geom: dict[str, float],
        dest_geom: dict[str, float],
        source_level: int,
        dest_level: int,
        local_frame: int,
        start_x: float,
        end_x: float,
        source_above_y: float,
        dest_above_y: float,
        corridor_y: float,
        lane_x: float | None,
    ) -> tuple[float, float]:
        start_y = self._block_center_y(source_geom, source_level, sample.capacity)
        end_y = self._block_center_y(dest_geom, dest_level, sample.capacity)
        step = local_frame + 1

        if step <= self.frames_per_lift:
            progress = step / self.frames_per_lift
            return start_x, start_y + (source_above_y - start_y) * progress

        step -= self.frames_per_lift
        if sample.layout_variant == "two_row":
            frames_per_rise = int(self.style.get("frames_per_rise", 2))
            frames_per_pre_drop = int(self.style.get("frames_per_pre_drop", 2))
            rows_differ = abs(source_geom["y"] - dest_geom["y"]) > 1e-6 and lane_x is not None
            if rows_differ:
                if step <= frames_per_rise:
                    progress = step / frames_per_rise
                    return start_x + (lane_x - start_x) * progress, source_above_y
                step -= frames_per_rise
                if step <= self.frames_per_horizontal:
                    progress = step / self.frames_per_horizontal
                    return lane_x, source_above_y + (dest_above_y - source_above_y) * progress
                step -= self.frames_per_horizontal
                if step <= frames_per_pre_drop:
                    progress = step / frames_per_pre_drop
                    return lane_x + (end_x - lane_x) * progress, dest_above_y
                step -= frames_per_pre_drop
                progress = step / self.frames_per_drop
                return end_x, dest_above_y + (end_y - dest_above_y) * progress
            # Same-row two-row moves should stay on that row instead of routing through the top corridor.
            horizontal_frames = frames_per_rise + self.frames_per_horizontal + frames_per_pre_drop
            if step <= horizontal_frames:
                progress = step / horizontal_frames
                return start_x + (end_x - start_x) * progress, source_above_y
            step -= horizontal_frames
            progress = step / self.frames_per_drop
            drop_start_y = source_above_y
            return end_x, drop_start_y + (end_y - drop_start_y) * progress

        if step <= self.frames_per_horizontal:
            progress = step / self.frames_per_horizontal
            return start_x + (end_x - start_x) * progress, source_above_y
        step -= self.frames_per_horizontal
        progress = step / self.frames_per_drop
        drop_start_y = dest_above_y if start_x == end_x else source_above_y
        return end_x, drop_start_y + (end_y - drop_start_y) * progress

    def _validate_two_row_carry_path(
        self,
        sample: PuzzleSpec,
        block_height: float,
        source_above_y: float,
        corridor_y: float,
        dest_above_y: float,
        lane_x: float | None,
        tubes_geometry: list[dict[str, float]],
    ) -> None:
        if sample.layout_variant != "two_row":
            return
        carry_top_margin = float(self.style.get("carry_top_margin", 6))
        min_center_y = block_height / 2 + carry_top_margin
        if corridor_y < min_center_y - 1e-6:
            raise RuntimeError(f"Two-row carry path leaves the frame for sample {sample.sample_id}")
        if corridor_y > min(source_above_y, dest_above_y) + 1e-6:
            raise RuntimeError(f"Two-row carry path clips tube rims for sample {sample.sample_id}")
        if lane_x is not None:
            if lane_x < block_height / 2 or lane_x > self.resolution[0] - block_height / 2:
                raise RuntimeError(f"Two-row lane leaves the frame for sample {sample.sample_id}")
            for geom in tubes_geometry:
                left = geom["x"]
                right = geom["x"] + geom["width"]
                if left + 1e-6 < lane_x < right - 1e-6:
                    raise RuntimeError(f"Two-row lane intersects a tube body for sample {sample.sample_id}")

    def _two_row_lane_x(
        self,
        source_geom: dict[str, float],
        dest_geom: dict[str, float],
        tubes_geometry: list[dict[str, float]],
    ) -> float:
        centers = sorted({round(geom["x"] + geom["width"] / 2, 6) for geom in tubes_geometry})
        source_center = round(source_geom["x"] + source_geom["width"] / 2, 6)
        dest_center = round(dest_geom["x"] + dest_geom["width"] / 2, 6)
        source_col = centers.index(source_center)
        dest_col = centers.index(dest_center)

        if source_col == dest_col:
            if source_col < len(centers) - 1:
                return (centers[source_col] + centers[source_col + 1]) / 2
            return (centers[source_col - 1] + centers[source_col]) / 2

        if dest_col > source_col:
            return (centers[source_col] + centers[source_col + 1]) / 2
        return (centers[source_col - 1] + centers[source_col]) / 2

    def _top_run_length(self, tube: Tube) -> int:
        if not tube:
            return 0
        top = tube[-1]
        run_length = 0
        for color in reversed(tube):
            if color != top:
                break
            run_length += 1
        return run_length

    def _is_solved_state(self, state: State, capacity: int) -> bool:
        for tube in state:
            if not tube:
                continue
            if len(tube) != capacity or len(set(tube)) != 1:
                return False
        return True

    def _hex_to_rgb(self, value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _generate_split_task(task: dict[str, Any]) -> dict[str, Any]:
    env = WaterPourEnv(task["env_config"])
    entries = env.generate_split(task["output_root"], int(task["level"]), str(task["split"]))
    return {"level": int(task["level"]), "split": str(task["split"]), "entries": entries}
