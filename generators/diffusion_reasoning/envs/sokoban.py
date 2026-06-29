from __future__ import annotations

import importlib.util
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .base_env import BaseEnv

PROMPT = (
    "The scene shows a grid-based layout containing a pusher agent, a movable box, and a target floor location. "
    "The walls are rendered in a single solid color, the player is shown as a solid colored triangle, and the box is rendered in a single solid color. "
    "The pushing rule is: the agent must move to a position directly adjacent to the box, then push the box one "
    "step at a time into an empty adjacent space, with the goal of resting the box on the target location. The box "
    "cannot be pulled, and neither the agent nor the box can pass through walls. Observe the maze layout to determine "
    "the precise sequence of steps. Animate the agent moving and pushing the box smoothly, ensuring discrete grid "
    "transitions where the box ends up on the exact target square without clipping."
)

Cell = tuple[int, int]
State = tuple[Cell, Cell]
Action = str

DIRECTION_MAP: dict[str, Cell] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

OPERATOR_SYMBOLS: dict[str, str] = {
    "spine_v": "V",
    "spine_h": "H",
    "corner": "C",
    "pocket": "P",
    "funnel": "F",
    "split": "S",
    "notch": "N",
}


@dataclass(frozen=True)
class BucketSpec:
    name: str
    topology_split: str
    quota: int
    min_actions: int
    max_actions: int
    min_pushes: int
    min_floor: int
    max_floor: int
    grammar_mode: str
    op_count_range: tuple[int, int]
    allowed_ops: tuple[str, ...]
    required_ops: tuple[str, ...]
    max_retries: int
    min_state_distance: int


@dataclass(frozen=True)
class LayoutSpec:
    walls: tuple[str, ...]
    operators: tuple[str, ...]
    floor_cells: tuple[Cell, ...]
    target: Cell
    grammar_mode: str


@dataclass(frozen=True)
class PuzzleSpec:
    sample_id: str
    difficulty_level: int
    split: str
    topology_split: str
    bucket_name: str
    grammar_mode: str
    board_size: int
    wall_rows: tuple[str, ...]
    operator_sequence: tuple[str, ...]
    initial_player: Cell
    initial_box: Cell
    target: Cell
    solution_actions: tuple[Action, ...]
    solution_states: tuple[State, ...]
    frame_count: int
    fps: int
    resolution: tuple[int, int]
    metrics: dict[str, int | float]


class SokobanEnv(BaseEnv):
    task_name = "sokoban"

    def __init__(self, env_config: dict[str, Any]) -> None:
        super().__init__(env_config)
        global_config = env_config.get("global", {})
        render = env_config.get("render", {})
        generation = env_config.get("generation", {})
        self.output_root = Path(env_config.get("output_root", global_config.get("default_output_root", "data")))
        self.levels = env_config.get("levels", {})
        self.global_seed = int(env_config.get("seed", 0))
        self.default_workers = int(generation.get("workers", max(1, min(4, os.cpu_count() or 1))))
        self.board_size = int(generation.get("board_size", 8))

        self.resolution = tuple(render.get("resolution", [208, 208]))
        self.fps = int(render.get("fps", 10))
        duration = render.get("duration_seconds", {})
        self.min_frames = int(duration.get("min", 3) * self.fps)
        self.max_frames = int(duration.get("max", 7) * self.fps)
        self.start_hold_frames = int(render.get("start_hold_frames", 3))
        self.end_hold_frames = int(render.get("end_hold_frames", 5))
        self.min_frames_per_action = int(render.get("min_frames_per_action", 2))
        self.max_frames_per_action = int(render.get("max_frames_per_action", 5))
        self.base_frames_per_action = int(render.get("base_frames_per_action", 3))
        self.style = render.get("style", {})

    def generate_dataset(
        self,
        output_root: str | Path | None = None,
        levels: list[int] | None = None,
        workers: int | None = None,
    ) -> None:
        resolved_root = Path(output_root) if output_root is not None else self.output_root
        target_levels = levels or sorted(int(level) for level in self.levels.keys())
        manifests: dict[int, dict[str, list[dict[str, Any]]]] = {
            int(level): {"train": [], "eval": []} for level in target_levels
        }
        for level in target_levels:
            for split in ("train", "eval"):
                manifests[int(level)][split] = self.generate_split(resolved_root, int(level), split)

        for level in target_levels:
            difficulty_dir = resolved_root / f"sokoban_{level}level"
            for split, entries in manifests[int(level)].items():
                manifest_path = difficulty_dir / f"{split}_manifest.json"
                with manifest_path.open("w", encoding="utf-8") as handle:
                    json.dump(entries, handle, indent=2, ensure_ascii=False)

    def generate_split(self, output_root: str | Path, level: int, split: str) -> list[dict[str, Any]]:
        resolved_root = Path(output_root)
        level_key = str(level) if str(level) in self.levels else level
        level_config = self.levels[level_key]
        difficulty_dir = resolved_root / f"sokoban_{level}level"
        split_dir = self.ensure_split_dir(difficulty_dir, split)
        bucket_specs = self._build_bucket_schedule(level_config, split)
        self._validate_bucket_search_space(level, split, bucket_specs)

        seen_puzzles: list[PuzzleSpec] = []
        entries: list[dict[str, Any]] = []
        total = len(bucket_specs)
        progress_interval = max(1, total // 20) if total else 1
        for index, bucket in enumerate(bucket_specs):
            puzzle = self._make_puzzle_spec(level, split, index, bucket, seen_puzzles)
            seen_puzzles.append(puzzle)
            artifacts = self._render_sample(split_dir, puzzle)
            entries.append(
                {
                    "sample_id": puzzle.sample_id,
                    "video": artifacts["video"],
                    "metadata": artifacts["metadata"],
                    "topology_split": puzzle.topology_split,
                    "bucket": puzzle.bucket_name,
                    "grammar_mode": puzzle.grammar_mode,
                }
            )
            rendered = index + 1
            if rendered == 1 or rendered == total or rendered % progress_interval == 0:
                print(
                    f"[sokoban] level {level} {split}: rendered {rendered}/{total} samples",
                    flush=True,
                )
        return entries

    def _build_bucket_schedule(self, level_config: dict[str, Any], split: str) -> list[BucketSpec]:
        if split == "train":
            bucket_dicts = self._scale_bucket_dicts(
                list(level_config.get("train_buckets", [])),
                int(level_config["counts"]["train"]),
            )
        else:
            total = int(level_config["counts"]["eval"])
            ood_ratio = float(level_config.get("eval_ood_ratio", 0.4))
            ood_total = min(total, round(total * ood_ratio))
            id_total = total - ood_total
            bucket_dicts = self._scale_bucket_dicts(list(level_config.get("eval_id_buckets", [])), id_total)
            bucket_dicts.extend(self._scale_bucket_dicts(list(level_config.get("eval_ood_buckets", [])), ood_total))

        schedule: list[BucketSpec] = []
        for bucket_dict in bucket_dicts:
            spec = self._bucket_from_dict(bucket_dict)
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
            quota=int(payload["quota"]),
            min_actions=int(payload["action_range"][0]),
            max_actions=int(payload["action_range"][1]),
            min_pushes=int(payload.get("min_pushes", 1)),
            min_floor=int(payload.get("floor_range", [18, 26])[0]),
            max_floor=int(payload.get("floor_range", [18, 26])[1]),
            grammar_mode=str(payload.get("grammar_mode", "core")),
            op_count_range=(int(payload.get("op_count_range", [2, 4])[0]), int(payload.get("op_count_range", [2, 4])[1])),
            allowed_ops=tuple(str(item) for item in payload.get("allowed_ops", [])),
            required_ops=tuple(str(item) for item in payload.get("required_ops", [])),
            max_retries=int(payload.get("max_retries", 320)),
            min_state_distance=int(payload.get("min_state_distance", 3)),
        )

    def _validate_bucket_search_space(self, level: int, split: str, buckets: list[BucketSpec]) -> None:
        grouped: dict[tuple[Any, ...], list[BucketSpec]] = {}
        for bucket in buckets:
            key = (
                bucket.topology_split,
                bucket.grammar_mode,
                bucket.op_count_range,
                bucket.allowed_ops,
                bucket.required_ops,
                bucket.min_floor,
                bucket.max_floor,
            )
            grouped.setdefault(key, []).append(bucket)

        for key, bucket_group in grouped.items():
            quota = len(bucket_group)
            estimate = self._estimate_bucket_capacity(bucket_group[0], probes=min(120, max(40, quota // 2)))
            minimum_required = max(2 * quota, 100)
            if estimate < minimum_required and not self._probe_bucket_feasibility(bucket_group[0], target_successes=min(3, max(1, quota // 10))):
                raise RuntimeError(
                    f"Sokoban bucket group {key} appears undersized at level {level} {split}: "
                    f"estimated capacity={estimate}, required>={minimum_required}"
                )

    def _estimate_bucket_capacity(self, bucket: BucketSpec, probes: int) -> int:
        signatures: set[tuple[str, ...]] = set()
        reach_scores: list[int] = []
        for probe_idx in range(probes):
            rng = np.random.default_rng(self.global_seed + probe_idx * 31_337 + len(bucket.name))
            layout = None
            for _ in range(6):
                layout = self._build_layout(bucket, rng)
                if layout is not None:
                    break
            if layout is None:
                continue
            signature = self._canonical_wall_signature(layout.walls)
            signatures.add(signature)
            reach_scores.append(max(1, len(layout.floor_cells) * max(len(layout.floor_cells) - 1, 1)))
        if not signatures:
            return 0
        return int(len(signatures) * max(1, sum(reach_scores) // max(len(reach_scores), 1)))

    def _probe_bucket_feasibility(self, bucket: BucketSpec, target_successes: int) -> bool:
        successes = 0
        attempt_budget = max(80, target_successes * 40)
        seen_layouts: set[tuple[str, ...]] = set()
        for attempt_idx in range(attempt_budget):
            rng = np.random.default_rng(self.global_seed + 700_001 + attempt_idx * 9_973 + len(bucket.name))
            layout = self._build_layout(bucket, rng)
            if layout is None:
                continue
            wall_signature = self._canonical_wall_signature(layout.walls)
            if wall_signature in seen_layouts:
                continue
            seen_layouts.add(wall_signature)
            candidate = self._sample_candidate_from_layout(layout, bucket, rng)
            if candidate is None:
                continue
            actions, states = self._solve_shortest(layout.walls, candidate[0], candidate[1], layout.target)
            if actions is None or states is None:
                continue
            metrics = self._compute_metrics(layout, actions, states)
            if not self._accept_metrics(metrics, bucket):
                continue
            successes += 1
            if successes >= target_successes:
                return True
        return False

    def _make_puzzle_spec(
        self,
        level: int,
        split: str,
        index: int,
        bucket: BucketSpec,
        kept_puzzles: list[PuzzleSpec],
    ) -> PuzzleSpec:
        base_seed = self._sample_seed(level, split, index)
        for relaxed_bucket in self._bucket_attempts(bucket):
            search_phases = self._search_phases(relaxed_bucket)
            for phase in search_phases:
                retry_budget = int(phase["retry_budget"])
                candidate_attempts = int(phase["candidate_attempts"])
                min_state_distance = int(phase["min_state_distance"])
                for attempt in range(retry_budget):
                    rng = np.random.default_rng(base_seed + int(phase["seed_offset"]) + attempt * 9_973)
                    layout = self._build_layout(relaxed_bucket, rng)
                    if layout is None:
                        continue
                    for _ in range(candidate_attempts):
                        candidate = self._sample_candidate_from_layout(layout, relaxed_bucket, rng)
                        if candidate is None:
                            continue
                        initial_player, initial_box = candidate
                        actions, states = self._solve_shortest(layout.walls, initial_player, initial_box, layout.target)
                        if actions is None or states is None:
                            continue
                        metrics = self._compute_metrics(layout, actions, states)
                        if not self._accept_metrics(metrics, relaxed_bucket):
                            continue
                        if not self._far_enough(layout, initial_player, initial_box, kept_puzzles, min_state_distance):
                            continue
                        frame_count = self._frame_count(len(actions))
                        return PuzzleSpec(
                            sample_id=f"sokoban_l{level}_{split}_{index:04d}",
                            difficulty_level=level,
                            split=split,
                            topology_split=bucket.topology_split,
                            bucket_name=bucket.name,
                            grammar_mode=layout.grammar_mode,
                            board_size=self.board_size,
                            wall_rows=layout.walls,
                            operator_sequence=layout.operators,
                            initial_player=initial_player,
                            initial_box=initial_box,
                            target=layout.target,
                            solution_actions=tuple(actions),
                            solution_states=tuple(states),
                            frame_count=frame_count,
                            fps=self.fps,
                            resolution=self.resolution,
                            metrics=metrics,
                        )
        raise RuntimeError(f"Failed to generate Sokoban puzzle for bucket {bucket.name}")

    def _bucket_attempts(self, bucket: BucketSpec) -> list[BucketSpec]:
        attempts = [bucket]
        if bucket.max_actions >= 20 or bucket.topology_split == "ood":
            attempts.append(
                BucketSpec(
                    name=bucket.name,
                    topology_split=bucket.topology_split,
                    quota=bucket.quota,
                    min_actions=max(6, bucket.min_actions - 1),
                    max_actions=max(bucket.max_actions, bucket.min_actions),
                    min_pushes=max(1, bucket.min_pushes - 1),
                    min_floor=max(12, bucket.min_floor - 1),
                    max_floor=min((self.board_size - 2) ** 2, bucket.max_floor + 1),
                    grammar_mode=bucket.grammar_mode,
                    op_count_range=bucket.op_count_range,
                    allowed_ops=bucket.allowed_ops,
                    required_ops=bucket.required_ops,
                    max_retries=bucket.max_retries * 2,
                    min_state_distance=max(1, bucket.min_state_distance - 1),
                )
            )
            attempts.append(
                BucketSpec(
                    name=bucket.name,
                    topology_split=bucket.topology_split,
                    quota=bucket.quota,
                    min_actions=max(6, bucket.min_actions - 2),
                    max_actions=max(bucket.max_actions + 1, bucket.min_actions),
                    min_pushes=max(1, bucket.min_pushes - 1),
                    min_floor=max(12, bucket.min_floor - 2),
                    max_floor=min((self.board_size - 2) ** 2, bucket.max_floor + 2),
                    grammar_mode=bucket.grammar_mode,
                    op_count_range=(max(1, bucket.op_count_range[0] - 1), bucket.op_count_range[1]),
                    allowed_ops=bucket.allowed_ops,
                    required_ops=tuple(op for op in bucket.required_ops if op != "split"),
                    max_retries=bucket.max_retries * 3,
                    min_state_distance=max(1, bucket.min_state_distance - 2),
                )
            )
        return attempts

    def _search_phases(self, bucket: BucketSpec) -> list[dict[str, int]]:
        base_candidate_attempts = 4 if bucket.max_actions <= 12 else 10
        phases = [
            {
                "retry_budget": bucket.max_retries,
                "candidate_attempts": base_candidate_attempts,
                "min_state_distance": bucket.min_state_distance,
                "seed_offset": 0,
            }
        ]
        if bucket.topology_split == "ood" or bucket.max_actions >= 20:
            phases.append(
                {
                    "retry_budget": bucket.max_retries * 2,
                    "candidate_attempts": base_candidate_attempts * 2,
                    "min_state_distance": max(1, bucket.min_state_distance - 1),
                    "seed_offset": 1_000_000,
                }
            )
        phases.append(
            {
                "retry_budget": bucket.max_retries * 2,
                "candidate_attempts": base_candidate_attempts * 3,
                "min_state_distance": max(1, bucket.min_state_distance - 2),
                "seed_offset": 2_000_000,
            }
        )
        return phases

    def _sample_seed(self, level: int, split: str, index: int) -> int:
        split_offset = 0 if split == "train" else 100_000
        return self.global_seed + level * 10_000 + split_offset + index

    def _build_layout(self, bucket: BucketSpec, rng: np.random.Generator) -> LayoutSpec | None:
        walls = [[True] * self.board_size for _ in range(self.board_size)]
        for row in range(1, self.board_size - 1):
            for col in range(1, self.board_size - 1):
                walls[row][col] = False

        op_total = int(rng.integers(bucket.op_count_range[0], bucket.op_count_range[1] + 1))
        available_ops = list(bucket.allowed_ops)
        if not available_ops:
            available_ops = list(OPERATOR_SYMBOLS)

        operators: list[str] = []
        for required in bucket.required_ops:
            if not self._apply_operator(walls, required, rng):
                return None
            operators.append(required)

        for _ in range(max(0, op_total - len(operators))):
            op = str(rng.choice(available_ops))
            if self._apply_operator(walls, op, rng):
                operators.append(op)

        if not operators:
            return None

        floor_cells = tuple(sorted(self._floor_cells(tuple("".join("#" if cell else "." for cell in row) for row in walls))))
        if not (bucket.min_floor <= len(floor_cells) <= bucket.max_floor):
            return None
        if not self._is_floor_connected(floor_cells):
            return None

        target = self._choose_target_cell(floor_cells, walls, rng)
        if target is None:
            return None

        wall_rows = tuple("".join("#" if cell else "." for cell in row) for row in walls)
        return LayoutSpec(
            walls=wall_rows,
            operators=tuple(operators),
            floor_cells=floor_cells,
            target=target,
            grammar_mode=bucket.grammar_mode,
        )

    def _apply_operator(self, walls: list[list[bool]], name: str, rng: np.random.Generator) -> bool:
        interior_min = 1
        interior_max = self.board_size - 2

        if name == "spine_v":
            col = int(rng.integers(2, self.board_size - 2))
            start = int(rng.integers(interior_min + 1, interior_max - 1))
            length = int(rng.integers(2, min(5, interior_max - start + 1)))
            gap = int(rng.integers(start, start + length))
            for row in range(start, start + length):
                if row != gap:
                    walls[row][col] = True
            return True

        if name == "spine_h":
            row = int(rng.integers(2, self.board_size - 2))
            start = int(rng.integers(interior_min + 1, interior_max - 1))
            length = int(rng.integers(2, min(5, interior_max - start + 1)))
            gap = int(rng.integers(start, start + length))
            for col in range(start, start + length):
                if col != gap:
                    walls[row][col] = True
            return True

        if name == "corner":
            anchor_row = int(rng.integers(1, self.board_size - 3))
            anchor_col = int(rng.integers(1, self.board_size - 3))
            height = int(rng.integers(2, 4))
            width = int(rng.integers(2, 4))
            for row in range(anchor_row, min(anchor_row + height, self.board_size - 1)):
                walls[row][anchor_col] = True
            for col in range(anchor_col, min(anchor_col + width, self.board_size - 1)):
                walls[anchor_row][col] = True
            return True

        if name == "pocket":
            side = str(rng.choice(["top", "bottom", "left", "right"]))
            if side in ("top", "bottom"):
                row = 2 if side == "top" else self.board_size - 3
                col = int(rng.integers(2, self.board_size - 2))
                walls[row][col] = True
                walls[row][max(1, col - 1)] = True
            else:
                col = 2 if side == "left" else self.board_size - 3
                row = int(rng.integers(2, self.board_size - 2))
                walls[row][col] = True
                walls[max(1, row - 1)][col] = True
            return True

        if name == "funnel":
            row = int(rng.integers(2, self.board_size - 2))
            col = int(rng.integers(2, self.board_size - 2))
            horizontal = bool(rng.integers(0, 2))
            if horizontal:
                walls[row - 1][col] = True
                walls[row + 1][col] = True
            else:
                walls[row][col - 1] = True
                walls[row][col + 1] = True
            return True

        if name == "split":
            horizontal = bool(rng.integers(0, 2))
            if horizontal:
                row = int(rng.integers(2, self.board_size - 2))
                gap = int(rng.integers(1, self.board_size - 1))
                for col in range(1, self.board_size - 1):
                    if col != gap:
                        walls[row][col] = True
            else:
                col = int(rng.integers(2, self.board_size - 2))
                gap = int(rng.integers(1, self.board_size - 1))
                for row in range(1, self.board_size - 1):
                    if row != gap:
                        walls[row][col] = True
            return True

        if name == "notch":
            row = int(rng.integers(1, self.board_size - 1))
            col = int(rng.integers(1, self.board_size - 1))
            walls[row][col] = True
            return True

        return False

    def _floor_cells(self, wall_rows: tuple[str, ...]) -> list[Cell]:
        return [
            (row, col)
            for row in range(self.board_size)
            for col in range(self.board_size)
            if wall_rows[row][col] == "."
        ]

    def _is_floor_connected(self, floor_cells: tuple[Cell, ...]) -> bool:
        if not floor_cells:
            return False
        floor_set = set(floor_cells)
        queue = deque([floor_cells[0]])
        seen = {floor_cells[0]}
        while queue:
            row, col = queue.popleft()
            for dr, dc in DIRECTION_MAP.values():
                nxt = (row + dr, col + dc)
                if nxt not in floor_set or nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return len(seen) == len(floor_cells)

    def _choose_target_cell(
        self,
        floor_cells: tuple[Cell, ...],
        walls: list[list[bool]],
        rng: np.random.Generator,
    ) -> Cell | None:
        candidates = [cell for cell in floor_cells if self._deadlock_like_score(walls, cell) <= 1]
        if not candidates:
            candidates = list(floor_cells)
        if not candidates:
            return None
        weights = np.array([1.0 + self._centrality(cell) for cell in candidates], dtype=float)
        weights /= weights.sum()
        index = int(rng.choice(len(candidates), p=weights))
        return candidates[index]

    def _sample_candidate_from_layout(
        self,
        layout: LayoutSpec,
        bucket: BucketSpec,
        rng: np.random.Generator,
    ) -> State | None:
        target = layout.target
        target_neighbors = [cell for cell in self._neighbors(target) if self._is_floor(layout.walls, cell)]
        if not target_neighbors:
            return None
        start_states = [(neighbor, target) for neighbor in target_neighbors]
        queue = deque(start_states)
        distance: dict[State, int] = {state: 0 for state in start_states}
        candidates: list[State] = []
        max_depth = bucket.max_actions + 4

        while queue:
            state = queue.popleft()
            current_depth = distance[state]
            if bucket.min_actions <= current_depth <= bucket.max_actions:
                if state[1] != target and self._deadlock_like_score_from_rows(layout.walls, state[1]) < 2:
                    if self._box_distance(state[1], target) >= max(2, bucket.min_pushes - 1):
                        candidates.append(state)
            if current_depth >= max_depth:
                continue
            for next_state, _ in self._reverse_neighbors(layout.walls, state):
                if next_state in distance:
                    continue
                distance[next_state] = current_depth + 1
                queue.append(next_state)

        if not candidates:
            return None
        scored = []
        for candidate in candidates:
            score = 0.0
            score += distance[candidate] * 1.2
            score += self._box_distance(candidate[1], target) * 0.8
            score += self._deadlock_like_score_from_rows(layout.walls, candidate[1]) * 0.6
            score += float(rng.random())
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = min(8, len(scored))
        return scored[int(rng.integers(0, top_k))][1]

    def _solve_shortest(
        self,
        wall_rows: tuple[str, ...],
        start_player: Cell,
        start_box: Cell,
        target: Cell,
    ) -> tuple[list[Action] | None, list[State] | None]:
        start: State = (start_player, start_box)
        queue = deque([start])
        parent: dict[State, tuple[State | None, Action | None]] = {start: (None, None)}
        while queue:
            state = queue.popleft()
            player, box = state
            if box == target:
                actions: list[Action] = []
                states: list[State] = []
                cursor = state
                while True:
                    states.append(cursor)
                    prev_state, action = parent[cursor]
                    if prev_state is None or action is None:
                        break
                    actions.append(action)
                    cursor = prev_state
                actions.reverse()
                states.reverse()
                return actions, states
            for next_state, action in self._forward_neighbors(wall_rows, state):
                if next_state in parent:
                    continue
                parent[next_state] = (state, action)
                queue.append(next_state)
        return None, None

    def _forward_neighbors(self, wall_rows: tuple[str, ...], state: State) -> list[tuple[State, Action]]:
        player, box = state
        result: list[tuple[State, Action]] = []
        for action, (dr, dc) in DIRECTION_MAP.items():
            next_player = (player[0] + dr, player[1] + dc)
            if not self._is_floor(wall_rows, next_player):
                continue
            if next_player == box:
                next_box = (box[0] + dr, box[1] + dc)
                if not self._is_floor(wall_rows, next_box):
                    continue
                result.append((((next_player[0], next_player[1]), next_box), action))
            else:
                result.append(((next_player, box), action))
        return result

    def _reverse_neighbors(self, wall_rows: tuple[str, ...], state: State) -> list[tuple[State, Action]]:
        player, box = state
        result: list[tuple[State, Action]] = []
        for action, (dr, dc) in DIRECTION_MAP.items():
            prev_player = (player[0] - dr, player[1] - dc)
            if self._is_floor(wall_rows, prev_player) and prev_player != box:
                result.append(((prev_player, box), action))
            if player == box:
                continue
            if abs(player[0] - box[0]) + abs(player[1] - box[1]) != 1:
                continue
            if (player[0] - box[0], player[1] - box[1]) != (-dr, -dc):
                continue
            source_player = (player[0] - dr, player[1] - dc)
            if not self._is_floor(wall_rows, source_player):
                continue
            if source_player == box:
                continue
            result.append(((source_player, player), action))
        return result

    def _is_floor(self, wall_rows: tuple[str, ...], cell: Cell) -> bool:
        row, col = cell
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        return wall_rows[row][col] == "."

    def _neighbors(self, cell: Cell) -> list[Cell]:
        row, col = cell
        return [(row + dr, col + dc) for dr, dc in DIRECTION_MAP.values()]

    def _compute_metrics(
        self,
        layout: LayoutSpec,
        actions: list[Action],
        states: list[State],
    ) -> dict[str, int | float]:
        push_count = 0
        for previous, current in zip(states, states[1:]):
            if previous[1] != current[1]:
                push_count += 1
        floor_count = len(layout.floor_cells)
        return {
            "optimal_solution_length": len(actions),
            "push_count": push_count,
            "player_only_steps": len(actions) - push_count,
            "reachable_floor_count": floor_count,
            "operator_count": len(layout.operators),
            "deadlock_like_cells": sum(self._deadlock_like_score_from_rows(layout.walls, cell) >= 2 for cell in layout.floor_cells),
            "box_target_distance": abs(states[0][1][0] - layout.target[0]) + abs(states[0][1][1] - layout.target[1]),
        }

    def _accept_metrics(self, metrics: dict[str, int | float], bucket: BucketSpec) -> bool:
        action_count = int(metrics["optimal_solution_length"])
        if action_count < bucket.min_actions or action_count > bucket.max_actions:
            return False
        if int(metrics["push_count"]) < bucket.min_pushes:
            return False
        if int(metrics["reachable_floor_count"]) < bucket.min_floor or int(metrics["reachable_floor_count"]) > bucket.max_floor:
            return False
        return True

    def _far_enough(
        self,
        layout: LayoutSpec,
        player: Cell,
        box: Cell,
        puzzles: list[PuzzleSpec],
        min_state_distance: int,
    ) -> bool:
        if min_state_distance <= 0:
            return True
        current_signature = self._canonical_wall_signature(layout.walls)
        for puzzle in puzzles:
            other_signature = self._canonical_wall_signature(puzzle.wall_rows)
            if current_signature != other_signature:
                continue
            distance = 0
            distance += abs(player[0] - puzzle.initial_player[0]) + abs(player[1] - puzzle.initial_player[1])
            distance += abs(box[0] - puzzle.initial_box[0]) + abs(box[1] - puzzle.initial_box[1])
            distance += abs(layout.target[0] - puzzle.target[0]) + abs(layout.target[1] - puzzle.target[1])
            if distance < min_state_distance:
                return False
        return True

    def _canonical_wall_signature(self, wall_rows: tuple[str, ...]) -> tuple[str, ...]:
        variants = []
        grid = [list(row) for row in wall_rows]
        for transformed in self._grid_symmetries(grid):
            variants.append(tuple("".join(row) for row in transformed))
        return min(variants)

    def _grid_symmetries(self, grid: list[list[str]]) -> list[list[list[str]]]:
        result = []
        current = [row[:] for row in grid]
        for _ in range(4):
            result.append([row[:] for row in current])
            result.append([list(reversed(row)) for row in current])
            current = self._rotate_grid(current)
        unique: list[list[list[str]]] = []
        seen: set[tuple[str, ...]] = set()
        for item in result:
            signature = tuple("".join(row) for row in item)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(item)
        return unique

    def _rotate_grid(self, grid: list[list[str]]) -> list[list[str]]:
        size = len(grid)
        return [[grid[size - 1 - col][row] for col in range(size)] for row in range(size)]

    def _deadlock_like_score(self, walls: list[list[bool]], cell: Cell) -> int:
        row, col = cell
        up = walls[row - 1][col]
        down = walls[row + 1][col]
        left = walls[row][col - 1]
        right = walls[row][col + 1]
        corners = 0
        corners += 1 if up and left else 0
        corners += 1 if up and right else 0
        corners += 1 if down and left else 0
        corners += 1 if down and right else 0
        return corners

    def _deadlock_like_score_from_rows(self, wall_rows: tuple[str, ...], cell: Cell) -> int:
        row, col = cell
        up = wall_rows[row - 1][col] == "#"
        down = wall_rows[row + 1][col] == "#"
        left = wall_rows[row][col - 1] == "#"
        right = wall_rows[row][col + 1] == "#"
        corners = 0
        corners += 1 if up and left else 0
        corners += 1 if up and right else 0
        corners += 1 if down and left else 0
        corners += 1 if down and right else 0
        return corners

    def _centrality(self, cell: Cell) -> float:
        center = (self.board_size - 1) / 2
        return max(0.0, 3.5 - (abs(cell[0] - center) + abs(cell[1] - center)))

    def _box_distance(self, left: Cell, right: Cell) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _frame_count(self, action_count: int) -> int:
        if action_count <= 0:
            return self.start_hold_frames + self.end_hold_frames + 1
        hold_total = self.start_hold_frames + self.end_hold_frames
        unclipped = action_count * self.base_frames_per_action + hold_total
        target_total = int(np.clip(unclipped, self.min_frames, self.max_frames))
        available = max(target_total - hold_total, action_count * self.min_frames_per_action)
        per_action = max(
            self.min_frames_per_action,
            min(self.max_frames_per_action, available // max(action_count, 1)),
        )
        return hold_total + action_count * per_action

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
        move_payloads: list[dict[str, Any]] = []
        for action, prev_state, next_state in zip(sample.solution_actions, sample.solution_states, sample.solution_states[1:]):
            move_payloads.append(
                {
                    "action": action,
                    "player_from": list(prev_state[0]),
                    "player_to": list(next_state[0]),
                    "box_from": list(prev_state[1]),
                    "box_to": list(next_state[1]),
                    "push": prev_state[1] != next_state[1],
                }
            )
        return {
            "task": self.task_name,
            "prompt": PROMPT,
            "sample_id": sample.sample_id,
            "difficulty_level": sample.difficulty_level,
            "split": sample.split,
            "topology_split": sample.topology_split,
            "bucket": sample.bucket_name,
            "grammar_mode": sample.grammar_mode,
            "puzzle": {
                "board_size": [sample.board_size, sample.board_size],
                "wall_rows": list(sample.wall_rows),
                "operators": list(sample.operator_sequence),
                "initial_player": list(sample.initial_player),
                "initial_box": list(sample.initial_box),
                "target": list(sample.target),
            },
            "solution": {
                "actions": move_payloads,
                "action_count": len(sample.solution_actions),
                "shortest_solution": True,
                "states": [
                    {
                        "player": list(state[0]),
                        "box": list(state[1]),
                    }
                    for state in sample.solution_states
                ],
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

    def _render_frames(self, sample: PuzzleSpec) -> list[np.ndarray]:
        action_count = len(sample.solution_actions)
        hold_total = self.start_hold_frames + self.end_hold_frames
        per_action = (
            0
            if action_count == 0
            else max(
                self.min_frames_per_action,
                min(
                    self.max_frames_per_action,
                    max(sample.frame_count - hold_total, action_count * self.min_frames_per_action) // max(action_count, 1),
                ),
            )
        )
        geometry = self._board_geometry()
        frames: list[np.ndarray] = []
        initial_frame = self._render_board(sample, sample.solution_states[0], geometry, progress=None, next_state=None)
        frames.extend([initial_frame] * self.start_hold_frames)
        for state, next_state, action in zip(sample.solution_states, sample.solution_states[1:], sample.solution_actions):
            for frame_idx in range(1, max(per_action, 1) + 1):
                progress = frame_idx / max(per_action, 1)
                frames.append(self._render_board(sample, state, geometry, progress=progress, next_state=next_state, action=action))
        final_frame = self._render_board(sample, sample.solution_states[-1], geometry, progress=None, next_state=None)
        frames.extend([final_frame] * self.end_hold_frames)
        return frames

    def _board_geometry(self) -> dict[str, float]:
        width, height = self.resolution
        panel_margin = float(self.style.get("panel_margin", 6))
        board_padding = float(self.style.get("board_padding", 18))
        cell_gap = float(self.style.get("cell_gap", 2))
        board_extent = min(width, height) - 2 * board_padding
        cell_size = (board_extent - (self.board_size - 1) * cell_gap) / self.board_size
        board_size_px = self.board_size * cell_size + (self.board_size - 1) * cell_gap
        origin_x = (width - board_size_px) / 2
        origin_y = (height - board_size_px) / 2
        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_size": cell_size,
            "cell_gap": cell_gap,
            "panel_margin": panel_margin,
        }

    def _cell_rect(self, cell: Cell, geometry: dict[str, float]) -> tuple[float, float, float, float]:
        row, col = cell
        x0 = geometry["origin_x"] + col * (geometry["cell_size"] + geometry["cell_gap"])
        y0 = geometry["origin_y"] + row * (geometry["cell_size"] + geometry["cell_gap"])
        x1 = x0 + geometry["cell_size"]
        y1 = y0 + geometry["cell_size"]
        return x0, y0, x1, y1

    def _render_board(
        self,
        sample: PuzzleSpec,
        state: State,
        geometry: dict[str, float],
        progress: float | None,
        next_state: State | None,
        action: Action | None = None,
    ) -> np.ndarray:
        width, height = sample.resolution
        background = self.style.get("background", "#202226")
        panel = self.style.get("panel", "#10131a")
        floor_color = self.style.get("floor", "#12161c")
        floor_grid = self.style.get("floor_grid", "#3e434d")
        wall_fill = self.style.get("wall_fill", "#40455c")
        target_color = self.style.get("target", "#82e17f")
        player_color = self.style.get("player", "#a9c7ff")
        box_color = self.style.get("box", "#f4b74e")

        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (geometry["panel_margin"], geometry["panel_margin"], width - geometry["panel_margin"], height - geometry["panel_margin"]),
            radius=int(self.style.get("panel_radius", 16)),
            fill=panel,
        )

        for row in range(sample.board_size):
            for col in range(sample.board_size):
                rect = self._cell_rect((row, col), geometry)
                draw.rectangle(rect, outline=floor_grid, width=1, fill=floor_color)
                if sample.wall_rows[row][col] == "#":
                    draw.rectangle(rect, fill=wall_fill, outline="#1f232c", width=1)

        target_rect = self._cell_rect(sample.target, geometry)
        self._draw_target(draw, target_rect, target_color)

        player_pos = state[0]
        box_pos = state[1]
        if progress is not None and next_state is not None:
            player_pos = self._lerp_cell(state[0], next_state[0], progress)
            box_pos = self._lerp_cell(state[1], next_state[1], progress)

        self._draw_box(draw, self._cell_rect_float(box_pos, geometry), box_color)
        self._draw_player(draw, self._cell_rect_float(player_pos, geometry), player_color)

        return np.array(image)

    def _cell_rect_float(self, cell: tuple[float, float], geometry: dict[str, float]) -> tuple[float, float, float, float]:
        row, col = cell
        x0 = geometry["origin_x"] + col * (geometry["cell_size"] + geometry["cell_gap"])
        y0 = geometry["origin_y"] + row * (geometry["cell_size"] + geometry["cell_gap"])
        x1 = x0 + geometry["cell_size"]
        y1 = y0 + geometry["cell_size"]
        return x0, y0, x1, y1

    def _lerp_cell(self, start: Cell, end: Cell, progress: float) -> tuple[float, float]:
        return (
            start[0] + (end[0] - start[0]) * progress,
            start[1] + (end[1] - start[1]) * progress,
        )

    def _draw_target(
        self,
        draw: ImageDraw.ImageDraw,
        rect: tuple[float, float, float, float],
        color: str,
    ) -> None:
        x0, y0, x1, y1 = rect
        inset = (x1 - x0) * 0.22
        width = max(2, int((x1 - x0) * 0.08))
        draw.ellipse((x0 + inset, y0 + inset, x1 - inset, y1 - inset), outline=color, width=width)

    def _draw_player(self, draw: ImageDraw.ImageDraw, rect: tuple[float, float, float, float], color: str) -> None:
        x0, y0, x1, y1 = rect
        width = x1 - x0
        height = y1 - y0
        inset_x = width * 0.18
        inset_y = height * 0.14
        points = [
            ((x0 + x1) / 2, y0 + inset_y),
            (x1 - inset_x, y1 - inset_y),
            (x0 + inset_x, y1 - inset_y),
        ]
        draw.polygon(points, fill=color)

    def _draw_box(self, draw: ImageDraw.ImageDraw, rect: tuple[float, float, float, float], color: str) -> None:
        x0, y0, x1, y1 = rect
        inset = (x1 - x0) * 0.20
        draw.rectangle((x0 + inset, y0 + inset, x1 - inset, y1 - inset), fill=color)
