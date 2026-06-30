from __future__ import annotations

import os
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .base_env import BaseEnv

PROMPT = (
    "The scene shows a grid containing connected square cells. The drawing rule is: "
    "start at the highlighted initial square, then trace a continuous, unbroken path to visit "
    "every required square exactly once without lifting the line or revisiting any previously "
    "visited square. Observe the occupied cell pattern to determine the correct route. "
    "Animate the drawing smoothly from square to square, ensuring the path fully covers the "
    "entire shape and ends in a state where every required square has been visited."
)

Cell = tuple[int, int]
Direction = tuple[int, int]
DIRECTIONS: tuple[Direction, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class PuzzleSpec:
    sample_id: str
    difficulty_level: int
    split: str
    topology_split: str
    family: str
    board_size: tuple[int, int]
    occupied_cells: list[Cell]
    solution_path: list[Cell]
    start_cell: Cell
    frame_count: int
    fps: int
    resolution: tuple[int, int]


class OneLineDrawingEnv(BaseEnv):
    task_name = "one_line_drawing"

    def __init__(self, env_config: dict[str, Any]) -> None:
        super().__init__(env_config)
        global_config = env_config.get("global", {})
        render = env_config.get("render", {})
        self.output_root = Path(env_config.get("output_root", global_config.get("default_output_root", "data")))
        self.resolution = tuple(render.get("resolution", [208, 208]))
        self.fps = int(render.get("fps", 10))
        self.min_frames = int(render.get("duration_seconds", {}).get("min", 8) * self.fps)
        self.max_frames = int(render.get("duration_seconds", {}).get("max", 20) * self.fps)
        self.frames_per_edge = int(render.get("frames_per_edge", 16))
        self.hold_frames = int(render.get("hold_frames", 6))
        self.levels = env_config.get("levels", {})
        self.style = render.get("style", {})
        self.global_seed = int(env_config.get("seed", 0))
        generation = env_config.get("generation", {})
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
            level_key = str(level) if str(level) in self.levels else level
            difficulty_dir = resolved_root / f"one_line_drawing_{level}level"
            for split, entries in manifests[int(level)].items():
                manifest_path = difficulty_dir / f"{split}_manifest.json"
                with manifest_path.open("w", encoding="utf-8") as handle:
                    json.dump(entries, handle, indent=2, ensure_ascii=False)

    def generate_split(
        self,
        output_root: str | Path,
        level: int,
        split: str,
    ) -> list[dict[str, Any]]:
        resolved_root = Path(output_root)
        level_key = str(level) if str(level) in self.levels else level
        level_config = self.levels[level_key]
        difficulty_dir = resolved_root / f"one_line_drawing_{level}level"
        split_dir = self.ensure_split_dir(difficulty_dir, split)
        sample_specs = self._build_split_specs(level, split, level_config)
        seen_signatures: set[tuple[Cell, ...]] = set()
        entries: list[dict[str, Any]] = []
        for index, sample in enumerate(sample_specs):
            puzzle = self._make_puzzle_spec(level, split, index, sample, level_config, seen_signatures)
            artifacts = self._render_sample(split_dir, puzzle)
            entries.append(
                {
                    "sample_id": puzzle.sample_id,
                    "video": artifacts["video"],
                    "metadata": artifacts["metadata"],
                    "topology_split": puzzle.topology_split,
                    "family": puzzle.family,
                }
            )
        return entries

    def _build_split_specs(self, level: int, split: str, level_config: dict[str, Any]) -> list[dict[str, Any]]:
        counts = level_config["counts"]
        total = int(counts[split])
        generator = level_config["generator"]
        train_families = list(generator["train_families"])
        eval_id_families = list(generator.get("eval_id_families", train_families))
        eval_ood_families = list(generator["eval_ood_families"])

        if split == "train":
            return [
                {
                    "topology_split": "id",
                    "family": train_families[index % len(train_families)],
                    "family_candidates": self._rotated_families(train_families, index),
                    "index": index,
                }
                for index in range(total)
            ]

        ood_count = min(total, round(total * float(level_config.get("eval_ood_ratio", 0.5))))
        id_count = total - ood_count
        specs = [
            {
                "topology_split": "id",
                "family": eval_id_families[index % len(eval_id_families)],
                "family_candidates": self._rotated_families(eval_id_families, index),
                "index": index,
            }
            for index in range(id_count)
        ]
        specs.extend(
            {
                "topology_split": "ood",
                "family": eval_ood_families[index % len(eval_ood_families)],
                "family_candidates": self._rotated_families(eval_ood_families, index),
                "index": id_count + index,
            }
            for index in range(ood_count)
        )
        return specs

    def _make_puzzle_spec(
        self,
        level: int,
        split: str,
        index: int,
        sample_config: dict[str, Any],
        level_config: dict[str, Any],
        seen_signatures: set[tuple[Cell, ...]],
    ) -> PuzzleSpec:
        board_size = tuple(level_config["board_size"])
        min_cells, max_cells = level_config["cell_count_range"]
        uniqueness_attempts = int(level_config.get("uniqueness_attempts", 200))
        max_shape_span = level_config.get("max_shape_span")
        base_seed = self._sample_seed(level, split, index)

        normalized_path: list[Cell] | None = None
        pattern_signature: tuple[Cell, ...] | None = None
        selected_family: str | None = None
        family_candidates = list(sample_config.get("family_candidates", [sample_config["family"]]))

        for family_offset, family in enumerate(family_candidates):
            family_seed = base_seed + family_offset * 1_000_000
            for uniqueness_offset in range(uniqueness_attempts):
                rng = np.random.default_rng(family_seed + uniqueness_offset * 100_003)
                target_cells = int(rng.integers(min_cells, max_cells + 1))
                path = self._generate_path(
                    board_size=board_size,
                    target_cells=target_cells,
                    family=family,
                    rng=rng,
                    max_attempts=int(level_config.get("max_attempts", 250)),
                )
                candidate_path = self._normalize_path(path)
                if max_shape_span is not None and not self._shape_within_span(candidate_path, int(max_shape_span)):
                    continue
                if not self._meets_shape_requirements(candidate_path, level_config):
                    continue
                candidate_signature = self._pattern_signature(candidate_path)
                if candidate_signature in seen_signatures:
                    continue
                normalized_path = candidate_path
                pattern_signature = candidate_signature
                selected_family = family
                seen_signatures.add(candidate_signature)
                break
            if normalized_path is not None:
                break

        if normalized_path is None or pattern_signature is None or selected_family is None:
            raise RuntimeError(
                "Failed to generate a unique one-line pattern "
                f"for level {level} split '{split}' using families {family_candidates}"
            )

        frame_count = int(
            np.clip((len(normalized_path) - 1) * self.frames_per_edge + self.hold_frames, self.min_frames, self.max_frames)
        )

        return PuzzleSpec(
            sample_id=f"one_line_drawing_l{level}_{split}_{index:04d}",
            difficulty_level=level,
            split=split,
            topology_split=sample_config["topology_split"],
            family=selected_family,
            board_size=board_size,
            occupied_cells=normalized_path,
            solution_path=normalized_path,
            start_cell=normalized_path[0],
            frame_count=frame_count,
            fps=self.fps,
            resolution=self.resolution,
        )

    def _sample_seed(self, level: int, split: str, index: int) -> int:
        split_offset = 0 if split == "train" else 10_000
        return self.global_seed + level * 1_000 + split_offset + index

    def _rotated_families(self, families: list[str], index: int) -> list[str]:
        if not families:
            return []
        offset = index % len(families)
        return families[offset:] + families[:offset]

    def _generate_path(
        self,
        board_size: tuple[int, int],
        target_cells: int,
        family: str,
        rng: np.random.Generator,
        max_attempts: int,
    ) -> list[Cell]:
        for _ in range(max_attempts):
            start = (int(rng.integers(0, board_size[0])), int(rng.integers(0, board_size[1])))
            path = [start]
            used = {start}
            if self._search_path(path, used, board_size, target_cells, family, rng):
                transformed = self._apply_random_transform(path, rng)
                self._validate_solution(transformed)
                return transformed
        raise RuntimeError(f"Failed to generate valid one-line path for family '{family}'")

    def _search_path(
        self,
        path: list[Cell],
        used: set[Cell],
        board_size: tuple[int, int],
        target_cells: int,
        family: str,
        rng: np.random.Generator,
    ) -> bool:
        if len(path) == target_cells:
            return True

        previous_direction: Direction | None = None
        if len(path) > 1:
            previous_direction = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])

        candidates = self._ordered_neighbors(path, used, board_size, family, previous_direction, rng)
        if not candidates:
            return False

        for neighbor in candidates:
            path.append(neighbor)
            used.add(neighbor)
            if self._remaining_cells_reachable(path, used, board_size, target_cells):
                if self._search_path(path, used, board_size, target_cells, family, rng):
                    return True
            used.remove(neighbor)
            path.pop()
        return False

    def _ordered_neighbors(
        self,
        path: list[Cell],
        used: set[Cell],
        board_size: tuple[int, int],
        family: str,
        previous_direction: Direction | None,
        rng: np.random.Generator,
    ) -> list[Cell]:
        current = path[-1]
        neighbors: list[Cell] = []
        for dx, dy in DIRECTIONS:
            candidate = (current[0] + dx, current[1] + dy)
            if not self._in_bounds(candidate, board_size) or candidate in used:
                continue
            neighbors.append(candidate)

        scored = [(self._neighbor_score(path, neighbor, family, previous_direction, board_size, rng), neighbor) for neighbor in neighbors]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [neighbor for _, neighbor in scored]

    def _neighbor_score(
        self,
        path: list[Cell],
        neighbor: Cell,
        family: str,
        previous_direction: Direction | None,
        board_size: tuple[int, int],
        rng: np.random.Generator,
    ) -> tuple[float, float, float, float]:
        current = path[-1]
        direction = (neighbor[0] - current[0], neighbor[1] - current[1])
        same_direction = 1.0 if previous_direction is not None and direction == previous_direction else 0.0
        changed_direction = 1.0 if previous_direction is not None and direction != previous_direction else 0.0
        edge_distance = min(neighbor[0], neighbor[1], board_size[0] - 1 - neighbor[0], board_size[1] - 1 - neighbor[1])
        center_distance = abs(neighbor[0] - (board_size[0] - 1) / 2) + abs(neighbor[1] - (board_size[1] - 1) / 2)
        degree = sum(
            1
            for dx, dy in DIRECTIONS
            if self._in_bounds((neighbor[0] + dx, neighbor[1] + dy), board_size)
        )
        jitter = float(rng.random())

        if family == "straight_walk":
            return (3.0 * same_direction, -edge_distance, center_distance, jitter)
        if family == "turning_walk":
            return (3.0 * changed_direction, edge_distance, -center_distance, jitter)
        if family == "snake_walk":
            parity_bonus = 1.0 if (neighbor[1] % 2 == 0 and direction[0] != 0) or (neighbor[1] % 2 == 1 and direction[0] == 0) else 0.0
            return (2.0 * parity_bonus, 1.5 * same_direction, -edge_distance, jitter)
        if family == "perimeter_walk":
            return (2.0 * center_distance, -edge_distance, -same_direction, jitter)
        if family == "turn_back_walk":
            return (2.0 * changed_direction, 0.5 * degree, center_distance, jitter)
        if family == "zigzag_walk":
            preferred_axis = 0 if len(path) % 2 == 0 else 1
            axis_bonus = 1.0 if abs(direction[preferred_axis]) == 1 else 0.0
            return (2.0 * axis_bonus, changed_direction, -edge_distance, jitter)
        return (jitter, -edge_distance, center_distance, same_direction)

    def _remaining_cells_reachable(
        self,
        path: list[Cell],
        used: set[Cell],
        board_size: tuple[int, int],
        target_cells: int,
    ) -> bool:
        remaining = target_cells - len(path)
        if remaining == 0:
            return True

        frontier = [path[-1]]
        seen = {path[-1]}
        reachable = 0
        while frontier:
            cell = frontier.pop()
            for dx, dy in DIRECTIONS:
                candidate = (cell[0] + dx, cell[1] + dy)
                if not self._in_bounds(candidate, board_size) or candidate in used or candidate in seen:
                    continue
                seen.add(candidate)
                frontier.append(candidate)
                reachable += 1
        return reachable >= remaining

    def _apply_random_transform(self, path: list[Cell], rng: np.random.Generator) -> list[Cell]:
        points = np.array(path, dtype=int)
        transform = int(rng.integers(0, 8))
        if transform in (1, 3, 5, 7):
            points[:, 0] *= -1
        if transform in (2, 3, 6, 7):
            points[:, 1] *= -1
        if transform in (4, 5, 6, 7):
            points = points[:, ::-1]
        return [tuple(point) for point in points.tolist()]

    def _normalize_path(self, path: list[Cell]) -> list[Cell]:
        min_x = min(cell[0] for cell in path)
        min_y = min(cell[1] for cell in path)
        return [(cell[0] - min_x, cell[1] - min_y) for cell in path]

    def _pattern_signature(self, path: list[Cell]) -> tuple[Cell, ...]:
        return tuple(sorted(path))

    def _shape_within_span(self, path: list[Cell], max_shape_span: int) -> bool:
        xs = [cell[0] for cell in path]
        ys = [cell[1] for cell in path]
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        return max(width, height) <= max_shape_span

    def _meets_shape_requirements(self, path: list[Cell], level_config: dict[str, Any]) -> bool:
        min_turns = int(level_config.get("min_turns", 0))
        min_side_contacts = int(level_config.get("min_side_contacts", 0))
        min_fill_ratio = float(level_config.get("min_fill_ratio", 0.0))
        max_straight_run = int(level_config.get("max_straight_run", 0))

        if min_turns and self._count_turns(path) < min_turns:
            return False
        if min_side_contacts and self._count_side_contacts(path) < min_side_contacts:
            return False
        if min_fill_ratio and self._fill_ratio(path) < min_fill_ratio:
            return False
        if max_straight_run and self._longest_straight_run(path) > max_straight_run:
            return False
        return True

    def _count_turns(self, path: list[Cell]) -> int:
        turns = 0
        previous_direction: Direction | None = None
        for current, nxt in zip(path[:-1], path[1:]):
            direction = (nxt[0] - current[0], nxt[1] - current[1])
            if previous_direction is not None and direction != previous_direction:
                turns += 1
            previous_direction = direction
        return turns

    def _count_side_contacts(self, path: list[Cell]) -> int:
        occupied = set(path)
        path_edges = {
            tuple(sorted((current, nxt)))
            for current, nxt in zip(path[:-1], path[1:])
        }
        contacts = 0
        for cell in occupied:
            for dx, dy in ((1, 0), (0, 1)):
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in occupied:
                    continue
                if tuple(sorted((cell, neighbor))) not in path_edges:
                    contacts += 1
        return contacts

    def _fill_ratio(self, path: list[Cell]) -> float:
        xs = [cell[0] for cell in path]
        ys = [cell[1] for cell in path]
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        return len(path) / max(width * height, 1)

    def _longest_straight_run(self, path: list[Cell]) -> int:
        longest = 1
        run_length = 1
        previous_direction: Direction | None = None
        for current, nxt in zip(path[:-1], path[1:]):
            direction = (nxt[0] - current[0], nxt[1] - current[1])
            if previous_direction is not None and direction == previous_direction:
                run_length += 1
            else:
                run_length = 1
            previous_direction = direction
            longest = max(longest, run_length)
        return longest

    def _validate_solution(self, path: list[Cell]) -> None:
        if len(set(path)) != len(path):
            raise ValueError("Solution revisits a cell")
        for current, nxt in zip(path[:-1], path[1:]):
            if abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) != 1:
                raise ValueError("Solution contains non-adjacent steps")
        occupied = set(path)
        frontier = [path[0]]
        seen = {path[0]}
        while frontier:
            cell = frontier.pop()
            for dx, dy in DIRECTIONS:
                candidate = (cell[0] + dx, cell[1] + dy)
                if candidate in occupied and candidate not in seen:
                    seen.add(candidate)
                    frontier.append(candidate)
        if seen != occupied:
            raise ValueError("Occupied cells are not connected")

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
        edges = [[list(a), list(b)] for a, b in zip(sample.solution_path[:-1], sample.solution_path[1:])]
        return {
            "task": self.task_name,
            "prompt": PROMPT,
            "sample_id": sample.sample_id,
            "difficulty_level": sample.difficulty_level,
            "split": sample.split,
            "topology_split": sample.topology_split,
            "family": sample.family,
            "graph": {
                "board_size": list(sample.board_size),
                "occupied_cells": [list(cell) for cell in sample.occupied_cells],
                "edges": edges,
                "start_cell": list(sample.start_cell),
            },
            "solution": {
                "path_cells": [list(cell) for cell in sample.solution_path],
                "cell_count": len(sample.solution_path),
            },
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
        width, height = sample.resolution
        cell_size, cell_centers = self._layout(sample.occupied_cells, width, height)
        segments = self._build_segments(sample.solution_path, cell_centers)

        frames: list[np.ndarray] = []
        for frame_index in range(sample.frame_count):
            progress = frame_index / max(sample.frame_count - 1, 1)
            visited_cells, drawn_segments, partial_segment = self._progress_state(sample.solution_path, segments, progress)
            frame = self._render_board_state(
                sample=sample,
                cell_centers=cell_centers,
                cell_size=cell_size,
                visited_cells=visited_cells,
                drawn_segments=drawn_segments,
                partial_segment=partial_segment,
            )
            frames.append(frame)
        return frames

    def _layout(self, occupied_cells: list[Cell], width: int, height: int) -> tuple[int, dict[Cell, tuple[int, int]]]:
        xs = [cell[0] for cell in occupied_cells]
        ys = [cell[1] for cell in occupied_cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cols = max_x - min_x + 1
        rows = max_y - min_y + 1
        margin = int(self.style.get("board_margin", 24))
        usable_w = width - margin * 2
        usable_h = height - margin * 2
        cell_size = max(10, int(min(usable_w / max(cols, 1), usable_h / max(rows, 1))))
        board_w = cols * cell_size
        board_h = rows * cell_size
        origin_x = int((width - board_w) / 2)
        origin_y = int((height - board_h) / 2)

        cell_centers: dict[Cell, tuple[int, int]] = {}
        for cell in occupied_cells:
            shifted_x = cell[0] - min_x
            shifted_y = cell[1] - min_y
            center_x = origin_x + shifted_x * cell_size + cell_size // 2
            center_y = origin_y + shifted_y * cell_size + cell_size // 2
            cell_centers[cell] = (center_x, center_y)
        return cell_size, cell_centers

    def _render_board_state(
        self,
        sample: PuzzleSpec,
        cell_centers: dict[Cell, tuple[int, int]],
        cell_size: int,
        visited_cells: set[Cell],
        drawn_segments: list[tuple[tuple[int, int], tuple[int, int]]],
        partial_segment: tuple[tuple[int, int], tuple[int, int]] | None,
    ) -> np.ndarray:
        width, height = sample.resolution
        colors = self._style_colors()
        frame = Image.new("RGBA", (width, height), colors["background"] + (255,))
        draw = ImageDraw.Draw(frame)
        radius = int(self.style.get("tile_radius", 8))

        for cell in sample.occupied_cells:
            center = cell_centers[cell]
            if cell == sample.start_cell or cell in visited_cells:
                color = colors["active_tile"]
                alpha = 235
            else:
                color = colors["inactive_tile"]
                alpha = 210
            self._draw_rounded_rect(draw, center, cell_size, cell_size, radius, color, alpha=alpha)

        trace_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        trace_draw = ImageDraw.Draw(trace_overlay)
        thickness = max(6, int(cell_size * 0.52))
        glow = thickness + 10

        for start, end in drawn_segments:
            trace_draw.line([start, end], fill=colors["trace_glow"] + (120,), width=glow)
            trace_draw.line([start, end], fill=colors["trace"] + (255,), width=thickness)

        if partial_segment is not None:
            start, current_point = partial_segment
            trace_draw.line([start, current_point], fill=colors["trace_glow"] + (120,), width=glow)
            trace_draw.line([start, current_point], fill=colors["trace"] + (255,), width=thickness)
            self._draw_circle(trace_draw, current_point, max(4, thickness // 3), colors["trace_head"] + (255,))

        frame = Image.alpha_composite(frame, trace_overlay)
        draw = ImageDraw.Draw(frame)
        start_center = cell_centers[sample.start_cell]
        configured_node_radius = int(self.style.get("node_radius", 14))
        node_radius_ratio = float(self.style.get("node_radius_ratio", 0.26))
        node_radius = max(6, min(configured_node_radius, int(cell_size * node_radius_ratio)))
        glow_radius = min(node_radius + max(3, cell_size // 12), max(node_radius + 1, int(cell_size * 0.42)))
        self._draw_circle(draw, start_center, glow_radius, colors["node_glow"] + (110,))
        self._draw_circle(draw, start_center, node_radius, colors["node"] + (255,))
        self._draw_circle(draw, start_center, max(3, int(node_radius * 0.38)), colors["start_core"] + (255,))

        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    def _build_segments(
        self,
        solution_path: list[Cell],
        cell_centers: dict[Cell, tuple[int, int]],
    ) -> list[tuple[float, float, Cell, tuple[int, int], tuple[int, int]]]:
        if len(solution_path) < 2:
            return []
        lengths: list[float] = []
        total_length = 0.0
        for current, nxt in zip(solution_path[:-1], solution_path[1:]):
            start = cell_centers[current]
            end = cell_centers[nxt]
            length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            lengths.append(length)
            total_length += length

        traversed = 0.0
        segments: list[tuple[float, float, Cell, tuple[int, int], tuple[int, int]]] = []
        for current, nxt, length in zip(solution_path[:-1], solution_path[1:], lengths):
            start_ratio = traversed / max(total_length, 1.0)
            traversed += length
            segments.append(
                (start_ratio, traversed / max(total_length, 1.0), nxt, cell_centers[current], cell_centers[nxt])
            )
        return segments

    def _progress_state(
        self,
        solution_path: list[Cell],
        segments: list[tuple[float, float, Cell, tuple[int, int], tuple[int, int]]],
        progress: float,
    ) -> tuple[set[Cell], list[tuple[tuple[int, int], tuple[int, int]]], tuple[tuple[int, int], tuple[int, int]] | None]:
        visited = {solution_path[0]}
        drawn_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        partial_segment: tuple[tuple[int, int], tuple[int, int]] | None = None

        for start_ratio, end_ratio, destination, start, end in segments:
            if progress >= end_ratio:
                drawn_segments.append((start, end))
                visited.add(destination)
                continue

            local_progress = (progress - start_ratio) / max(end_ratio - start_ratio, 1e-6)
            local_progress = float(np.clip(local_progress, 0.0, 1.0))
            current_point = (
                int(round(start[0] + (end[0] - start[0]) * local_progress)),
                int(round(start[1] + (end[1] - start[1]) * local_progress)),
            )
            partial_segment = (start, current_point)
            if local_progress >= 0.999:
                visited.add(destination)
            break

        return visited, drawn_segments, partial_segment

    def _style_colors(self) -> dict[str, tuple[int, int, int]]:
        return {
            "background": (51, 63, 74),
            "inactive_tile": (117, 118, 113),
            "active_tile": (84, 194, 94),
            "node_glow": (84, 194, 94),
            "node": (52, 170, 66),
            "start_core": (240, 246, 244),
            "trace_glow": (178, 244, 182),
            "trace": (96, 219, 106),
            "trace_head": (236, 248, 241),
        }

    def _draw_rounded_rect(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        width: int,
        height: int,
        radius: int,
        color: tuple[int, int, int],
        alpha: int = 215,
    ) -> None:
        cx, cy = center
        x1 = cx - width // 2
        y1 = cy - height // 2
        x2 = cx + width // 2
        y2 = cy + height // 2
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=color + (alpha,))

    def _draw_circle(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int, int],
    ) -> None:
        cx, cy = center
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)

    def _in_bounds(self, cell: Cell, board_size: tuple[int, int]) -> bool:
        return 0 <= cell[0] < board_size[0] and 0 <= cell[1] < board_size[1]


def _generate_split_task(task: dict[str, Any]) -> dict[str, Any]:
    env = OneLineDrawingEnv(task["env_config"])
    entries = env.generate_split(task["output_root"], int(task["level"]), str(task["split"]))
    return {"level": int(task["level"]), "split": str(task["split"]), "entries": entries}
