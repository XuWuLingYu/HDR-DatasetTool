#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


Cell = tuple[int, int]


@dataclass(frozen=True)
class Config:
    num_videos: int = 3000
    push_values: tuple[int, ...] = (1, 2, 3)
    count_per_push: int = 1000
    n_values: tuple[int, ...] = (5, 6, 7)
    fps: int = 10
    frames_per_move: int = 5
    tile_size: int = 16
    ball_radius: int = 5
    block_half_size: int = 5
    frame_size: int = 208
    block_delay_frames: float = 2.0
    block_push_fraction: float = 0.5
    output_dir: str = "maze_videos_recursive_division_push_blocks"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260515
    workers: int = 1
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7
    max_attempts: int = 512

    def validate(self) -> None:
        if self.num_videos < 1:
            raise ValueError("num_videos must be >= 1")
        if not self.push_values:
            raise ValueError("push_values must be non-empty")
        if any(value < 1 for value in self.push_values):
            raise ValueError("all push_values must be >= 1")
        if self.count_per_push < 1:
            raise ValueError("count_per_push must be >= 1")
        if not self.n_values:
            raise ValueError("n_values must be non-empty")
        if any(value < 4 for value in self.n_values):
            raise ValueError("all n_values must be >= 4")
        if self.fps < 1:
            raise ValueError("fps must be >= 1")
        if self.frames_per_move < 1:
            raise ValueError("frames_per_move must be >= 1")
        if self.tile_size < 4:
            raise ValueError("tile_size must be >= 4")
        if self.ball_radius < 2:
            raise ValueError("ball_radius must be >= 2")
        if self.block_half_size < 2:
            raise ValueError("block_half_size must be >= 2")
        if self.frame_size < 64:
            raise ValueError("frame_size must be >= 64")
        if not (0.0 <= self.block_delay_frames < self.frames_per_move):
            raise ValueError("block_delay_frames must be in [0, frames_per_move)")
        if not (0.0 < self.block_push_fraction <= 1.0):
            raise ValueError("block_push_fraction must be in (0, 1]")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        expected = len(self.push_values) * self.count_per_push
        if self.num_videos != expected:
            raise ValueError("num_videos must equal len(push_values) * count_per_push")


@dataclass(frozen=True)
class VideoTask:
    index: int
    maze_size: int
    required_pushes: int
    seed: int


@dataclass(frozen=True)
class CandidateBlocker:
    path_index: int
    approach: Cell
    blocked_cell: Cell
    pocket: Cell
    exit_cell: Cell


def cell_neighbors(cell: Cell, n: int) -> list[Cell]:
    row, col = cell
    out: list[Cell] = []
    if row > 0:
        out.append((row - 1, col))
    if row + 1 < n:
        out.append((row + 1, col))
    if col > 0:
        out.append((row, col - 1))
    if col + 1 < n:
        out.append((row, col + 1))
    return out


def add_edge(graph: dict[Cell, list[Cell]], a: Cell, b: Cell) -> None:
    graph[a].append(b)
    graph[b].append(a)


def add_edge_unique(graph: dict[Cell, list[Cell]], a: Cell, b: Cell) -> None:
    if b not in graph[a]:
        graph[a].append(b)
    if a not in graph[b]:
        graph[b].append(a)


def generate_fully_connected_grid(n: int) -> dict[Cell, list[Cell]]:
    graph: dict[Cell, list[Cell]] = {(r, c): [] for r in range(n) for c in range(n)}
    for row in range(n):
        for col in range(n):
            cell = (row, col)
            if row + 1 < n:
                add_edge(graph, cell, (row + 1, col))
            if col + 1 < n:
                add_edge(graph, cell, (row, col + 1))
    return graph


def remove_edge(graph: dict[Cell, list[Cell]], a: Cell, b: Cell) -> None:
    if b in graph[a]:
        graph[a].remove(b)
    if a in graph[b]:
        graph[b].remove(a)


def clone_graph(graph: dict[Cell, list[Cell]]) -> dict[Cell, list[Cell]]:
    return {cell: neighbors[:] for cell, neighbors in graph.items()}


def recursive_divide(
    graph: dict[Cell, list[Cell]],
    left: int,
    top: int,
    right: int,
    bottom: int,
    rng: random.Random,
) -> None:
    if (right - left < 1) or (bottom - top < 1):
        return

    x = rng.randint(left, right - 1)
    y = rng.randint(top, bottom - 1)

    to_disconnect: list[tuple[Cell, Cell]] = []
    for col in range(left, right + 1):
        to_disconnect.append(((y, col), (y + 1, col)))
    for row in range(top, bottom + 1):
        to_disconnect.append(((row, x), (row, x + 1)))

    closed_side = rng.randint(0, 3)
    gap_positions = {
        0: rng.randint(top, y),
        1: rng.randint(y + 1, bottom),
        2: rng.randint(left, x),
        3: rng.randint(x + 1, right),
    }
    gap_edges = {
        0: ((gap_positions[0], x), (gap_positions[0], x + 1)),
        1: ((gap_positions[1], x), (gap_positions[1], x + 1)),
        2: ((y, gap_positions[2]), (y + 1, gap_positions[2])),
        3: ((y, gap_positions[3]), (y + 1, gap_positions[3])),
    }
    keep_edges = {gap_edges[side] for side in range(4) if side != closed_side}
    for edge in to_disconnect:
        if edge not in keep_edges:
            remove_edge(graph, edge[0], edge[1])

    recursive_divide(graph, left, top, x, y, rng)
    recursive_divide(graph, x + 1, top, right, y, rng)
    recursive_divide(graph, left, y + 1, x, bottom, rng)
    recursive_divide(graph, x + 1, y + 1, right, bottom, rng)


def generate_recursive_division_maze(n: int, rng: random.Random) -> dict[Cell, list[Cell]]:
    graph = generate_fully_connected_grid(n)
    recursive_divide(graph, 0, 0, n - 1, n - 1, rng)
    return graph


def solve_path(graph: dict[Cell, list[Cell]], start: Cell, goal: Cell) -> list[Cell]:
    queue: list[Cell] = [start]
    parent: dict[Cell, Cell | None] = {start: None}
    head = 0

    while head < len(queue):
        current = queue[head]
        head += 1
        if current == goal:
            break
        for nxt in graph[current]:
            if nxt not in parent:
                parent[nxt] = current
                queue.append(nxt)

    if goal not in parent:
        raise RuntimeError("maze unexpectedly unsolved")

    path: list[Cell] = []
    node: Cell | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def graph_to_wall_grid(graph: dict[Cell, list[Cell]], n: int) -> np.ndarray:
    grid = np.ones((2 * n + 1, 2 * n + 1), dtype=np.uint8)
    for row in range(n):
        for col in range(n):
            grid[2 * row + 1, 2 * col + 1] = 0
            for nr, nc in graph[(row, col)]:
                grid[row + nr + 1, col + nc + 1] = 0
    return grid


def grid_to_background(grid: np.ndarray, tile_size: int, frame_size: int) -> np.ndarray:
    base = np.where(grid == 1, 0, 255).astype(np.uint8)
    expanded = np.asarray(
        Image.fromarray(base, mode="L").resize((frame_size, frame_size), resample=Image.Resampling.NEAREST)
    )
    return np.repeat(expanded[..., None], 3, axis=2)


def cell_center(cell: Cell, maze_size: int, frame_size: int) -> tuple[float, float]:
    row, col = cell
    scale = frame_size / (2 * maze_size + 1)
    center_y = (2 * row + 1.5) * scale - 0.5
    center_x = (2 * col + 1.5) * scale - 0.5
    return center_y, center_x


def make_ball_sprite(radius: int) -> tuple[np.ndarray, np.ndarray]:
    diameter = 2 * radius + 1
    yy, xx = np.ogrid[:diameter, :diameter]
    dist_sq = (yy - radius) ** 2 + (xx - radius) ** 2
    mask = dist_sq <= radius**2
    sprite = np.zeros((diameter, diameter, 3), dtype=np.uint8)
    sprite[..., 0] = 255
    return sprite, mask


def make_block_sprite(half_size: int) -> np.ndarray:
    side = 2 * half_size + 1
    sprite = np.zeros((side, side, 3), dtype=np.uint8)
    sprite[..., 0] = 140
    sprite[..., 1] = 140
    sprite[..., 2] = 140
    return sprite


def effective_ball_radius(config: Config, maze_size: int) -> int:
    native_side = (2 * maze_size + 1) * config.tile_size
    scaled_radius = config.ball_radius * config.frame_size / native_side
    return max(2, int(round(scaled_radius)))


def effective_block_half_size(config: Config, maze_size: int) -> int:
    native_side = (2 * maze_size + 1) * config.tile_size
    scaled_half = config.block_half_size * config.frame_size / native_side
    return max(2, int(round(scaled_half)))


def place_square(frame: np.ndarray, sprite: np.ndarray, center_y: float, center_x: float) -> None:
    half = sprite.shape[0] // 2
    top = int(round(center_y)) - half
    left = int(round(center_x)) - half
    bottom = top + sprite.shape[0]
    right = left + sprite.shape[1]
    frame[top:bottom, left:right] = sprite


def place_ball(frame: np.ndarray, sprite: np.ndarray, mask: np.ndarray, center_y: float, center_x: float) -> None:
    radius = mask.shape[0] // 2
    top = int(round(center_y)) - radius
    left = int(round(center_x)) - radius
    bottom = top + mask.shape[0]
    right = left + mask.shape[1]
    region = frame[top:bottom, left:right]
    region[mask] = sprite[mask]


def render_frame(
    background: np.ndarray,
    maze_size: int,
    block_cells: tuple[Cell, ...],
    ball_center: tuple[float, float],
    block_centers: dict[Cell, tuple[float, float]],
    ball_sprite: np.ndarray,
    ball_mask: np.ndarray,
    block_sprite: np.ndarray,
) -> np.ndarray:
    frame = background.copy()
    for block_cell in block_cells:
        center_y, center_x = block_centers[block_cell]
        place_square(frame, block_sprite, center_y, center_x)
    place_ball(frame, ball_sprite, ball_mask, ball_center[0], ball_center[1])
    return frame


def find_push_candidates(graph: dict[Cell, list[Cell]], path: list[Cell], n: int) -> list[CandidateBlocker]:
    path_set = set(path)
    candidates: list[CandidateBlocker] = []
    for index in range(1, len(path) - 1):
        approach = path[index - 1]
        blocked = path[index]
        exit_cell = path[index + 1]
        dr = blocked[0] - approach[0]
        dc = blocked[1] - approach[1]
        pocket = (blocked[0] + dr, blocked[1] + dc)
        if not (0 <= pocket[0] < n and 0 <= pocket[1] < n):
            continue
        if pocket == exit_cell:
            continue
        if pocket in path_set:
            continue
        if pocket in graph[blocked]:
            continue
        candidates.append(
            CandidateBlocker(
                path_index=index,
                approach=approach,
                blocked_cell=blocked,
                pocket=pocket,
                exit_cell=exit_cell,
            )
        )
    return candidates


def choose_disjoint_candidates(
    candidates: list[CandidateBlocker],
    required_pushes: int,
    rng: random.Random,
) -> list[CandidateBlocker] | None:
    if len(candidates) < required_pushes:
        return None

    for _ in range(64):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        chosen: list[CandidateBlocker] = []
        used_cells: set[Cell] = set()
        for candidate in shuffled:
            pair_cells = {candidate.blocked_cell, candidate.pocket}
            if pair_cells & used_cells:
                continue
            if any(abs(candidate.path_index - prev.path_index) <= 1 for prev in chosen):
                continue
            chosen.append(candidate)
            used_cells |= pair_cells
            if len(chosen) == required_pushes:
                chosen.sort(key=lambda item: item.path_index)
                return chosen
    return None


def carve_dead_end(graph: dict[Cell, list[Cell]], candidate: CandidateBlocker) -> None:
    add_edge_unique(graph, candidate.blocked_cell, candidate.pocket)
    for neighbor in graph[candidate.pocket][:]:
        if neighbor != candidate.blocked_cell:
            remove_edge(graph, candidate.pocket, neighbor)


def canonical_blocks(blocks: set[Cell]) -> tuple[Cell, ...]:
    return tuple(sorted(blocks))


def solve_push_puzzle(
    graph: dict[Cell, list[Cell]],
    start: Cell,
    goal: Cell,
    initial_blocks: tuple[Cell, ...],
) -> dict[str, object] | None:
    start_state = (start, initial_blocks)
    dist: dict[tuple[Cell, tuple[Cell, ...]], tuple[int, int]] = {start_state: (0, 0)}
    parent: dict[tuple[Cell, tuple[Cell, ...]], tuple[tuple[Cell, tuple[Cell, ...]], dict[str, object]] | None] = {
        start_state: None
    }
    heap: list[tuple[int, int, Cell, tuple[Cell, ...]]] = [(0, 0, start, initial_blocks)]

    while heap:
        pushes, steps, player, blocks = heapq.heappop(heap)
        state = (player, blocks)
        if (pushes, steps) != dist[state]:
            continue
        if player == goal:
            actions: list[dict[str, object]] = []
            cursor = state
            while parent[cursor] is not None:
                prev_state, action = parent[cursor]
                actions.append(action)
                cursor = prev_state
            actions.reverse()
            return {
                "required_pushes": pushes,
                "steps": steps,
                "actions": actions,
                "final_blocks": [list(cell) for cell in blocks],
            }

        block_set = set(blocks)
        for nxt in graph[player]:
            dr = nxt[0] - player[0]
            dc = nxt[1] - player[1]
            if nxt in block_set:
                dest = (nxt[0] + dr, nxt[1] + dc)
                if dest not in graph[nxt] or dest in block_set:
                    continue
                new_blocks = set(block_set)
                new_blocks.remove(nxt)
                new_blocks.add(dest)
                next_blocks = canonical_blocks(new_blocks)
                next_state = (nxt, next_blocks)
                candidate_cost = (pushes + 1, steps + 1)
                action = {
                    "type": "push",
                    "player_from": list(player),
                    "player_to": list(nxt),
                    "block_from": list(nxt),
                    "block_to": list(dest),
                }
            else:
                next_state = (nxt, blocks)
                candidate_cost = (pushes, steps + 1)
                action = {
                    "type": "move",
                    "player_from": list(player),
                    "player_to": list(nxt),
                }

            if next_state not in dist or candidate_cost < dist[next_state]:
                dist[next_state] = candidate_cost
                parent[next_state] = (state, action)
                heapq.heappush(heap, (candidate_cost[0], candidate_cost[1], next_state[0], next_state[1]))
    return None


def generate_push_block_sample(config: Config, rng: random.Random, maze_size: int, required_pushes: int) -> dict[str, object]:
    start = (0, 0)
    goal = (maze_size - 1, maze_size - 1)

    for _ in range(config.max_attempts):
        original_graph = generate_recursive_division_maze(maze_size, rng)
        base_path = solve_path(original_graph, start, goal)
        candidates = find_push_candidates(original_graph, base_path, maze_size)
        chosen = choose_disjoint_candidates(candidates, required_pushes, rng)
        if chosen is None:
            continue

        graph = clone_graph(original_graph)
        for candidate in chosen:
            carve_dead_end(graph, candidate)

        initial_blocks = canonical_blocks({candidate.blocked_cell for candidate in chosen})
        solution = solve_push_puzzle(graph, start, goal, initial_blocks)
        if solution is None:
            continue
        if int(solution["required_pushes"]) != required_pushes:
            continue

        return {
            "graph": graph,
            "base_path": base_path,
            "initial_blocks": initial_blocks,
            "solution": solution,
            "gadget_cells": chosen,
        }

    raise RuntimeError(
        f"failed to construct a {maze_size}x{maze_size} push-maze requiring {required_pushes} pushes"
    )


def interpolate(a: tuple[float, float], b: tuple[float, float], alpha: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha)


def iter_frame_states(
    actions: list[dict[str, object]],
    initial_blocks: tuple[Cell, ...],
    maze_size: int,
    frame_size: int,
    frames_per_move: int,
    block_delay_frames: float,
    block_push_fraction: float,
    ball_radius_px: int,
    block_half_px: int,
) -> list[tuple[tuple[float, float], tuple[Cell, ...], dict[Cell, tuple[float, float]]]]:
    frames: list[tuple[tuple[float, float], tuple[Cell, ...], dict[Cell, tuple[float, float]]]] = []
    player_cell = tuple(actions[0]["player_from"]) if actions else (0, 0)
    block_cells = tuple(initial_blocks)
    player_center = cell_center(player_cell, maze_size, frame_size)
    static_centers = {cell: cell_center(cell, maze_size, frame_size) for cell in block_cells}
    frames.append((player_center, block_cells, static_centers))

    current_blocks = list(block_cells)
    current_block_centers = {cell: cell_center(cell, maze_size, frame_size) for cell in current_blocks}
    current_player = player_cell
    current_player_center = player_center
    for action in actions:
        next_player = tuple(action["player_to"])
        start_player_center = current_player_center
        end_player_center = cell_center(next_player, maze_size, frame_size)
        block_from = tuple(action["block_from"]) if action["type"] == "push" else None
        block_to = tuple(action["block_to"]) if action["type"] == "push" else None

        if block_from is None or block_to is None:
            for step in range(1, frames_per_move + 1):
                alpha = step / frames_per_move
                frame_player_center = interpolate(start_player_center, end_player_center, alpha)
                frame_block_centers = {cell: current_block_centers[cell] for cell in current_blocks}
                frames.append((frame_player_center, tuple(current_blocks), frame_block_centers))
            current_player_center = end_player_center
        else:
            block_start_center = current_block_centers[block_from]
            block_end_center = interpolate(
                cell_center(block_from, maze_size, frame_size),
                cell_center(block_to, maze_size, frame_size),
                block_push_fraction,
            )
            for step in range(1, frames_per_move + 1):
                player_alpha = step / frames_per_move
                if step <= block_delay_frames:
                    block_alpha = 0.0
                else:
                    block_alpha = (step - block_delay_frames) / max(1, frames_per_move - block_delay_frames)
                moving_block_center = interpolate(block_start_center, block_end_center, block_alpha)
                frame_player_center = interpolate(start_player_center, end_player_center, player_alpha)
                frame_block_centers: dict[Cell, tuple[float, float]] = {}
                for cell in current_blocks:
                    if cell == block_from:
                        frame_block_centers[cell] = moving_block_center
                    else:
                        frame_block_centers[cell] = current_block_centers[cell]
                frames.append((frame_player_center, tuple(current_blocks), frame_block_centers))
            current_player_center = end_player_center

        if block_from is not None and block_to is not None:
            current_blocks[current_blocks.index(block_from)] = block_to
            del current_block_centers[block_from]
            current_block_centers[block_to] = block_end_center
        current_player = next_player

    return frames


def video_path_for(output_dir: Path, index: int) -> Path:
    return output_dir / f"maze_push_{index:04d}.mp4"


def npy_path_for(output_dir: Path, index: int) -> Path:
    return output_dir / f"maze_push_{index:04d}.npy"


def metadata_path_for(output_dir: Path, name: str) -> Path:
    return output_dir / name


def balanced_counts(total: int, num_buckets: int, offset: int) -> list[int]:
    base = total // num_buckets
    remainder = total % num_buckets
    counts = [base] * num_buckets
    for delta in range(remainder):
        counts[(offset + delta) % num_buckets] += 1
    return counts


def build_video_tasks(config: Config) -> list[VideoTask]:
    tasks: list[VideoTask] = []
    index = 0
    for push_offset, required_pushes in enumerate(config.push_values):
        per_n_counts = balanced_counts(config.count_per_push, len(config.n_values), push_offset)
        for n_value, count in zip(config.n_values, per_n_counts):
            for _ in range(count):
                tasks.append(
                    VideoTask(
                        index=index,
                        maze_size=n_value,
                        required_pushes=required_pushes,
                        seed=config.seed + index,
                    )
                )
                index += 1
    return tasks


def create_single_video(task: VideoTask, config: Config) -> dict[str, object]:
    rng = random.Random(task.seed)
    output_dir = Path(config.output_dir)
    video_path = video_path_for(output_dir, task.index)
    npy_path = npy_path_for(output_dir, task.index)
    if video_path.exists() and npy_path.exists() and not config.overwrite:
        return {
            "index": task.index,
            "seed": task.seed,
            "file": video_path.name,
            "npy_file": npy_path.name,
            "maze_size": task.maze_size,
            "required_pushes": task.required_pushes,
            "skipped_existing": True,
        }

    sample = generate_push_block_sample(config, rng, task.maze_size, task.required_pushes)
    graph = sample["graph"]
    initial_blocks = sample["initial_blocks"]
    solution = sample["solution"]
    actions = solution["actions"]
    ball_radius_px = effective_ball_radius(config, task.maze_size)
    block_half_px = effective_block_half_size(config, task.maze_size)
    frames = iter_frame_states(
        actions,
        initial_blocks,
        task.maze_size,
        config.frame_size,
        config.frames_per_move,
        config.block_delay_frames,
        config.block_push_fraction,
        ball_radius_px,
        block_half_px,
    )

    grid = graph_to_wall_grid(graph, task.maze_size)
    background = grid_to_background(grid, config.tile_size, config.frame_size)
    ball_sprite, ball_mask = make_ball_sprite(ball_radius_px)
    block_sprite = make_block_sprite(block_half_px)

    with imageio.get_writer(
        video_path,
        fps=config.fps,
        codec=config.codec,
        quality=config.quality,
        macro_block_size=1,
    ) as writer:
        for player_center, block_cells, block_centers in frames:
            frame = render_frame(
                background,
                task.maze_size,
                block_cells,
                player_center,
                block_centers,
                ball_sprite,
                ball_mask,
                block_sprite,
            )
            writer.append_data(frame)

    np.save(
        npy_path,
        {
            "maze_size": task.maze_size,
            "required_pushes": task.required_pushes,
            "start": [0, 0],
            "goal": [task.maze_size - 1, task.maze_size - 1],
            "graph": {f"{cell[0]},{cell[1]}": [list(nei) for nei in neighbors] for cell, neighbors in graph.items()},
            "base_path": [list(cell) for cell in sample["base_path"]],
            "initial_blocks": [list(cell) for cell in initial_blocks],
            "gadget_cells": [
                {
                    "path_index": item.path_index,
                    "approach": list(item.approach),
                    "blocked_cell": list(item.blocked_cell),
                    "pocket": list(item.pocket),
                    "exit_cell": list(item.exit_cell),
                    "carved_from_wall": True,
                }
                for item in sample["gadget_cells"]
            ],
            "solution_actions": actions,
            "final_blocks": solution["final_blocks"],
        },
        allow_pickle=True,
    )

    return {
        "index": task.index,
        "seed": task.seed,
        "file": video_path.name,
        "npy_file": npy_path.name,
        "algorithm": "recursive_division_push_blocks",
        "maze_size": task.maze_size,
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
        "fps": config.fps,
        "frames_per_move": config.frames_per_move,
        "block_delay_frames": config.block_delay_frames,
        "block_push_fraction": config.block_push_fraction,
        "num_frames": len(frames),
        "required_pushes": task.required_pushes,
        "minimal_pushes": int(solution["required_pushes"]),
        "solution_steps": int(solution["steps"]),
        "start": [0, 0],
        "goal": [task.maze_size - 1, task.maze_size - 1],
        "base_path": [list(cell) for cell in sample["base_path"]],
        "initial_blocks": [list(cell) for cell in initial_blocks],
        "final_blocks": solution["final_blocks"],
        "carved_dead_ends": [
            {
                "path_index": item.path_index,
                "approach": list(item.approach),
                "blocked_cell": list(item.blocked_cell),
                "pocket": list(item.pocket),
                "exit_cell": list(item.exit_cell),
            }
            for item in sample["gadget_cells"]
        ],
        "solution_actions": actions,
        "skipped_existing": False,
    }


def run_generation(config: Config) -> None:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path_for(output_dir, config.metadata_name)
    tasks = build_video_tasks(config)

    records: list[dict[str, object]] = []
    if config.workers == 1:
        for task in tqdm(tasks, total=len(tasks), desc="Generating push-block mazes"):
            records.append(create_single_video(task, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            iterator = executor.map(create_single_video, tasks, repeat(config), chunksize=1)
            for record in tqdm(iterator, total=len(tasks), desc="Generating push-block mazes"):
                records.append(record)

    records.sort(key=lambda item: int(item["index"]))
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of integers")
    return tuple(int(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate recursive-division mazes with pushable gray blocks.")
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--push-values", type=str, default="1,2,3")
    parser.add_argument("--count-per-push", type=int, default=1000)
    parser.add_argument("--n-values", type=str, default="5,6,7")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frames-per-move", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--ball-radius", type=int, default=5)
    parser.add_argument("--block-half-size", type=int, default=5)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--block-delay-frames", type=float, default=2.0)
    parser.add_argument("--block-push-fraction", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="maze_videos_recursive_division_push_blocks")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--max-attempts", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Config(
        num_videos=args.num_videos,
        push_values=parse_int_list(args.push_values),
        count_per_push=args.count_per_push,
        n_values=parse_int_list(args.n_values),
        fps=args.fps,
        frames_per_move=args.frames_per_move,
        tile_size=args.tile_size,
        ball_radius=args.ball_radius,
        block_half_size=args.block_half_size,
        frame_size=args.frame_size,
        block_delay_frames=args.block_delay_frames,
        block_push_fraction=args.block_push_fraction,
        output_dir=args.output_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        workers=args.workers,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    run_generation(config)


if __name__ == "__main__":
    main()
