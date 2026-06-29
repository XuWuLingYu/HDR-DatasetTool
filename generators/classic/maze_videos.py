#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


Cell = tuple[int, int]


@dataclass(frozen=True)
class Config:
    n: int = 6
    num_videos: int = 1000
    n_values: tuple[int, ...] = ()
    n_counts: tuple[int, ...] = ()
    shuffle_n_values: bool = True
    algorithm: str = "candidate_dfs"
    fps: int = 10
    frames_per_move: int = 5
    tile_size: int = 16
    ball_radius: int = 5
    frame_size: int | None = None
    output_dir: str = "outputs/maze/videos"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260409
    workers: int = 1
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7
    candidate_mazes: int = 8
    top_candidates: int = 3
    branch_weight: float = 0.35
    loop_weight: float = 0.25

    def validate(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2")
        if self.algorithm not in {"candidate_dfs", "recursive_division"}:
            raise ValueError("algorithm must be one of: candidate_dfs, recursive_division")
        if self.num_videos < 1:
            raise ValueError("num_videos must be >= 1")
        if self.n_values and any(n < 2 for n in self.n_values):
            raise ValueError("all n_values must be >= 2")
        if self.n_counts and any(count < 1 for count in self.n_counts):
            raise ValueError("all n_counts must be >= 1")
        if bool(self.n_values) != bool(self.n_counts):
            raise ValueError("n_values and n_counts must be provided together")
        if self.n_values and len(self.n_values) != len(self.n_counts):
            raise ValueError("n_values and n_counts must have the same length")
        if self.n_values and sum(self.n_counts) != self.num_videos:
            raise ValueError("sum(n_counts) must equal num_videos")
        if self.fps < 1:
            raise ValueError("fps must be >= 1")
        if self.frames_per_move < 1:
            raise ValueError("frames_per_move must be >= 1")
        if self.tile_size < 4:
            raise ValueError("tile_size must be >= 4")
        if self.ball_radius < 2:
            raise ValueError("ball_radius must be >= 2")
        if self.ball_radius >= self.tile_size // 2 and self.frame_size is None:
            raise ValueError("ball_radius must be smaller than half of tile_size")
        if self.frame_size is not None and self.frame_size < 32:
            raise ValueError("frame_size must be >= 32 when provided")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.candidate_mazes < 1:
            raise ValueError("candidate_mazes must be >= 1")
        if self.top_candidates < 1:
            raise ValueError("top_candidates must be >= 1")
        if self.top_candidates > self.candidate_mazes:
            raise ValueError("top_candidates must be <= candidate_mazes")


@dataclass(frozen=True)
class VideoTask:
    index: int
    maze_size: int
    seed: int


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


def has_edge(graph: dict[Cell, list[Cell]], a: Cell, b: Cell) -> bool:
    return b in graph[a]


def graph_edge_count(graph: dict[Cell, list[Cell]]) -> int:
    return sum(len(neighbors) for neighbors in graph.values()) // 2


def all_adjacent_pairs(n: int) -> list[tuple[Cell, Cell]]:
    pairs: list[tuple[Cell, Cell]] = []
    for row in range(n):
        for col in range(n):
            cell = (row, col)
            if row + 1 < n:
                pairs.append((cell, (row + 1, col)))
            if col + 1 < n:
                pairs.append((cell, (row, col + 1)))
    return pairs


def generate_perfect_maze(n: int, rng: random.Random) -> dict[Cell, list[Cell]]:
    graph: dict[Cell, list[Cell]] = {(r, c): [] for r in range(n) for c in range(n)}
    start = (0, 0)
    stack = [start]
    visited = {start}

    while stack:
        current = stack[-1]
        choices = [cell for cell in cell_neighbors(current, n) if cell not in visited]
        if not choices:
            stack.pop()
            continue

        nxt = rng.choice(choices)
        add_edge(graph, current, nxt)
        visited.add(nxt)
        stack.append(nxt)

    expected_edges = n * n - 1
    actual_edges = graph_edge_count(graph)
    if len(visited) != n * n or actual_edges != expected_edges:
        raise RuntimeError("Maze generation failed to create a spanning tree")
    return graph


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


def add_random_cycles(graph: dict[Cell, list[Cell]], n: int, rng: random.Random) -> int:
    available = [(a, b) for a, b in all_adjacent_pairs(n) if not has_edge(graph, a, b)]
    if not available:
        return 0
    max_extra_edges = min(len(available), max(1, n - 1))
    extra_edges = rng.randint(1, max_extra_edges)
    rng.shuffle(available)
    for a, b in available[:extra_edges]:
        add_edge(graph, a, b)
    return extra_edges


def path_branch_stats(graph: dict[Cell, list[Cell]], path: list[Cell]) -> tuple[int, int]:
    branched_nodes = 0
    total_extra_edges = 0
    last_index = len(path) - 1
    for index, cell in enumerate(path):
        path_degree = 1 if index == 0 or index == last_index else 2
        extra_edges = max(0, len(graph[cell]) - path_degree)
        if extra_edges > 0:
            branched_nodes += 1
            total_extra_edges += extra_edges
    return branched_nodes, total_extra_edges


def evaluate_graph(
    graph: dict[Cell, list[Cell]],
    n: int,
    branch_weight: float,
    loop_weight: float,
    rng: random.Random,
) -> dict[str, object]:
    start = (0, 0)
    goal = (n - 1, n - 1)
    path = solve_path(graph, start, goal)
    branch_nodes, total_extra_edges = path_branch_stats(graph, path)
    cycle_edges = graph_edge_count(graph) - (n * n - 1)
    score = (
        len(path)
        + branch_weight * branch_nodes
        + 0.15 * total_extra_edges
        + loop_weight * cycle_edges
        + 0.05 * rng.random()
    )
    return {
        "graph": graph,
        "path": path,
        "branch_nodes": branch_nodes,
        "path_total_extra_edges": total_extra_edges,
        "cycle_edges": cycle_edges,
        "score": score,
    }


def generate_varied_maze(config: Config, rng: random.Random, maze_size: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for _ in range(config.candidate_mazes):
        graph = generate_perfect_maze(maze_size, rng)
        add_random_cycles(graph, maze_size, rng)
        candidates.append(
            evaluate_graph(
                graph=graph,
                n=maze_size,
                branch_weight=config.branch_weight,
                loop_weight=config.loop_weight,
                rng=rng,
            )
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    top_k = min(config.top_candidates, len(candidates))
    top_candidates = candidates[:top_k]
    weights = list(range(top_k, 0, -1))
    chosen = rng.choices(top_candidates, weights=weights, k=1)[0]
    chosen["candidate_rank"] = top_candidates.index(chosen) + 1
    chosen["candidate_pool_size"] = len(candidates)
    return chosen


def generate_recursive_division_sample(config: Config, rng: random.Random, maze_size: int) -> dict[str, object]:
    graph = generate_recursive_division_maze(maze_size, rng)
    path = solve_path(graph, (0, 0), (maze_size - 1, maze_size - 1))
    branch_nodes, total_extra_edges = path_branch_stats(graph, path)
    cycle_edges = graph_edge_count(graph) - (maze_size * maze_size - 1)
    return {
        "graph": graph,
        "path": path,
        "branch_nodes": branch_nodes,
        "path_total_extra_edges": total_extra_edges,
        "cycle_edges": cycle_edges,
        "candidate_rank": 1,
        "candidate_pool_size": 1,
        "score": float(len(path)),
    }


def solve_path(graph: dict[Cell, list[Cell]], start: Cell, goal: Cell) -> list[Cell]:
    queue: deque[Cell] = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nxt in graph[current]:
            if nxt not in parent:
                parent[nxt] = current
                queue.append(nxt)

    if goal not in parent:
        raise RuntimeError("Generated maze is unexpectedly unsolved")

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
                wall_row = row + nr + 1
                wall_col = col + nc + 1
                grid[wall_row, wall_col] = 0
    return grid


def render_side_pixels(config: Config, maze_size: int) -> int:
    if config.frame_size is not None:
        return config.frame_size
    return (2 * maze_size + 1) * config.tile_size


def grid_to_background(grid: np.ndarray, tile_size: int, frame_size: int | None) -> np.ndarray:
    base = np.where(grid == 1, 0, 255).astype(np.uint8)
    if frame_size is None:
        expanded = np.repeat(np.repeat(base, tile_size, axis=0), tile_size, axis=1)
    else:
        expanded = np.asarray(Image.fromarray(base, mode="L").resize((frame_size, frame_size), resample=Image.Resampling.NEAREST))
    return np.repeat(expanded[..., None], 3, axis=2)


def make_ball_sprite(radius: int) -> tuple[np.ndarray, np.ndarray]:
    diameter = 2 * radius + 1
    yy, xx = np.ogrid[:diameter, :diameter]
    dist_sq = (yy - radius) ** 2 + (xx - radius) ** 2
    mask = dist_sq <= radius**2
    sprite = np.zeros((diameter, diameter, 3), dtype=np.uint8)
    sprite[..., 0] = 255
    return sprite, mask


def effective_ball_radius(config: Config, maze_size: int) -> int:
    native_side = (2 * maze_size + 1) * config.tile_size
    output_side = render_side_pixels(config, maze_size)
    scaled_radius = config.ball_radius * output_side / native_side
    return max(2, int(round(scaled_radius)))


def cell_center(cell: Cell, maze_size: int, tile_size: int, frame_size: int | None) -> tuple[float, float]:
    row, col = cell
    logical_row = 2 * row + 1
    logical_col = 2 * col + 1
    if frame_size is None:
        center_y = logical_row * tile_size + (tile_size - 1) / 2.0
        center_x = logical_col * tile_size + (tile_size - 1) / 2.0
    else:
        scale = frame_size / (2 * maze_size + 1)
        center_y = (logical_row + 0.5) * scale - 0.5
        center_x = (logical_col + 0.5) * scale - 0.5
    return center_y, center_x


def iter_positions(
    path: list[Cell],
    maze_size: int,
    tile_size: int,
    frames_per_move: int,
    frame_size: int | None,
) -> Iterable[tuple[float, float]]:
    yield cell_center(path[0], maze_size, tile_size, frame_size)
    for current, nxt in zip(path, path[1:]):
        cur_y, cur_x = cell_center(current, maze_size, tile_size, frame_size)
        nxt_y, nxt_x = cell_center(nxt, maze_size, tile_size, frame_size)
        for step in range(1, frames_per_move + 1):
            alpha = step / frames_per_move
            yield (
                cur_y + (nxt_y - cur_y) * alpha,
                cur_x + (nxt_x - cur_x) * alpha,
            )


def render_frame(
    background: np.ndarray,
    ball_sprite: np.ndarray,
    ball_mask: np.ndarray,
    center_y: float,
    center_x: float,
) -> np.ndarray:
    frame = background.copy()
    radius = ball_mask.shape[0] // 2
    top = int(round(center_y)) - radius
    left = int(round(center_x)) - radius
    bottom = top + ball_mask.shape[0]
    right = left + ball_mask.shape[1]
    region = frame[top:bottom, left:right]
    region[ball_mask] = ball_sprite[ball_mask]
    return frame


def video_path_for(output_dir: Path, index: int) -> Path:
    return output_dir / f"maze_{index:04d}.mp4"


def metadata_path_for(output_dir: Path, name: str) -> Path:
    return output_dir / name


def build_video_tasks(config: Config) -> list[VideoTask]:
    if config.n_values:
        maze_sizes: list[int] = []
        for maze_size, count in zip(config.n_values, config.n_counts):
            maze_sizes.extend([maze_size] * count)
        if config.shuffle_n_values:
            shuffle_rng = random.Random(config.seed)
            shuffle_rng.shuffle(maze_sizes)
    else:
        maze_sizes = [config.n] * config.num_videos
    return [
        VideoTask(index=index, maze_size=maze_size, seed=config.seed + index)
        for index, maze_size in enumerate(maze_sizes)
    ]


def create_single_video(task: VideoTask, config: Config) -> dict[str, object]:
    rng = random.Random(task.seed)
    output_dir = Path(config.output_dir)
    video_path = video_path_for(output_dir, task.index)

    if config.algorithm == "recursive_division":
        maze = generate_recursive_division_sample(config, rng, task.maze_size)
    else:
        maze = generate_varied_maze(config, rng, task.maze_size)
    graph = maze["graph"]
    start = (0, 0)
    goal = (task.maze_size - 1, task.maze_size - 1)
    path = maze["path"]
    side_pixels = render_side_pixels(config, task.maze_size)
    record = {
        "index": task.index,
        "seed": task.seed,
        "file": video_path.name,
        "algorithm": config.algorithm,
        "maze_size": task.maze_size,
        "frame_height": side_pixels,
        "frame_width": side_pixels,
        "fps": config.fps,
        "frames_per_move": config.frames_per_move,
        "num_frames": 1 + (len(path) - 1) * config.frames_per_move,
        "path_length_cells": len(path),
        "path_branch_nodes": maze["branch_nodes"],
        "path_total_extra_edges": maze["path_total_extra_edges"],
        "cycle_edges": maze["cycle_edges"],
        "candidate_rank": maze["candidate_rank"],
        "candidate_pool_size": maze["candidate_pool_size"],
        "selection_score": round(float(maze["score"]), 4),
        "start": list(start),
        "goal": list(goal),
        "solution_path": [list(cell) for cell in path],
    }
    if video_path.exists() and not config.overwrite:
        record["skipped_existing"] = True
        return record

    grid = graph_to_wall_grid(graph, task.maze_size)
    background = grid_to_background(grid, config.tile_size, config.frame_size)
    ball_sprite, ball_mask = make_ball_sprite(effective_ball_radius(config, task.maze_size))

    with imageio.get_writer(
        video_path,
        fps=config.fps,
        codec=config.codec,
        quality=config.quality,
        macro_block_size=1,
    ) as writer:
        for center_y, center_x in iter_positions(
            path,
            task.maze_size,
            config.tile_size,
            config.frames_per_move,
            config.frame_size,
        ):
            frame = render_frame(background, ball_sprite, ball_mask, center_y, center_x)
            writer.append_data(frame)

    record["skipped_existing"] = False
    return record


def run_generation(config: Config) -> None:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path_for(output_dir, config.metadata_name)
    tasks = build_video_tasks(config)

    records: list[dict[str, object]] = []
    if config.workers == 1:
        for task in tqdm(tasks, total=len(tasks), desc="Generating videos"):
            records.append(create_single_video(task, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            iterator = executor.map(create_single_video, tasks, repeat(config), chunksize=1)
            for record in tqdm(iterator, total=len(tasks), desc="Generating videos"):
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
    parser = argparse.ArgumentParser(description="Generate a maze video dataset with a moving red ball.")
    parser.add_argument("--n", type=int, default=6, help="Maze size N for an N x N maze.")
    parser.add_argument("--num-videos", type=int, default=1000, help="Number of mp4 videos to generate.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="candidate_dfs",
        help="Maze generator: candidate_dfs or recursive_division.",
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="",
        help="Optional comma-separated maze sizes, for example 6,7,8.",
    )
    parser.add_argument(
        "--n-counts",
        type=str,
        default="",
        help="Optional comma-separated counts aligned with --n-values, for example 1000,1000,1000.",
    )
    parser.add_argument(
        "--keep-n-order",
        action="store_true",
        help="Keep the provided --n-values groups in order instead of shuffling them together.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Video frames per second.")
    parser.add_argument(
        "--frames-per-move",
        type=int,
        default=5,
        help="How many frames the ball spends moving across one maze cell.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Pixel size of each wall/open tile in the rendered 2N+1 maze grid.",
    )
    parser.add_argument("--ball-radius", type=int, default=5, help="Ball radius in pixels.")
    parser.add_argument(
        "--frame-size",
        type=int,
        default=None,
        help="Optional fixed square output size in pixels. When set, all videos share the same H and W.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/maze/videos", help="Directory for generated mp4 files.")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl", help="Metadata filename.")
    parser.add_argument("--seed", type=int, default=20260409, help="Base random seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use >1 to speed up large exports.",
    )
    parser.add_argument("--codec", type=str, default="libx264", help="Video codec passed to imageio.")
    parser.add_argument("--quality", type=int, default=7, help="imageio quality value from 0 to 10.")
    parser.add_argument(
        "--candidate-mazes",
        type=int,
        default=8,
        help="How many random maze candidates to sample before selecting one.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=3,
        help="Select randomly from the top-k scored candidates to keep variety.",
    )
    parser.add_argument(
        "--branch-weight",
        type=float,
        default=0.35,
        help="Soft preference for mazes whose chosen path has visible branches.",
    )
    parser.add_argument(
        "--loop-weight",
        type=float,
        default=0.25,
        help="Soft preference for mazes with extra loops, allowing multiple solutions.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_values = parse_int_list(args.n_values) if args.n_values else ()
    n_counts = parse_int_list(args.n_counts) if args.n_counts else ()
    config = Config(
        n=args.n,
        num_videos=args.num_videos,
        n_values=n_values,
        n_counts=n_counts,
        shuffle_n_values=not args.keep_n_order,
        algorithm=args.algorithm,
        fps=args.fps,
        frames_per_move=args.frames_per_move,
        tile_size=args.tile_size,
        ball_radius=args.ball_radius,
        frame_size=args.frame_size,
        output_dir=args.output_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        workers=args.workers,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
        candidate_mazes=args.candidate_mazes,
        top_candidates=args.top_candidates,
        branch_weight=args.branch_weight,
        loop_weight=args.loop_weight,
    )
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    run_generation(config)


if __name__ == "__main__":
    main()
