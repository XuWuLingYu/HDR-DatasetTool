#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from generators.classic.maze_videos import (
    Config,
    effective_ball_radius,
    graph_to_wall_grid,
    grid_to_background,
    iter_positions,
    make_ball_sprite,
    render_frame,
)


Cell = tuple[int, int]


@dataclass(frozen=True)
class HardTestConfig:
    maze_size: int = 6
    fps: int = 10
    frames_per_move: int = 5
    tile_size: int = 16
    ball_radius: int = 5
    frame_size: int = 208
    output_video_dir: str = "maze_test_hard"
    output_struct_dir: str = "maze_test_hard_npy"
    seed: int = 20260415
    codec: str = "libx264"
    quality: int = 7
    overwrite: bool = True
    write_videos: bool = True


def video_name(index: int) -> str:
    return f"maze_{index:04d}.mp4"


def npy_name(index: int) -> str:
    return f"maze_{index:04d}.npy"


def add_edge(graph: dict[Cell, list[Cell]], a: Cell, b: Cell) -> None:
    graph[a].append(b)
    graph[b].append(a)


def serialize_edges(graph: dict[Cell, list[Cell]]) -> np.ndarray:
    edges: list[list[int]] = []
    for (r1, c1), neighbors in graph.items():
        for r2, c2 in neighbors:
            if (r1, c1) < (r2, c2):
                edges.append([r1, c1, r2, c2])
    return np.asarray(edges, dtype=np.int16)


def cell_centers_for(maze_size: int, frame_size: int) -> np.ndarray:
    scale = frame_size / (2 * maze_size + 1)
    centers = np.zeros((maze_size, maze_size, 2), dtype=np.float32)
    for row in range(maze_size):
        for col in range(maze_size):
            logical_row = 2 * row + 1
            logical_col = 2 * col + 1
            centers[row, col, 0] = (logical_row + 0.5) * scale - 0.5
            centers[row, col, 1] = (logical_col + 0.5) * scale - 0.5
    return centers


def build_runtime_config(config: HardTestConfig) -> Config:
    return Config(
        n=config.maze_size,
        num_videos=1,
        algorithm="candidate_dfs",
        fps=config.fps,
        frames_per_move=config.frames_per_move,
        tile_size=config.tile_size,
        ball_radius=config.ball_radius,
        frame_size=config.frame_size,
        output_dir=config.output_video_dir,
        metadata_name="metadata.jsonl",
        seed=config.seed,
        workers=1,
        overwrite=config.overwrite,
        codec=config.codec,
        quality=config.quality,
        candidate_mazes=1,
        top_candidates=1,
        branch_weight=0.0,
        loop_weight=0.0,
    )


def build_spiral_path_n6() -> list[Cell]:
    path = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (4, 4),
        (3, 4),
        (2, 4),
        (1, 4),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (4, 2),
        (3, 2),
        (2, 2),
        (1, 2),
        (1, 1),
        (2, 1),
        (2, 0),
        (3, 0),
        (3, 1),
        (4, 1),
        (4, 0),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
    ]
    if len(path) != 35:
        raise RuntimeError("Expected the N=6 hard spiral path to visit 35 cells")
    if len(set(path)) != len(path):
        raise RuntimeError("Hard spiral path contains repeated cells")
    for current, nxt in zip(path, path[1:]):
        if abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) != 1:
            raise RuntimeError(f"Non-adjacent step detected: {current} -> {nxt}")
    if path[0] != (0, 0) or path[-1] != (5, 5):
        raise RuntimeError("Hard spiral path must start at the top-left and end at the bottom-right")
    return path


def build_spiral_path_n7() -> list[Cell]:
    path = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
        (5, 5),
        (4, 5),
        (3, 5),
        (2, 5),
        (1, 5),
        (1, 4),
        (1, 3),
        (1, 2),
        (1, 1),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (5, 3),
        (5, 2),
        (5, 1),
        (4, 1),
        (3, 1),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 3),
        (3, 2),
        (4, 2),
        (4, 3),
        (4, 4),
        (5, 4),
        (6, 4),
        (6, 5),
        (6, 6),
    ]
    if len(path) != 49:
        raise RuntimeError("Expected the N=7 hard spiral path to visit all 49 cells")
    if len(set(path)) != len(path):
        raise RuntimeError("Hard spiral path contains repeated cells")
    for current, nxt in zip(path, path[1:]):
        if abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) != 1:
            raise RuntimeError(f"Non-adjacent step detected: {current} -> {nxt}")
    if path[0] != (0, 0) or path[-1] != (6, 6):
        raise RuntimeError("Hard spiral path must start at the top-left and end at the bottom-right")
    return path


def path_to_graph(path: list[Cell], maze_size: int) -> dict[Cell, list[Cell]]:
    graph: dict[Cell, list[Cell]] = {(row, col): [] for row in range(maze_size) for col in range(maze_size)}
    for current, nxt in zip(path, path[1:]):
        add_edge(graph, current, nxt)
    return graph


def build_sample(config: HardTestConfig) -> tuple[dict[str, object], dict[str, object], np.ndarray, np.ndarray, np.ndarray, list[Cell]]:
    if config.maze_size == 6:
        path = build_spiral_path_n6()
        hard_case_name = "spiral_n6_long_path"
        description = "A hand-authored 6x6 spiral-like long corridor from top-left to bottom-right."
    elif config.maze_size == 7:
        path = build_spiral_path_n7()
        hard_case_name = "spiral_n7_full_grid"
        description = "A hand-authored 7x7 spiral-like Hamiltonian corridor from top-left to bottom-right."
    else:
        raise ValueError("This hard-case generator currently supports only maze_size=6 or maze_size=7")

    runtime_config = build_runtime_config(config)
    runtime_config.validate()
    graph = path_to_graph(path, config.maze_size)
    wall_grid = graph_to_wall_grid(graph, config.maze_size)
    background = grid_to_background(wall_grid, config.tile_size, config.frame_size)
    ball_sprite, ball_mask = make_ball_sprite(effective_ball_radius(runtime_config, config.maze_size))

    metadata = {
        "index": 0,
        "seed": config.seed,
        "file": video_name(0),
        "algorithm": "hard_spiral_manual",
        "maze_size": config.maze_size,
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
        "fps": config.fps,
        "frames_per_move": config.frames_per_move,
        "num_frames": 1 + (len(path) - 1) * config.frames_per_move,
        "path_length_cells": len(path),
        "path_branch_nodes": 0,
        "path_total_extra_edges": 0,
        "cycle_edges": 0,
        "candidate_rank": 1,
        "candidate_pool_size": 1,
        "selection_score": float(len(path)),
        "start": [0, 0],
        "goal": [config.maze_size - 1, config.maze_size - 1],
        "solution_path": [list(cell) for cell in path],
        "npy_file": npy_name(0),
        "hard_case_name": hard_case_name,
        "description": description,
    }

    struct = {
        "index": 0,
        "seed": config.seed,
        "file": video_name(0),
        "algorithm": "hard_spiral_manual",
        "maze_size": config.maze_size,
        "frame_size": config.frame_size,
        "fps": config.fps,
        "frames_per_move": config.frames_per_move,
        "start": np.asarray([0, 0], dtype=np.int16),
        "goal": np.asarray([config.maze_size - 1, config.maze_size - 1], dtype=np.int16),
        "solution_path": np.asarray(path, dtype=np.int16),
        "wall_grid": wall_grid.astype(np.uint8),
        "edges": serialize_edges(graph),
        "cell_centers": cell_centers_for(config.maze_size, config.frame_size),
        "hard_case_name": hard_case_name,
    }
    return metadata, struct, background, ball_sprite, ball_mask, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a single hard spiral-like maze test sample.")
    parser.add_argument("--maze-size", type=int, default=6)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frames-per-move", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--ball-radius", type=int, default=5)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-video-dir", type=str, default="maze_test_hard")
    parser.add_argument("--output-struct-dir", type=str, default="maze_test_hard_npy")
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-video-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = HardTestConfig(
        maze_size=args.maze_size,
        fps=args.fps,
        frames_per_move=args.frames_per_move,
        tile_size=args.tile_size,
        ball_radius=args.ball_radius,
        frame_size=args.frame_size,
        output_video_dir=args.output_video_dir,
        output_struct_dir=args.output_struct_dir,
        seed=args.seed,
        codec=args.codec,
        quality=args.quality,
        overwrite=args.overwrite,
        write_videos=not args.skip_video_write,
    )

    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))

    video_dir = Path(config.output_video_dir)
    struct_dir = Path(config.output_struct_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)

    metadata, struct, background, ball_sprite, ball_mask, path = build_sample(config)
    video_path = video_dir / metadata["file"]
    npy_path = struct_dir / metadata["npy_file"]

    if config.write_videos:
        with imageio.get_writer(
            video_path,
            fps=config.fps,
            codec=config.codec,
            quality=config.quality,
            macro_block_size=1,
        ) as writer:
            for center_y, center_x in iter_positions(
                path,
                config.maze_size,
                config.tile_size,
                config.frames_per_move,
                config.frame_size,
            ):
                frame = render_frame(background, ball_sprite, ball_mask, center_y, center_x)
                writer.append_data(frame)

    np.save(npy_path, struct, allow_pickle=True)
    with (video_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
