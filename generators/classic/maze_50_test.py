#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from generators.classic.maze_videos import (
    Config,
    effective_ball_radius,
    generate_recursive_division_sample,
    generate_varied_maze,
    graph_to_wall_grid,
    grid_to_background,
    iter_positions,
    make_ball_sprite,
    render_frame,
)


@dataclass(frozen=True)
class TestConfig:
    algorithm: str = "recursive_division"
    sizes: tuple[int, ...] = (6, 7, 8, 9, 10)
    count_per_size: int = 10
    fps: int = 10
    frames_per_move: int = 5
    tile_size: int = 16
    ball_radius: int = 5
    frame_size: int = 208
    output_video_dir: str = "maze_50_test"
    output_struct_dir: str = "maze_50_test_npy"
    seed: int = 20260411
    codec: str = "libx264"
    quality: int = 7
    candidate_mazes: int = 8
    top_candidates: int = 3
    branch_weight: float = 0.35
    loop_weight: float = 0.25
    overwrite: bool = True
    write_videos: bool = True

    @property
    def num_videos(self) -> int:
        return len(self.sizes) * self.count_per_size


def video_name(index: int) -> str:
    return f"maze_{index:04d}.mp4"


def npy_name(index: int) -> str:
    return f"maze_{index:04d}.npy"


def serialize_edges(graph: dict[tuple[int, int], list[tuple[int, int]]]) -> np.ndarray:
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


def build_runtime_config(test_config: TestConfig) -> Config:
    return Config(
        n=test_config.sizes[0],
        num_videos=test_config.num_videos,
        algorithm=test_config.algorithm,
        fps=test_config.fps,
        frames_per_move=test_config.frames_per_move,
        tile_size=test_config.tile_size,
        ball_radius=test_config.ball_radius,
        frame_size=test_config.frame_size,
        output_dir=test_config.output_video_dir,
        metadata_name="metadata.jsonl",
        seed=test_config.seed,
        workers=1,
        overwrite=test_config.overwrite,
        codec=test_config.codec,
        quality=test_config.quality,
        candidate_mazes=test_config.candidate_mazes,
        top_candidates=test_config.top_candidates,
        branch_weight=test_config.branch_weight,
        loop_weight=test_config.loop_weight,
    )


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of integers")
    return tuple(int(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 50-sample maze test data with videos and structured npy.")
    parser.add_argument("--algorithm", type=str, default="recursive_division")
    parser.add_argument("--sizes", type=str, default="6,7,8,9,10")
    parser.add_argument("--count-per-size", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frames-per-move", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--ball-radius", type=int, default=5)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-video-dir", type=str, default="maze_50_test")
    parser.add_argument("--output-struct-dir", type=str, default="maze_50_test_npy")
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--candidate-mazes", type=int, default=8)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--branch-weight", type=float, default=0.35)
    parser.add_argument("--loop-weight", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-video-write", action="store_true")
    return parser


def generate_single_sample(index: int, maze_size: int, config: Config) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
    rng_seed = config.seed + index
    rng = __import__("random").Random(rng_seed)
    if config.algorithm == "recursive_division":
        maze = generate_recursive_division_sample(config, rng, maze_size)
    else:
        maze = generate_varied_maze(config, rng, maze_size)

    graph = maze["graph"]
    path = maze["path"]
    wall_grid = graph_to_wall_grid(graph, maze_size)
    background = grid_to_background(wall_grid, config.tile_size, config.frame_size)
    ball_sprite, ball_mask = make_ball_sprite(effective_ball_radius(config, maze_size))

    metadata = {
        "index": index,
        "seed": rng_seed,
        "file": video_name(index),
        "algorithm": config.algorithm,
        "maze_size": maze_size,
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
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
        "start": [0, 0],
        "goal": [maze_size - 1, maze_size - 1],
        "solution_path": [list(cell) for cell in path],
        "npy_file": npy_name(index),
    }

    struct = {
        "index": index,
        "seed": rng_seed,
        "file": video_name(index),
        "algorithm": config.algorithm,
        "maze_size": maze_size,
        "frame_size": config.frame_size,
        "fps": config.fps,
        "frames_per_move": config.frames_per_move,
        "start": np.asarray([0, 0], dtype=np.int16),
        "goal": np.asarray([maze_size - 1, maze_size - 1], dtype=np.int16),
        "solution_path": np.asarray(path, dtype=np.int16),
        "wall_grid": wall_grid.astype(np.uint8),
        "edges": serialize_edges(graph),
        "cell_centers": cell_centers_for(maze_size, config.frame_size),
    }
    return metadata, struct, background, ball_sprite, ball_mask, path


def main() -> None:
    args = build_parser().parse_args()
    test_config = TestConfig(
        algorithm=args.algorithm,
        sizes=parse_int_list(args.sizes),
        count_per_size=args.count_per_size,
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
        candidate_mazes=args.candidate_mazes,
        top_candidates=args.top_candidates,
        branch_weight=args.branch_weight,
        loop_weight=args.loop_weight,
        overwrite=args.overwrite,
        write_videos=not args.skip_video_write,
    )
    config = build_runtime_config(test_config)
    config.validate()

    video_dir = Path(test_config.output_video_dir)
    struct_dir = Path(test_config.output_struct_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps(asdict(test_config), ensure_ascii=False, indent=2))

    metadata_records: list[dict[str, object]] = []
    index = 0
    for maze_size in test_config.sizes:
        for _ in tqdm(range(test_config.count_per_size), desc=f"N={maze_size}", leave=True):
            metadata, struct, background, ball_sprite, ball_mask, path = generate_single_sample(index, maze_size, config)
            video_path = video_dir / metadata["file"]
            npy_path = struct_dir / metadata["npy_file"]

            if test_config.write_videos:
                with imageio.get_writer(
                    video_path,
                    fps=config.fps,
                    codec=config.codec,
                    quality=config.quality,
                    macro_block_size=1,
                ) as writer:
                    for center_y, center_x in iter_positions(
                        path,
                        maze_size,
                        config.tile_size,
                        config.frames_per_move,
                        config.frame_size,
                    ):
                        frame = render_frame(background, ball_sprite, ball_mask, center_y, center_x)
                        writer.append_data(frame)

            np.save(npy_path, struct, allow_pickle=True)
            metadata_records.append(metadata)
            index += 1

    with (video_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in metadata_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "algorithm": test_config.algorithm,
        "sizes": np.asarray(test_config.sizes, dtype=np.int16),
        "count_per_size": test_config.count_per_size,
        "frame_size": test_config.frame_size,
        "fps": test_config.fps,
        "frames_per_move": test_config.frames_per_move,
        "records": metadata_records,
    }
    np.save(struct_dir / "manifest.npy", manifest, allow_pickle=True)


if __name__ == "__main__":
    main()
