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

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


FIXED_DISK_COLORS = np.asarray(
    [
        [46, 166, 78],   # small -> green
        [65, 105, 225],  # blue
        [236, 105, 180], # pink
        [139, 94, 60],   # brown
        [148, 87, 235],  # purple
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class HanoiConfig:
    num_disks: int = 2
    num_videos: int = 3000
    disk_values: tuple[int, ...] = ()
    disk_counts: tuple[int, ...] = ()
    shuffle_disk_values: bool = True
    fps: int = 10
    frames_per_move: int = 15
    frame_size: int = 208
    output_dir: str = "outputs/hanoi/videos"
    struct_dir: str = "outputs/hanoi/struct"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260412
    workers: int = 1
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7
    initial_rods: tuple[int, ...] = (0, 1)
    goal_rod: int = 2
    allow_solved_initial: bool = False
    required_initial_rods_present: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.num_disks < 1:
            raise ValueError("num_disks must be >= 1")
        if self.num_videos < 1:
            raise ValueError("num_videos must be >= 1")
        if self.fps < 1:
            raise ValueError("fps must be >= 1")
        if self.frames_per_move < 1:
            raise ValueError("frames_per_move must be >= 1")
        if self.frame_size < 96:
            raise ValueError("frame_size must be >= 96")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if bool(self.disk_values) != bool(self.disk_counts):
            raise ValueError("disk_values and disk_counts must be provided together")
        if self.disk_values and len(self.disk_values) != len(self.disk_counts):
            raise ValueError("disk_values and disk_counts must have the same length")
        if self.disk_values and any(value < 1 for value in self.disk_values):
            raise ValueError("all disk_values must be >= 1")
        if self.disk_counts and any(count < 1 for count in self.disk_counts):
            raise ValueError("all disk_counts must be >= 1")
        if self.disk_values and sum(self.disk_counts) != self.num_videos:
            raise ValueError("sum(disk_counts) must equal num_videos")
        if not self.initial_rods:
            raise ValueError("initial_rods must not be empty")
        if any(rod not in (0, 1, 2) for rod in self.initial_rods):
            raise ValueError("initial_rods must be chosen from 0,1,2")
        if len(set(self.initial_rods)) != len(self.initial_rods):
            raise ValueError("initial_rods must not contain duplicates")
        if self.goal_rod not in (0, 1, 2):
            raise ValueError("goal_rod must be one of 0,1,2")
        if any(rod not in self.initial_rods for rod in self.required_initial_rods_present):
            raise ValueError("required_initial_rods_present must be a subset of initial_rods")
        if len(set(self.required_initial_rods_present)) != len(self.required_initial_rods_present):
            raise ValueError("required_initial_rods_present must not contain duplicates")
        if self.num_disks > len(FIXED_DISK_COLORS):
            raise ValueError(f"num_disks must be <= {len(FIXED_DISK_COLORS)} for the fixed palette")


@dataclass(frozen=True)
class VideoTask:
    index: int
    num_disks: int
    seed: int


def video_name(index: int) -> str:
    return f"hanoi_{index:04d}.mp4"


def npy_name(index: int) -> str:
    return f"hanoi_{index:04d}.npy"


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of integers")
    return tuple(int(value) for value in values)


def expand_disk_schedule(config: HanoiConfig) -> list[int]:
    if config.disk_values:
        schedule: list[int] = []
        for num_disks, count in zip(config.disk_values, config.disk_counts):
            schedule.extend([num_disks] * count)
    else:
        schedule = [config.num_disks] * config.num_videos
    if config.shuffle_disk_values:
        random.Random(config.seed).shuffle(schedule)
    return schedule


def build_tasks(config: HanoiConfig) -> list[VideoTask]:
    schedule = expand_disk_schedule(config)
    return [VideoTask(index=index, num_disks=num_disks, seed=config.seed + index) for index, num_disks in enumerate(schedule)]


def assignment_to_state(assignment: tuple[int, ...], num_disks: int) -> list[list[int]]:
    state = [[], [], []]
    for disk in range(num_disks - 1, -1, -1):
        state[assignment[disk]].append(disk)
    return state


def sample_initial_assignment(
    num_disks: int,
    initial_rods: tuple[int, ...],
    goal_rod: int,
    allow_solved_initial: bool,
    required_initial_rods_present: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    while True:
        rods_per_disk = [0] * num_disks
        for disk in range(num_disks - 1, -1, -1):
            rods_per_disk[disk] = rng.choice(initial_rods)
        assignment = tuple(rods_per_disk)
        if any(required_rod not in assignment for required_rod in required_initial_rods_present):
            continue
        if allow_solved_initial or any(rod != goal_rod for rod in assignment):
            return assignment


def valid_moves_from_assignment(state: tuple[int, ...]) -> list[tuple[int, int, int, tuple[int, ...]]]:
    num_disks = len(state)
    moves: list[tuple[int, int, int, tuple[int, ...]]] = []
    for disk in range(num_disks):
        source = state[disk]
        if any(state[smaller] == source for smaller in range(disk)):
            continue
        for target in (0, 1, 2):
            if target == source:
                continue
            if any(state[smaller] == target for smaller in range(disk)):
                continue
            updated = list(state)
            updated[disk] = target
            moves.append((disk, source, target, tuple(updated)))
    return moves


def solve_general_hanoi(initial_assignment: tuple[int, ...], goal_rod: int) -> list[tuple[int, int, int]]:
    goal_state = tuple(goal_rod for _ in initial_assignment)
    if initial_assignment == goal_state:
        return []

    queue: deque[tuple[int, ...]] = deque([initial_assignment])
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {initial_assignment: None}
    parent_move: dict[tuple[int, ...], tuple[int, int, int] | None] = {initial_assignment: None}

    while queue:
        current = queue.popleft()
        if current == goal_state:
            break
        for disk, source, target, nxt in valid_moves_from_assignment(current):
            if nxt in parent:
                continue
            parent[nxt] = current
            parent_move[nxt] = (disk, source, target)
            queue.append(nxt)

    if goal_state not in parent:
        raise RuntimeError("Failed to find a valid Hanoi solution to the goal rod")

    moves: list[tuple[int, int, int]] = []
    node = goal_state
    while parent[node] is not None:
        move = parent_move[node]
        assert move is not None
        moves.append(move)
        node = parent[node]
    moves.reverse()
    return moves


def clone_state(state: list[list[int]]) -> list[list[int]]:
    return [list(rod) for rod in state]


def apply_move(state: list[list[int]], move: tuple[int, int, int]) -> None:
    disk, source, target = move
    if not state[source] or state[source][-1] != disk:
        raise RuntimeError(f"illegal move {move}: source top mismatch")
    if state[target] and state[target][-1] < disk:
        raise RuntimeError(f"illegal move {move}: larger disk on smaller disk")
    state[source].pop()
    state[target].append(disk)


def state_sequence(initial_state_rods: list[list[int]], moves: list[tuple[int, int, int]]) -> list[list[list[int]]]:
    state = clone_state(initial_state_rods)
    states = [clone_state(state)]
    for move in moves:
        apply_move(state, move)
        states.append(clone_state(state))
    return states


def state_array(states: list[list[list[int]]], num_disks: int) -> np.ndarray:
    packed = np.full((len(states), 3, num_disks), fill_value=-1, dtype=np.int16)
    for state_index, state in enumerate(states):
        for rod_index, rod in enumerate(state):
            for depth, disk in enumerate(rod):
                packed[state_index, rod_index, depth] = disk
    return packed


def rod_x_positions(frame_size: int) -> np.ndarray:
    margin = frame_size * 0.18
    return np.asarray(
        [
            margin,
            frame_size * 0.5,
            frame_size - margin,
        ],
        dtype=np.float32,
    )


def layout(num_disks: int, frame_size: int) -> dict[str, object]:
    disk_height = max(10, int(round(frame_size * 0.055)))
    base_y = frame_size - int(round(frame_size * 0.12))
    rod_top_y = max(22, base_y - disk_height * (num_disks + 1))
    rod_xs = rod_x_positions(frame_size)
    spacing = float(rod_xs[1] - rod_xs[0])
    max_width = int(round(spacing * 0.72))
    min_width = max(18, int(round(max_width * 0.45)))
    widths = np.linspace(min_width, max_width, num_disks).round().astype(np.int16)
    lift_y = max(12.0, float(rod_top_y - disk_height))
    return {
        "frame_size": frame_size,
        "disk_height": disk_height,
        "base_y": float(base_y),
        "rod_top_y": float(rod_top_y),
        "rod_xs": rod_xs,
        "disk_widths": widths,
        "lift_y": lift_y,
    }


def disk_center(layout_info: dict[str, object], rod_index: int, level_from_bottom: int) -> np.ndarray:
    base_y = float(layout_info["base_y"])
    disk_height = int(layout_info["disk_height"])
    rod_x = float(np.asarray(layout_info["rod_xs"])[rod_index])
    center_y = base_y - disk_height * (level_from_bottom + 0.5)
    return np.asarray([center_y, rod_x], dtype=np.float32)


def disk_positions(state: list[list[int]], layout_info: dict[str, object]) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    for rod_index, rod in enumerate(state):
        for level_from_bottom, disk in enumerate(rod):
            positions[disk] = disk_center(layout_info, rod_index, level_from_bottom)
    return positions


def fixed_disk_colors(num_disks: int) -> np.ndarray:
    return FIXED_DISK_COLORS[:num_disks].copy()


def background_image(layout_info: dict[str, object]) -> np.ndarray:
    frame_size = int(layout_info["frame_size"])
    image = Image.new("RGB", (frame_size, frame_size), color=(247, 247, 244))
    draw = ImageDraw.Draw(image)
    base_y = float(layout_info["base_y"])
    rod_top_y = float(layout_info["rod_top_y"])
    rod_xs = np.asarray(layout_info["rod_xs"])
    rod_color = (110, 85, 60)
    base_color = (95, 95, 95)
    draw.rounded_rectangle(
        [
            (frame_size * 0.1, base_y + 6),
            (frame_size * 0.9, base_y + 16),
        ],
        radius=4,
        fill=base_color,
    )
    rod_width = max(4, int(round(frame_size * 0.02)))
    for rod_x in rod_xs.tolist():
        draw.rounded_rectangle(
            [
                (rod_x - rod_width / 2, rod_top_y),
                (rod_x + rod_width / 2, base_y + 4),
            ],
            radius=2,
            fill=rod_color,
        )
    return np.asarray(image, dtype=np.uint8)


def disk_sprites(layout_info: dict[str, object], colors: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    disk_height = int(layout_info["disk_height"])
    sprites: list[tuple[np.ndarray, np.ndarray]] = []
    for disk_index, width in enumerate(np.asarray(layout_info["disk_widths"]).tolist()):
        sprite = Image.new("RGBA", (int(width) + 6, disk_height + 6), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        draw.rounded_rectangle(
            [(3, 3), (3 + int(width), 3 + disk_height)],
            radius=max(2, disk_height // 3),
            fill=tuple(int(channel) for channel in colors[disk_index].tolist()) + (255,),
            outline=(35, 35, 35, 255),
            width=1,
        )
        arr = np.asarray(sprite, dtype=np.uint8)
        sprites.append((arr[..., :3], arr[..., 3] > 0))
    return sprites


def overlay_sprite(frame: np.ndarray, sprite_rgb: np.ndarray, sprite_mask: np.ndarray, center_y: float, center_x: float) -> None:
    top = int(round(center_y - sprite_rgb.shape[0] / 2))
    left = int(round(center_x - sprite_rgb.shape[1] / 2))
    bottom = top + sprite_rgb.shape[0]
    right = left + sprite_rgb.shape[1]

    clip_top = max(0, top)
    clip_left = max(0, left)
    clip_bottom = min(frame.shape[0], bottom)
    clip_right = min(frame.shape[1], right)
    if clip_top >= clip_bottom or clip_left >= clip_right:
        return

    src_top = clip_top - top
    src_left = clip_left - left
    src_bottom = src_top + (clip_bottom - clip_top)
    src_right = src_left + (clip_right - clip_left)

    mask = sprite_mask[src_top:src_bottom, src_left:src_right]
    target = frame[clip_top:clip_bottom, clip_left:clip_right]
    source = sprite_rgb[src_top:src_bottom, src_left:src_right]
    target[mask] = source[mask]


def render_frame(
    background: np.ndarray,
    stationary_positions: dict[int, np.ndarray],
    sprites: list[tuple[np.ndarray, np.ndarray]],
    moving_disk: int | None = None,
    moving_position: np.ndarray | None = None,
) -> np.ndarray:
    frame = background.copy()
    for disk in sorted(stationary_positions.keys(), reverse=True):
        if moving_disk is not None and disk == moving_disk:
            continue
        sprite_rgb, sprite_mask = sprites[disk]
        center_y, center_x = stationary_positions[disk]
        overlay_sprite(frame, sprite_rgb, sprite_mask, float(center_y), float(center_x))

    if moving_disk is not None and moving_position is not None:
        sprite_rgb, sprite_mask = sprites[moving_disk]
        overlay_sprite(frame, sprite_rgb, sprite_mask, float(moving_position[0]), float(moving_position[1]))
    return frame


def interpolate_polyline(points: list[np.ndarray], num_frames: int) -> list[np.ndarray]:
    if num_frames <= 0:
        return []
    segment_lengths: list[float] = []
    for start, end in zip(points, points[1:]):
        segment_lengths.append(float(np.linalg.norm(end - start)))
    total_length = sum(segment_lengths)
    if total_length <= 1e-6:
        return [points[-1].copy() for _ in range(num_frames)]

    cumulative = [0.0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    samples = np.linspace(total_length / num_frames, total_length, num_frames)
    out: list[np.ndarray] = []
    segment_index = 0
    for sample in samples.tolist():
        while segment_index + 1 < len(cumulative) and sample > cumulative[segment_index + 1]:
            segment_index += 1
        start = points[segment_index]
        end = points[segment_index + 1]
        start_distance = cumulative[segment_index]
        end_distance = cumulative[segment_index + 1]
        if end_distance - start_distance <= 1e-6:
            out.append(end.copy())
            continue
        alpha = (sample - start_distance) / (end_distance - start_distance)
        out.append((1.0 - alpha) * start + alpha * end)
    return out


def moving_positions_for_move(
    move: tuple[int, int, int],
    state_before: list[list[int]],
    state_after: list[list[int]],
    layout_info: dict[str, object],
    frames_per_move: int,
) -> list[np.ndarray]:
    disk, source, target = move
    start_level = len(state_before[source]) - 1
    end_level = len(state_after[target]) - 1
    start = disk_center(layout_info, source, start_level)
    end = disk_center(layout_info, target, end_level)
    lift_y = float(layout_info["lift_y"])
    points = [
        start,
        np.asarray([lift_y, start[1]], dtype=np.float32),
        np.asarray([lift_y, end[1]], dtype=np.float32),
        end,
    ]
    return interpolate_polyline(points, frames_per_move)


def sample_payload(task: VideoTask, config: HanoiConfig) -> tuple[dict[str, object], dict[str, object], list[np.ndarray]]:
    rng = random.Random(task.seed)
    initial_assignment = sample_initial_assignment(
        task.num_disks,
        config.initial_rods,
        config.goal_rod,
        config.allow_solved_initial,
        config.required_initial_rods_present,
        rng,
    )
    initial_state_rods = assignment_to_state(initial_assignment, task.num_disks)
    moves = solve_general_hanoi(initial_assignment, config.goal_rod)
    states = state_sequence(initial_state_rods, moves)
    layout_info = layout(task.num_disks, config.frame_size)
    colors = fixed_disk_colors(task.num_disks)

    metadata = {
        "index": task.index,
        "seed": task.seed,
        "file": video_name(task.index),
        "npy_file": npy_name(task.index),
        "num_disks": task.num_disks,
        "num_rods": 3,
        "start_rod": -1,
        "goal_rod": config.goal_rod,
        "allowed_initial_rods": list(config.initial_rods),
        "required_initial_rods_present": list(config.required_initial_rods_present),
        "initial_assignment": list(initial_assignment),
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
        "fps": config.fps,
        "frames_per_move": config.frames_per_move,
        "num_moves": len(moves),
        "num_frames": 1 + len(moves) * config.frames_per_move,
        "move_sequence": [list(move) for move in moves],
        "disk_colors": colors.tolist(),
        "initial_state": [list(rod) for rod in states[0]],
        "final_state": [list(rod) for rod in states[-1]],
    }

    struct = {
        "index": task.index,
        "seed": task.seed,
        "file": video_name(task.index),
        "num_disks": task.num_disks,
        "num_rods": 3,
        "frame_size": config.frame_size,
        "fps": config.fps,
        "frames_per_move": config.frames_per_move,
        "move_sequence": np.asarray(moves, dtype=np.int16),
        "states": state_array(states, task.num_disks),
        "disk_colors": colors.astype(np.uint8),
        "rod_xs": np.asarray(layout_info["rod_xs"], dtype=np.float32),
        "disk_widths": np.asarray(layout_info["disk_widths"], dtype=np.int16),
        "disk_height": np.int16(layout_info["disk_height"]),
        "base_y": np.float32(layout_info["base_y"]),
        "rod_top_y": np.float32(layout_info["rod_top_y"]),
        "lift_y": np.float32(layout_info["lift_y"]),
        "start_rod": np.int16(-1),
        "goal_rod": np.int16(config.goal_rod),
        "allowed_initial_rods": np.asarray(config.initial_rods, dtype=np.int16),
        "required_initial_rods_present": np.asarray(config.required_initial_rods_present, dtype=np.int16),
        "initial_assignment": np.asarray(initial_assignment, dtype=np.int16),
    }

    background = background_image(layout_info)
    sprites = disk_sprites(layout_info, colors)
    frames: list[np.ndarray] = []
    frames.append(render_frame(background, disk_positions(states[0], layout_info), sprites))

    for move, state_before, state_after in zip(moves, states[:-1], states[1:]):
        moving_disk = move[0]
        static_state = clone_state(state_before)
        static_state[move[1]].pop()
        stationary_positions = disk_positions(static_state, layout_info)
        for moving_position in moving_positions_for_move(move, state_before, state_after, layout_info, config.frames_per_move):
            frames.append(render_frame(background, stationary_positions, sprites, moving_disk=moving_disk, moving_position=moving_position))

    return metadata, struct, frames


def process_task(task: VideoTask, config: HanoiConfig) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    struct_dir = Path(config.struct_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / video_name(task.index)
    struct_path = struct_dir / npy_name(task.index)
    if not config.overwrite and video_path.exists() and struct_path.exists():
        struct = np.load(struct_path, allow_pickle=True).item()
        states = np.asarray(struct["states"])
        return {
            "index": task.index,
            "seed": task.seed,
            "file": video_name(task.index),
            "npy_file": npy_name(task.index),
            "num_disks": int(struct["num_disks"]),
            "num_rods": int(struct["num_rods"]),
            "start_rod": int(struct["start_rod"]),
            "goal_rod": int(struct["goal_rod"]),
            "allowed_initial_rods": np.asarray(struct["allowed_initial_rods"]).tolist(),
            "required_initial_rods_present": np.asarray(struct.get("required_initial_rods_present", np.asarray([], dtype=np.int16))).tolist(),
            "initial_assignment": np.asarray(struct["initial_assignment"]).tolist(),
            "frame_height": int(struct["frame_size"]),
            "frame_width": int(struct["frame_size"]),
            "fps": int(struct["fps"]),
            "frames_per_move": int(struct["frames_per_move"]),
            "num_moves": int(len(struct["move_sequence"])),
            "num_frames": 1 + int(len(struct["move_sequence"])) * int(struct["frames_per_move"]),
            "move_sequence": np.asarray(struct["move_sequence"]).tolist(),
            "disk_colors": np.asarray(struct["disk_colors"]).tolist(),
            "initial_state": states[0].tolist(),
            "final_state": states[-1].tolist(),
        }

    metadata, struct, frames = sample_payload(task, config)
    with imageio.get_writer(
        video_path,
        fps=config.fps,
        codec=config.codec,
        quality=config.quality,
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    np.save(struct_path, struct, allow_pickle=True)
    return metadata


def metadata_path(config: HanoiConfig) -> Path:
    return Path(config.output_dir) / config.metadata_name


def write_metadata(config: HanoiConfig, records: list[dict[str, object]]) -> None:
    with metadata_path(config).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(config: HanoiConfig) -> list[dict[str, object]]:
    config.validate()
    tasks = build_tasks(config)
    records: list[dict[str, object]] = []
    if config.workers == 1:
        for task in tqdm(tasks, desc="Generating Hanoi videos", leave=True):
            records.append(process_task(task, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            iterator = executor.map(process_task, tasks, repeat(config))
            for record in tqdm(iterator, total=len(tasks), desc="Generating Hanoi videos", leave=True):
                records.append(record)
    records.sort(key=lambda item: int(item["index"]))
    write_metadata(config, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Tower of Hanoi video dataset.")
    parser.add_argument("--num-disks", type=int, default=2)
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--disk-values", type=str, default="2,3,4,5")
    parser.add_argument("--disk-counts", type=str, default="750,750,750,750")
    parser.add_argument("--keep-disk-order", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frames-per-move", type=int, default=15)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-dir", type=str, default="outputs/hanoi/videos")
    parser.add_argument("--struct-dir", type=str, default="outputs/hanoi/struct")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--initial-rods", type=str, default="0,1")
    parser.add_argument("--goal-rod", type=int, default=2)
    parser.add_argument("--allow-solved-initial", action="store_true")
    parser.add_argument("--required-initial-rods-present", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    disk_values = parse_int_list(args.disk_values) if args.disk_values else ()
    disk_counts = parse_int_list(args.disk_counts) if args.disk_counts else ()
    required_initial_rods_present = parse_int_list(args.required_initial_rods_present) if args.required_initial_rods_present else ()
    config = HanoiConfig(
        num_disks=args.num_disks,
        num_videos=args.num_videos,
        disk_values=disk_values,
        disk_counts=disk_counts,
        shuffle_disk_values=not args.keep_disk_order,
        fps=args.fps,
        frames_per_move=args.frames_per_move,
        frame_size=args.frame_size,
        output_dir=args.output_dir,
        struct_dir=args.struct_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        workers=args.workers,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
        initial_rods=parse_int_list(args.initial_rods),
        goal_rod=args.goal_rod,
        allow_solved_initial=args.allow_solved_initial,
        required_initial_rods_present=required_initial_rods_present,
    )
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    generate_dataset(config)


if __name__ == "__main__":
    main()
