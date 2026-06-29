#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


@dataclass(frozen=True)
class ShellGameConfig:
    num_swaps: int = 1
    num_videos: int = 3000
    swap_values: tuple[int, ...] = ()
    swap_counts: tuple[int, ...] = ()
    shuffle_swap_values: bool = True
    fps: int = 20
    frame_size: int = 208
    output_base_dir: str = "outputs/shell_game/train"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260413
    workers: int = 1
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7

    def validate(self) -> None:
        if self.num_swaps < 1:
            raise ValueError("num_swaps must be >= 1")
        if self.num_videos < 1:
            raise ValueError("num_videos must be >= 1")
        if self.fps < 1:
            raise ValueError("fps must be >= 1")
        if self.frame_size < 96:
            raise ValueError("frame_size must be >= 96")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if bool(self.swap_values) != bool(self.swap_counts):
            raise ValueError("swap_values and swap_counts must be provided together")
        if self.swap_values and len(self.swap_values) != len(self.swap_counts):
            raise ValueError("swap_values and swap_counts must have the same length")
        if self.swap_values and sum(self.swap_counts) != self.num_videos:
            raise ValueError("sum(swap_counts) must equal num_videos")
        if self.swap_values and any(value < 1 for value in self.swap_values):
            raise ValueError("all swap_values must be >= 1")


@dataclass(frozen=True)
class VideoTask:
    index: int
    num_swaps: int
    seed: int


BACKGROUND_RGB = (255, 255, 255)
BALL_RGB = (220, 45, 45)
CUP_RGB = (10, 10, 10)
SLOT_XS = (52.0, 104.0, 156.0)
BALL_Y = 162.0
CUP_BASE_Y = 146.0
CUP_START_Y = -18.0
CUP_WIDTH = 36
CUP_HEIGHT = 52
SWAP_ARC = 24.0
INITIAL_HOLD_FRAMES = 10
DESCENT_FRAMES = 14
SWAP_FRAMES = 16
CONDITION_FINAL_HOLD = 10
TARGET_INITIAL_HOLD = 8
REVEAL_FRAMES = 14
TARGET_FINAL_HOLD = 10
SWAP_PAIRS = ((0, 1), (0, 2), (1, 2))


def condition_filename(index: int) -> str:
    return f"sample_{index:04d}.mp4"


def target_filename(index: int) -> str:
    return f"sample_{index:04d}.mp4"


def struct_filename(index: int) -> str:
    return f"sample_{index:04d}.npy"


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of integers")
    return tuple(int(value) for value in values)


def expand_swap_schedule(config: ShellGameConfig) -> list[int]:
    if config.swap_values:
        schedule: list[int] = []
        for swap_value, count in zip(config.swap_values, config.swap_counts):
            schedule.extend([swap_value] * count)
    else:
        schedule = [config.num_swaps] * config.num_videos
    if config.shuffle_swap_values:
        random.Random(config.seed).shuffle(schedule)
    return schedule


def build_tasks(config: ShellGameConfig) -> list[VideoTask]:
    schedule = expand_swap_schedule(config)
    return [VideoTask(index=index, num_swaps=num_swaps, seed=config.seed + index) for index, num_swaps in enumerate(schedule)]


def base_dirs(config: ShellGameConfig) -> tuple[Path, Path, Path]:
    root = Path(config.output_base_dir)
    condition_dir = root / "condition_videos"
    target_dir = root / "target_videos"
    struct_dir = root / "npy"
    return condition_dir, target_dir, struct_dir


def ensure_dirs(config: ShellGameConfig) -> tuple[Path, Path, Path]:
    condition_dir, target_dir, struct_dir = base_dirs(config)
    condition_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    return condition_dir, target_dir, struct_dir


def cup_slot_positions() -> np.ndarray:
    return np.asarray([[SLOT_XS[0], CUP_BASE_Y], [SLOT_XS[1], CUP_BASE_Y], [SLOT_XS[2], CUP_BASE_Y]], dtype=np.float32)


def cup_center_at(slot_index: int) -> np.ndarray:
    return np.asarray([SLOT_XS[slot_index], CUP_BASE_Y], dtype=np.float32)


def ball_center_at(slot_index: int) -> np.ndarray:
    return np.asarray([SLOT_XS[slot_index], BALL_Y], dtype=np.float32)


def draw_ball(draw: ImageDraw.ImageDraw, center: np.ndarray) -> None:
    x, y = float(center[0]), float(center[1])
    radius = 7
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=BALL_RGB)


def draw_cup(draw: ImageDraw.ImageDraw, center: np.ndarray) -> None:
    x, y = float(center[0]), float(center[1])
    left = x - CUP_WIDTH / 2
    right = x + CUP_WIDTH / 2
    top = y - CUP_HEIGHT / 2
    bottom = y + CUP_HEIGHT / 2
    lip_h = 10
    body_points = [
        (left + 2, bottom),
        (left + 6, top + lip_h),
        (right - 6, top + lip_h),
        (right - 2, bottom),
    ]
    draw.polygon(body_points, fill=CUP_RGB)
    draw.rounded_rectangle([(left + 4, top), (right - 4, top + lip_h + 6)], radius=5, fill=CUP_RGB)


def render_scene(ball_slot: int | None, cup_positions: dict[int, np.ndarray], lifted_cup_id: int | None = None) -> np.ndarray:
    image = Image.new("RGB", (208, 208), color=BACKGROUND_RGB)
    draw = ImageDraw.Draw(image)
    if ball_slot is not None:
        draw_ball(draw, ball_center_at(ball_slot))
    for cup_id in sorted(cup_positions.keys()):
        if lifted_cup_id is not None and cup_id == lifted_cup_id:
            continue
        draw_cup(draw, cup_positions[cup_id])
    if lifted_cup_id is not None:
        draw_cup(draw, cup_positions[lifted_cup_id])
    return np.asarray(image, dtype=np.uint8)


def initial_cup_positions() -> dict[int, np.ndarray]:
    return {cup_id: np.asarray([SLOT_XS[cup_id], CUP_START_Y], dtype=np.float32) for cup_id in range(3)}


def settled_cup_positions() -> dict[int, np.ndarray]:
    return {cup_id: cup_center_at(cup_id) for cup_id in range(3)}


def descent_frames() -> list[dict[int, np.ndarray]]:
    start = initial_cup_positions()
    end = settled_cup_positions()
    frames: list[dict[int, np.ndarray]] = []
    for alpha in np.linspace(0.0, 1.0, DESCENT_FRAMES):
        frame_positions = {
            cup_id: ((1.0 - alpha) * start[cup_id] + alpha * end[cup_id]).astype(np.float32)
            for cup_id in range(3)
        }
        frames.append(frame_positions)
    return frames


def swap_animation(
    current_slots: list[int],
    pair: tuple[int, int],
    rng: random.Random,
) -> tuple[list[dict[int, np.ndarray]], list[int]]:
    left_slot, right_slot = pair
    cup_a = current_slots[left_slot]
    cup_b = current_slots[right_slot]
    cup_up, cup_down = (cup_a, cup_b) if rng.random() < 0.5 else (cup_b, cup_a)
    start_positions = {cup_id: cup_center_at(slot_index) for slot_index, cup_id in enumerate(current_slots)}
    frames: list[dict[int, np.ndarray]] = []
    for alpha in np.linspace(0.0, 1.0, SWAP_FRAMES):
        positions = {cup_id: start_positions[cup_id].copy() for cup_id in current_slots}
        x_left = SLOT_XS[left_slot]
        x_right = SLOT_XS[right_slot]
        y_shift = math.sin(math.pi * float(alpha)) * SWAP_ARC

        positions[cup_a] = np.asarray(
            [
                (1.0 - alpha) * x_left + alpha * x_right,
                CUP_BASE_Y - y_shift if cup_a == cup_up else CUP_BASE_Y + y_shift,
            ],
            dtype=np.float32,
        )
        positions[cup_b] = np.asarray(
            [
                (1.0 - alpha) * x_right + alpha * x_left,
                CUP_BASE_Y - y_shift if cup_b == cup_up else CUP_BASE_Y + y_shift,
            ],
            dtype=np.float32,
        )
        frames.append(positions)

    updated_slots = list(current_slots)
    updated_slots[left_slot], updated_slots[right_slot] = updated_slots[right_slot], updated_slots[left_slot]
    return frames, updated_slots


def reveal_animation(final_slots: list[int], winning_cup_id: int) -> list[dict[int, np.ndarray]]:
    base_positions = {cup_id: cup_center_at(slot_index) for slot_index, cup_id in enumerate(final_slots)}
    frames: list[dict[int, np.ndarray]] = []
    for alpha in np.linspace(0.0, 1.0, REVEAL_FRAMES):
        positions = {cup_id: position.copy() for cup_id, position in base_positions.items()}
        positions[winning_cup_id] = np.asarray([positions[winning_cup_id][0], CUP_BASE_Y - 56.0 * float(alpha)], dtype=np.float32)
        frames.append(positions)
    return frames


def sample_swap_sequence(num_swaps: int, rng: random.Random) -> list[tuple[int, int]]:
    return [tuple(sorted(rng.choice(SWAP_PAIRS))) for _ in range(num_swaps)]


def final_ball_slot(initial_ball_slot: int, swaps: list[tuple[int, int]]) -> tuple[int, list[int], int]:
    slots = [0, 1, 2]
    ball_cup_id = slots[initial_ball_slot]
    for left_slot, right_slot in swaps:
        slots[left_slot], slots[right_slot] = slots[right_slot], slots[left_slot]
    winning_slot = slots.index(ball_cup_id)
    return winning_slot, slots, ball_cup_id


def condition_video_frames(initial_ball_slot: int, swaps: list[tuple[int, int]], rng: random.Random) -> tuple[list[np.ndarray], list[int]]:
    frames: list[np.ndarray] = []
    slots = [0, 1, 2]

    for _ in range(INITIAL_HOLD_FRAMES):
        frames.append(render_scene(initial_ball_slot, {}))

    for positions in descent_frames():
        frames.append(render_scene(initial_ball_slot, positions))

    current_slots = list(slots)
    for pair in swaps:
        swap_frames, current_slots = swap_animation(current_slots, pair, rng)
        for positions in swap_frames:
            frames.append(render_scene(None, positions))

    final_positions = {cup_id: cup_center_at(slot_index) for slot_index, cup_id in enumerate(current_slots)}
    for _ in range(CONDITION_FINAL_HOLD):
        frames.append(render_scene(None, final_positions))
    return frames, current_slots


def target_video_frames(winning_slot: int, final_slots: list[int], winning_cup_id: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    final_positions = {cup_id: cup_center_at(slot_index) for slot_index, cup_id in enumerate(final_slots)}
    for _ in range(TARGET_INITIAL_HOLD):
        frames.append(render_scene(None, final_positions))
    for positions in reveal_animation(final_slots, winning_cup_id):
        frames.append(render_scene(winning_slot, positions, lifted_cup_id=winning_cup_id))
    lifted_positions = reveal_animation(final_slots, winning_cup_id)[-1]
    for _ in range(TARGET_FINAL_HOLD):
        frames.append(render_scene(winning_slot, lifted_positions, lifted_cup_id=winning_cup_id))
    return frames


def sample_payload(task: VideoTask, config: ShellGameConfig) -> tuple[dict[str, object], dict[str, object], list[np.ndarray], list[np.ndarray]]:
    rng = random.Random(task.seed)
    initial_ball_slot = rng.randrange(3)
    swaps = sample_swap_sequence(task.num_swaps, rng)
    winning_slot, _computed_slots, winning_cup_id = final_ball_slot(initial_ball_slot, swaps)
    condition_frames, final_slots = condition_video_frames(initial_ball_slot, swaps, rng)
    target_frames = target_video_frames(winning_slot, final_slots, winning_cup_id)

    metadata = {
        "index": task.index,
        "seed": task.seed,
        "num_swaps": task.num_swaps,
        "fps": config.fps,
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
        "condition_file": condition_filename(task.index),
        "target_file": target_filename(task.index),
        "npy_file": struct_filename(task.index),
        "initial_ball_slot": int(initial_ball_slot),
        "swap_sequence": [list(pair) for pair in swaps],
        "final_ball_slot": int(winning_slot),
        "winning_cup_id": int(winning_cup_id),
        "condition_num_frames": len(condition_frames),
        "target_num_frames": len(target_frames),
    }
    struct = {
        "index": np.int16(task.index),
        "seed": np.int32(task.seed),
        "num_swaps": np.int16(task.num_swaps),
        "fps": np.int16(config.fps),
        "frame_size": np.int16(config.frame_size),
        "initial_ball_slot": np.int16(initial_ball_slot),
        "swap_sequence": np.asarray(swaps, dtype=np.int16),
        "final_ball_slot": np.int16(winning_slot),
        "winning_cup_id": np.int16(winning_cup_id),
        "slot_xs": np.asarray(SLOT_XS, dtype=np.float32),
        "ball_y": np.float32(BALL_Y),
        "cup_base_y": np.float32(CUP_BASE_Y),
    }
    return metadata, struct, condition_frames, target_frames


def process_task(task: VideoTask, config: ShellGameConfig) -> dict[str, object]:
    condition_dir, target_dir, struct_dir = ensure_dirs(config)
    condition_path = condition_dir / condition_filename(task.index)
    target_path = target_dir / target_filename(task.index)
    struct_path = struct_dir / struct_filename(task.index)

    if not config.overwrite and condition_path.exists() and target_path.exists() and struct_path.exists():
        struct = np.load(struct_path, allow_pickle=True).item()
        return {
            "index": int(struct["index"]),
            "seed": int(struct["seed"]),
            "num_swaps": int(struct["num_swaps"]),
            "fps": int(struct["fps"]),
            "frame_height": int(struct["frame_size"]),
            "frame_width": int(struct["frame_size"]),
            "condition_file": condition_filename(task.index),
            "target_file": target_filename(task.index),
            "npy_file": struct_filename(task.index),
            "initial_ball_slot": int(struct["initial_ball_slot"]),
            "swap_sequence": np.asarray(struct["swap_sequence"]).tolist(),
            "final_ball_slot": int(struct["final_ball_slot"]),
            "winning_cup_id": int(struct["winning_cup_id"]),
            "condition_num_frames": int(imageio.get_reader(condition_path).count_frames()),
            "target_num_frames": int(imageio.get_reader(target_path).count_frames()),
        }

    metadata, struct, condition_frames, target_frames = sample_payload(task, config)
    with imageio.get_writer(condition_path, fps=config.fps, codec=config.codec, quality=config.quality, macro_block_size=1) as writer:
        for frame in condition_frames:
            writer.append_data(frame)
    with imageio.get_writer(target_path, fps=config.fps, codec=config.codec, quality=config.quality, macro_block_size=1) as writer:
        for frame in target_frames:
            writer.append_data(frame)
    np.save(struct_path, struct, allow_pickle=True)
    return metadata


def write_metadata(config: ShellGameConfig, records: list[dict[str, object]]) -> None:
    root = Path(config.output_base_dir)
    with (root / config.metadata_name).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(config: ShellGameConfig) -> list[dict[str, object]]:
    config.validate()
    ensure_dirs(config)
    tasks = build_tasks(config)
    records: list[dict[str, object]] = []
    if config.workers == 1:
        for task in tqdm(tasks, desc="Generating shell-game videos", leave=True):
            records.append(process_task(task, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            iterator = executor.map(process_task, tasks, repeat(config))
            for record in tqdm(iterator, total=len(tasks), desc="Generating shell-game videos", leave=True):
                records.append(record)
    records.sort(key=lambda item: int(item["index"]))
    write_metadata(config, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate shell-game condition/target videos.")
    parser.add_argument("--num-swaps", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--swap-values", type=str, default="1,2,3,4,5")
    parser.add_argument("--swap-counts", type=str, default="600,600,600,600,600")
    parser.add_argument("--keep-swap-order", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-base-dir", type=str, default="outputs/shell_game/train")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ShellGameConfig(
        num_swaps=args.num_swaps,
        num_videos=args.num_videos,
        swap_values=parse_int_list(args.swap_values) if args.swap_values else (),
        swap_counts=parse_int_list(args.swap_counts) if args.swap_counts else (),
        shuffle_swap_values=not args.keep_swap_order,
        fps=args.fps,
        frame_size=args.frame_size,
        output_base_dir=args.output_base_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        workers=args.workers,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
    )
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    generate_dataset(config)


if __name__ == "__main__":
    main()
