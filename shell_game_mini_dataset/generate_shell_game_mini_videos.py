#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
import random

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


BASE_FRAME_SIZE = 208.0
BACKGROUND_RGB = (255, 255, 255)
BALL_RGB = (220, 45, 45)
CUP_RGB = (10, 10, 10)
BASE_SLOT_XS = (52.0, 104.0, 156.0)
BASE_BALL_Y = 162.0
BASE_CUP_BASE_Y = 146.0
BASE_CUP_START_Y = -18.0
BASE_CUP_WIDTH = 36.0
BASE_CUP_HEIGHT = 52.0
BASE_BALL_RADIUS = 7.0
BASE_CUP_LIP_HEIGHT = 10.0
BASE_REVEAL_LIFT = 56.0
INITIAL_HOLD_FRAMES = 10
DESCENT_FRAMES = 14
REVEAL_FRAMES = 14
TARGET_FINAL_HOLD = 10


@dataclass(frozen=True)
class ShellGameMiniConfig:
    occlusion_seconds: int = 1
    num_videos: int = 3000
    wait_values: tuple[int, ...] = ()
    wait_counts: tuple[int, ...] = ()
    shuffle_wait_values: bool = True
    fps: int = 20
    frame_size: int = 208
    output_base_dir: str = "shell_game_mini_dataset/train"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260415
    workers: int = 1
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7

    def validate(self) -> None:
        if self.occlusion_seconds < 1:
            raise ValueError("occlusion_seconds must be >= 1")
        if self.num_videos < 1:
            raise ValueError("num_videos must be >= 1")
        if self.fps < 1:
            raise ValueError("fps must be >= 1")
        if self.frame_size < 96:
            raise ValueError("frame_size must be >= 96")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if bool(self.wait_values) != bool(self.wait_counts):
            raise ValueError("wait_values and wait_counts must be provided together")
        if self.wait_values and len(self.wait_values) != len(self.wait_counts):
            raise ValueError("wait_values and wait_counts must have the same length")
        if self.wait_values and sum(self.wait_counts) != self.num_videos:
            raise ValueError("sum(wait_counts) must equal num_videos")
        if self.wait_values and any(value < 1 for value in self.wait_values):
            raise ValueError("all wait_values must be >= 1")


@dataclass(frozen=True)
class VideoTask:
    index: int
    wait_seconds: int
    seed: int


@dataclass(frozen=True)
class SceneLayout:
    frame_size: int
    scale: float
    slot_xs: tuple[float, float, float]
    ball_y: float
    cup_base_y: float
    cup_start_y: float
    cup_width: float
    cup_height: float
    ball_radius: float
    cup_lip_height: float
    reveal_lift: float


def build_layout(frame_size: int) -> SceneLayout:
    scale = float(frame_size) / BASE_FRAME_SIZE
    return SceneLayout(
        frame_size=frame_size,
        scale=scale,
        slot_xs=tuple(value * scale for value in BASE_SLOT_XS),
        ball_y=BASE_BALL_Y * scale,
        cup_base_y=BASE_CUP_BASE_Y * scale,
        cup_start_y=BASE_CUP_START_Y * scale,
        cup_width=BASE_CUP_WIDTH * scale,
        cup_height=BASE_CUP_HEIGHT * scale,
        ball_radius=max(2.0, BASE_BALL_RADIUS * scale),
        cup_lip_height=BASE_CUP_LIP_HEIGHT * scale,
        reveal_lift=BASE_REVEAL_LIFT * scale,
    )


def ref_filename(index: int) -> str:
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


def expand_wait_schedule(config: ShellGameMiniConfig) -> list[int]:
    if config.wait_values:
        schedule: list[int] = []
        for wait_value, count in zip(config.wait_values, config.wait_counts):
            schedule.extend([wait_value] * count)
    else:
        schedule = [config.occlusion_seconds] * config.num_videos
    if config.shuffle_wait_values:
        random.Random(config.seed).shuffle(schedule)
    return schedule


def build_tasks(config: ShellGameMiniConfig) -> list[VideoTask]:
    schedule = expand_wait_schedule(config)
    return [VideoTask(index=index, wait_seconds=wait_seconds, seed=config.seed + index) for index, wait_seconds in enumerate(schedule)]


def base_dirs(config: ShellGameMiniConfig) -> tuple[Path, Path, Path]:
    root = Path(config.output_base_dir)
    ref_dir = root / "ref_videos"
    target_dir = root / "target_videos"
    struct_dir = root / "npy"
    return ref_dir, target_dir, struct_dir


def ensure_dirs(config: ShellGameMiniConfig) -> tuple[Path, Path, Path]:
    ref_dir, target_dir, struct_dir = base_dirs(config)
    ref_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir, target_dir, struct_dir


def cup_center_at(slot_index: int, layout: SceneLayout) -> np.ndarray:
    return np.asarray([layout.slot_xs[slot_index], layout.cup_base_y], dtype=np.float32)


def ball_center_at(slot_index: int, layout: SceneLayout) -> np.ndarray:
    return np.asarray([layout.slot_xs[slot_index], layout.ball_y], dtype=np.float32)


def draw_ball(draw: ImageDraw.ImageDraw, center: np.ndarray, layout: SceneLayout) -> None:
    x, y = float(center[0]), float(center[1])
    radius = float(layout.ball_radius)
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=BALL_RGB)


def draw_cup(draw: ImageDraw.ImageDraw, center: np.ndarray, layout: SceneLayout) -> None:
    x, y = float(center[0]), float(center[1])
    left = x - layout.cup_width / 2.0
    right = x + layout.cup_width / 2.0
    top = y - layout.cup_height / 2.0
    bottom = y + layout.cup_height / 2.0
    lip_h = layout.cup_lip_height
    body_points = [
        (left + 2.0 * layout.scale, bottom),
        (left + 6.0 * layout.scale, top + lip_h),
        (right - 6.0 * layout.scale, top + lip_h),
        (right - 2.0 * layout.scale, bottom),
    ]
    draw.polygon(body_points, fill=CUP_RGB)
    draw.rounded_rectangle(
        [(left + 4.0 * layout.scale, top), (right - 4.0 * layout.scale, top + lip_h + 6.0 * layout.scale)],
        radius=max(2.0, 5.0 * layout.scale),
        fill=CUP_RGB,
    )


def render_scene(
    ball_slot: int | None,
    cup_positions: dict[int, np.ndarray],
    layout: SceneLayout,
    lifted_cup_id: int | None = None,
) -> np.ndarray:
    image = Image.new("RGB", (layout.frame_size, layout.frame_size), color=BACKGROUND_RGB)
    draw = ImageDraw.Draw(image)
    if ball_slot is not None:
        draw_ball(draw, ball_center_at(ball_slot, layout), layout)
    for cup_id in sorted(cup_positions.keys()):
        if lifted_cup_id is not None and cup_id == lifted_cup_id:
            continue
        draw_cup(draw, cup_positions[cup_id], layout)
    if lifted_cup_id is not None:
        draw_cup(draw, cup_positions[lifted_cup_id], layout)
    return np.asarray(image, dtype=np.uint8)


def initial_cup_positions(layout: SceneLayout) -> dict[int, np.ndarray]:
    return {cup_id: np.asarray([layout.slot_xs[cup_id], layout.cup_start_y], dtype=np.float32) for cup_id in range(3)}


def settled_cup_positions(layout: SceneLayout) -> dict[int, np.ndarray]:
    return {cup_id: cup_center_at(cup_id, layout) for cup_id in range(3)}


def descent_frames(layout: SceneLayout) -> list[dict[int, np.ndarray]]:
    start = initial_cup_positions(layout)
    end = settled_cup_positions(layout)
    frames: list[dict[int, np.ndarray]] = []
    for alpha in np.linspace(0.0, 1.0, DESCENT_FRAMES):
        frame_positions = {
            cup_id: ((1.0 - alpha) * start[cup_id] + alpha * end[cup_id]).astype(np.float32)
            for cup_id in range(3)
        }
        frames.append(frame_positions)
    return frames


def reveal_animation(winning_cup_id: int, layout: SceneLayout) -> list[dict[int, np.ndarray]]:
    base_positions = settled_cup_positions(layout)
    frames: list[dict[int, np.ndarray]] = []
    for alpha in np.linspace(0.0, 1.0, REVEAL_FRAMES):
        positions = {cup_id: position.copy() for cup_id, position in base_positions.items()}
        positions[winning_cup_id] = np.asarray(
            [positions[winning_cup_id][0], layout.cup_base_y - layout.reveal_lift * float(alpha)],
            dtype=np.float32,
        )
        frames.append(positions)
    return frames


def ref_video_frames(initial_ball_slot: int, wait_seconds: int, layout: SceneLayout, fps: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for _ in range(INITIAL_HOLD_FRAMES):
        frames.append(render_scene(initial_ball_slot, {}, layout))
    for positions in descent_frames(layout):
        frames.append(render_scene(initial_ball_slot, positions, layout))
    settled_positions = settled_cup_positions(layout)
    for _ in range(wait_seconds * fps):
        frames.append(render_scene(None, settled_positions, layout))
    return frames


def target_video_frames(final_ball_slot: int, winning_cup_id: int, layout: SceneLayout) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    reveal_frames = reveal_animation(winning_cup_id, layout)
    for positions in reveal_frames:
        frames.append(render_scene(final_ball_slot, positions, layout, lifted_cup_id=winning_cup_id))
    lifted_positions = reveal_frames[-1]
    for _ in range(TARGET_FINAL_HOLD):
        frames.append(render_scene(final_ball_slot, lifted_positions, layout, lifted_cup_id=winning_cup_id))
    return frames


def sample_payload(
    task: VideoTask,
    config: ShellGameMiniConfig,
) -> tuple[dict[str, object], dict[str, object], list[np.ndarray], list[np.ndarray]]:
    rng = random.Random(task.seed)
    layout = build_layout(config.frame_size)
    initial_ball_slot = rng.randrange(3)
    final_ball_slot = initial_ball_slot
    winning_cup_id = initial_ball_slot
    wait_frames = task.wait_seconds * config.fps
    ref_frames = ref_video_frames(initial_ball_slot, task.wait_seconds, layout, config.fps)
    target_frames = target_video_frames(final_ball_slot, winning_cup_id, layout)

    metadata = {
        "index": task.index,
        "seed": task.seed,
        "occlusion_seconds": task.wait_seconds,
        "wait_seconds": task.wait_seconds,
        "wait_frames": wait_frames,
        "fps": config.fps,
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
        "ref_file": ref_filename(task.index),
        "target_file": target_filename(task.index),
        "npy_file": struct_filename(task.index),
        "initial_ball_slot": int(initial_ball_slot),
        "final_ball_slot": int(final_ball_slot),
        "winning_cup_id": int(winning_cup_id),
        "num_swaps": 0,
        "swap_sequence": [],
        "ref_num_frames": len(ref_frames),
        "target_num_frames": len(target_frames),
    }
    struct = {
        "index": np.int16(task.index),
        "seed": np.int32(task.seed),
        "occlusion_seconds": np.int16(task.wait_seconds),
        "wait_seconds": np.int16(task.wait_seconds),
        "wait_frames": np.int16(wait_frames),
        "fps": np.int16(config.fps),
        "frame_size": np.int16(config.frame_size),
        "initial_ball_slot": np.int16(initial_ball_slot),
        "final_ball_slot": np.int16(final_ball_slot),
        "winning_cup_id": np.int16(winning_cup_id),
        "num_swaps": np.int16(0),
        "swap_sequence": np.empty((0, 2), dtype=np.int16),
        "slot_xs": np.asarray(layout.slot_xs, dtype=np.float32),
        "ball_y": np.float32(layout.ball_y),
        "cup_base_y": np.float32(layout.cup_base_y),
    }
    return metadata, struct, ref_frames, target_frames


def process_task(task: VideoTask, config: ShellGameMiniConfig) -> dict[str, object]:
    ref_dir, target_dir, struct_dir = ensure_dirs(config)
    ref_path = ref_dir / ref_filename(task.index)
    target_path = target_dir / target_filename(task.index)
    struct_path = struct_dir / struct_filename(task.index)

    if not config.overwrite and ref_path.exists() and target_path.exists() and struct_path.exists():
        struct = np.load(struct_path, allow_pickle=True).item()
        return {
            "index": int(struct["index"]),
            "seed": int(struct["seed"]),
            "occlusion_seconds": int(struct["occlusion_seconds"]),
            "wait_seconds": int(struct["wait_seconds"]),
            "wait_frames": int(struct["wait_frames"]),
            "fps": int(struct["fps"]),
            "frame_height": int(struct["frame_size"]),
            "frame_width": int(struct["frame_size"]),
            "ref_file": ref_filename(task.index),
            "target_file": target_filename(task.index),
            "npy_file": struct_filename(task.index),
            "initial_ball_slot": int(struct["initial_ball_slot"]),
            "final_ball_slot": int(struct["final_ball_slot"]),
            "winning_cup_id": int(struct["winning_cup_id"]),
            "num_swaps": int(struct["num_swaps"]),
            "swap_sequence": np.asarray(struct["swap_sequence"]).tolist(),
            "ref_num_frames": int(imageio.get_reader(ref_path).count_frames()),
            "target_num_frames": int(imageio.get_reader(target_path).count_frames()),
        }

    metadata, struct, ref_frames, target_frames = sample_payload(task, config)
    with imageio.get_writer(ref_path, fps=config.fps, codec=config.codec, quality=config.quality, macro_block_size=1) as writer:
        for frame in ref_frames:
            writer.append_data(frame)
    with imageio.get_writer(target_path, fps=config.fps, codec=config.codec, quality=config.quality, macro_block_size=1) as writer:
        for frame in target_frames:
            writer.append_data(frame)
    np.save(struct_path, struct, allow_pickle=True)
    return metadata


def write_metadata(config: ShellGameMiniConfig, records: list[dict[str, object]]) -> None:
    root = Path(config.output_base_dir)
    with (root / config.metadata_name).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(config: ShellGameMiniConfig) -> list[dict[str, object]]:
    config.validate()
    ensure_dirs(config)
    tasks = build_tasks(config)
    records: list[dict[str, object]] = []
    if config.workers == 1:
        for task in tqdm(tasks, desc="Generating shell-game mini videos", leave=True):
            records.append(process_task(task, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            iterator = executor.map(process_task, tasks, repeat(config))
            for record in tqdm(iterator, total=len(tasks), desc="Generating shell-game mini videos", leave=True):
                records.append(record)
    records.sort(key=lambda item: int(item["index"]))
    write_metadata(config, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate shell-game mini ref/target videos.")
    parser.add_argument("--occlusion-seconds", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--wait-values", type=str, default="1,2,3,4,5")
    parser.add_argument("--wait-counts", type=str, default="600,600,600,600,600")
    parser.add_argument("--keep-wait-order", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-base-dir", type=str, default="shell_game_mini_dataset/train")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ShellGameMiniConfig(
        occlusion_seconds=args.occlusion_seconds,
        num_videos=args.num_videos,
        wait_values=parse_int_list(args.wait_values) if args.wait_values else (),
        wait_counts=parse_int_list(args.wait_counts) if args.wait_counts else (),
        shuffle_wait_values=not args.keep_wait_order,
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
