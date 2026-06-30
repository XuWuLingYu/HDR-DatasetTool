#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
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


SHAPE_NAMES = ("ball", "square", "triangle", "parallelogram", "star")


@dataclass(frozen=True)
class ShapeReturnConfig:
    shape_name: str = "ball"
    num_videos: int = 3000
    shape_values: tuple[str, ...] = ()
    shape_counts: tuple[int, ...] = ()
    shuffle_shape_values: bool = True
    fps: int = 20
    move_seconds: float = 1.0
    offscreen_wait_seconds: float = 1.0
    frame_size: int = 208
    output_dir: str = "outputs/shape_return/videos"
    struct_dir: str = "outputs/shape_return/struct"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260412
    workers: int = 1
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7

    def validate(self) -> None:
        if self.shape_name not in SHAPE_NAMES:
            raise ValueError(f"shape_name must be one of {SHAPE_NAMES}")
        if self.num_videos < 1:
            raise ValueError("num_videos must be >= 1")
        if self.fps < 1:
            raise ValueError("fps must be >= 1")
        if self.move_seconds <= 0:
            raise ValueError("move_seconds must be > 0")
        if self.offscreen_wait_seconds < 0:
            raise ValueError("offscreen_wait_seconds must be >= 0")
        if self.frame_size < 64:
            raise ValueError("frame_size must be >= 64")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if bool(self.shape_values) != bool(self.shape_counts):
            raise ValueError("shape_values and shape_counts must be provided together")
        if self.shape_values and len(self.shape_values) != len(self.shape_counts):
            raise ValueError("shape_values and shape_counts must have the same length")
        if self.shape_values and sum(self.shape_counts) != self.num_videos:
            raise ValueError("sum(shape_counts) must equal num_videos")
        if self.shape_values and any(name not in SHAPE_NAMES for name in self.shape_values):
            raise ValueError(f"all shape_values must be in {SHAPE_NAMES}")

    @property
    def move_frames(self) -> int:
        return max(1, int(round(self.move_seconds * self.fps)))

    @property
    def wait_frames(self) -> int:
        return max(0, int(round(self.offscreen_wait_seconds * self.fps)))


@dataclass(frozen=True)
class VideoTask:
    index: int
    shape_name: str
    seed: int


def video_name(index: int) -> str:
    return f"shape_return_{index:04d}.mp4"


def npy_name(index: int) -> str:
    return f"shape_return_{index:04d}.npy"


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of integers")
    return tuple(int(value) for value in values)


def parse_shape_list(text: str) -> tuple[str, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of shape names")
    return tuple(values)


def expand_shape_schedule(config: ShapeReturnConfig) -> list[str]:
    if config.shape_values:
        schedule: list[str] = []
        for shape_name, count in zip(config.shape_values, config.shape_counts):
            schedule.extend([shape_name] * count)
    else:
        schedule = [config.shape_name] * config.num_videos
    if config.shuffle_shape_values:
        random.Random(config.seed).shuffle(schedule)
    return schedule


def build_tasks(config: ShapeReturnConfig) -> list[VideoTask]:
    schedule = expand_shape_schedule(config)
    return [VideoTask(index=index, shape_name=shape_name, seed=config.seed + index) for index, shape_name in enumerate(schedule)]


def sample_nonwhite_color(rng: random.Random) -> np.ndarray:
    while True:
        hue = rng.random()
        saturation = 0.5 + 0.4 * rng.random()
        value = 0.35 + 0.45 * rng.random()
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb = np.asarray([int(round(red * 255)), int(round(green * 255)), int(round(blue * 255))], dtype=np.uint8)
        if int(rgb.max()) <= 225 and int(rgb.mean()) <= 190:
            return rgb


def regular_polygon(center: tuple[float, float], radius: float, sides: int, rotation: float) -> list[tuple[float, float]]:
    cx, cy = center
    points: list[tuple[float, float]] = []
    for idx in range(sides):
        angle = rotation + 2.0 * math.pi * idx / sides
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def star_polygon(center: tuple[float, float], outer_radius: float, inner_radius: float, rotation: float) -> list[tuple[float, float]]:
    cx, cy = center
    points: list[tuple[float, float]] = []
    for idx in range(10):
        angle = rotation + math.pi * idx / 5.0
        radius = outer_radius if idx % 2 == 0 else inner_radius
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def create_shape_sprite(shape_name: str, size: int, color: np.ndarray, shear_px: int = 0) -> tuple[np.ndarray, np.ndarray]:
    canvas_size = int(size) + 8 + abs(int(shear_px))
    image = Image.new("RGBA", (canvas_size, canvas_size), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = tuple(int(channel) for channel in color.tolist()) + (255,)
    outline = (30, 30, 30, 255)
    left = 4 + max(0, int(shear_px))
    top = 4
    right = left + int(size)
    bottom = top + int(size)
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    radius = size / 2.0

    if shape_name == "ball":
        draw.ellipse([(left, top), (right, bottom)], fill=fill, outline=outline, width=1)
    elif shape_name == "square":
        draw.rectangle([(left, top), (right, bottom)], fill=fill, outline=outline, width=1)
    elif shape_name == "triangle":
        points = regular_polygon((cx, cy), radius, sides=3, rotation=-math.pi / 2.0)
        draw.polygon(points, fill=fill, outline=outline)
    elif shape_name == "parallelogram":
        skew = int(shear_px)
        points = [
            (left + skew, top),
            (right + skew, top),
            (right - skew, bottom),
            (left - skew, bottom),
        ]
        draw.polygon(points, fill=fill, outline=outline)
    elif shape_name == "star":
        points = star_polygon((cx, cy), outer_radius=radius, inner_radius=radius * 0.45, rotation=-math.pi / 2.0)
        draw.polygon(points, fill=fill, outline=outline)
    else:
        raise ValueError(f"unknown shape {shape_name}")

    arr = np.asarray(image, dtype=np.uint8)
    return arr[..., :3], arr[..., 3] > 0


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
    source = sprite_rgb[src_top:src_bottom, src_left:src_right]
    target = frame[clip_top:clip_bottom, clip_left:clip_right]
    target[mask] = source[mask]


def background(frame_size: int) -> np.ndarray:
    return np.full((frame_size, frame_size, 3), fill_value=255, dtype=np.uint8)


def direction_unit(direction: str) -> np.ndarray:
    mapping = {
        "left": np.asarray([0.0, -1.0], dtype=np.float32),
        "right": np.asarray([0.0, 1.0], dtype=np.float32),
        "up": np.asarray([-1.0, 0.0], dtype=np.float32),
        "down": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    return mapping[direction]


def center_point(frame_size: int) -> np.ndarray:
    center = (frame_size - 1) / 2.0
    return np.asarray([center, center], dtype=np.float32)


def offscreen_center(start_center: np.ndarray, direction: str, sprite_shape: tuple[int, int], frame_size: int, overshoot: float) -> np.ndarray:
    half_h = sprite_shape[0] / 2.0
    half_w = sprite_shape[1] / 2.0
    if direction == "left":
        return np.asarray([start_center[0], -half_w - overshoot], dtype=np.float32)
    if direction == "right":
        return np.asarray([start_center[0], frame_size + half_w + overshoot], dtype=np.float32)
    if direction == "up":
        return np.asarray([-half_h - overshoot, start_center[1]], dtype=np.float32)
    return np.asarray([frame_size + half_h + overshoot, start_center[1]], dtype=np.float32)


def interpolate_positions(start: np.ndarray, end: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps <= 0:
        return []
    return [
        ((1.0 - alpha) * start + alpha * end).astype(np.float32)
        for alpha in np.linspace(1.0 / steps, 1.0, steps)
    ]


def trajectory(start: np.ndarray, end: np.ndarray, move_frames: int, wait_frames: int) -> list[np.ndarray]:
    out = [start.astype(np.float32)]
    out.extend(interpolate_positions(start, end, move_frames))
    out.extend([end.astype(np.float32) for _ in range(wait_frames)])
    out.extend(interpolate_positions(end, start, move_frames))
    return out


def sample_payload(task: VideoTask, config: ShapeReturnConfig) -> tuple[dict[str, object], dict[str, object], list[np.ndarray]]:
    rng = random.Random(task.seed)
    frame = background(config.frame_size)
    shape_size = rng.randint(int(round(config.frame_size * 0.12)), int(round(config.frame_size * 0.2)))
    shear_px = rng.randint(4, max(5, shape_size // 4)) if task.shape_name == "parallelogram" else 0
    color = sample_nonwhite_color(rng)
    sprite_rgb, sprite_mask = create_shape_sprite(task.shape_name, shape_size, color, shear_px=shear_px)
    start = center_point(config.frame_size)
    direction = "right"
    end = offscreen_center(start, direction, sprite_rgb.shape[:2], config.frame_size, overshoot=8.0)
    positions = trajectory(start, end, config.move_frames, config.wait_frames)
    visible = []
    frames: list[np.ndarray] = []
    for position in positions:
        current = frame.copy()
        overlay_sprite(current, sprite_rgb, sprite_mask, float(position[0]), float(position[1]))
        frames.append(current)
        visible.append(
            bool(
                (-sprite_rgb.shape[0] <= position[0] <= config.frame_size + sprite_rgb.shape[0])
                and (-sprite_rgb.shape[1] <= position[1] <= config.frame_size + sprite_rgb.shape[1])
            )
        )

    metadata = {
        "index": task.index,
        "seed": task.seed,
        "file": video_name(task.index),
        "npy_file": npy_name(task.index),
        "shape_name": task.shape_name,
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
        "fps": config.fps,
        "move_seconds": config.move_seconds,
        "move_frames": config.move_frames,
        "offscreen_wait_seconds": config.offscreen_wait_seconds,
        "wait_frames": config.wait_frames,
        "num_frames": len(frames),
        "color_rgb": color.tolist(),
        "background_rgb": [255, 255, 255],
        "shape_size": int(shape_size),
        "direction": direction,
        "start_center": start.tolist(),
        "offscreen_center": end.tolist(),
    }

    struct = {
        "index": task.index,
        "seed": task.seed,
        "file": video_name(task.index),
        "shape_name": task.shape_name,
        "frame_size": config.frame_size,
        "fps": config.fps,
        "move_frames": np.int16(config.move_frames),
        "wait_frames": np.int16(config.wait_frames),
        "positions": np.asarray(positions, dtype=np.float32),
        "visible": np.asarray(visible, dtype=bool),
        "color_rgb": color.astype(np.uint8),
        "background_rgb": np.asarray([255, 255, 255], dtype=np.uint8),
        "shape_size": np.int16(shape_size),
        "direction": direction,
        "sprite_shape": np.asarray(sprite_rgb.shape[:2], dtype=np.int16),
        "start_center": start.astype(np.float32),
        "offscreen_center": end.astype(np.float32),
    }
    return metadata, struct, frames


def process_task(task: VideoTask, config: ShapeReturnConfig) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    struct_dir = Path(config.struct_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / video_name(task.index)
    struct_path = struct_dir / npy_name(task.index)

    if not config.overwrite and video_path.exists() and struct_path.exists():
        struct = np.load(struct_path, allow_pickle=True).item()
        return {
            "index": task.index,
            "seed": task.seed,
            "file": video_name(task.index),
            "npy_file": npy_name(task.index),
            "shape_name": str(struct["shape_name"]),
            "frame_height": int(struct["frame_size"]),
            "frame_width": int(struct["frame_size"]),
            "fps": int(struct["fps"]),
            "move_seconds": float(config.move_seconds),
            "move_frames": int(struct["move_frames"]),
            "offscreen_wait_seconds": float(config.offscreen_wait_seconds),
            "wait_frames": int(struct["wait_frames"]),
            "num_frames": int(len(struct["positions"])),
            "color_rgb": np.asarray(struct["color_rgb"]).tolist(),
            "background_rgb": [255, 255, 255],
            "shape_size": int(struct["shape_size"]),
            "direction": str(struct["direction"]),
            "start_center": np.asarray(struct["start_center"]).tolist(),
            "offscreen_center": np.asarray(struct["offscreen_center"]).tolist(),
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


def write_metadata(config: ShapeReturnConfig, records: list[dict[str, object]]) -> None:
    path = Path(config.output_dir) / config.metadata_name
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(config: ShapeReturnConfig) -> list[dict[str, object]]:
    config.validate()
    tasks = build_tasks(config)
    records: list[dict[str, object]] = []
    if config.workers == 1:
        for task in tqdm(tasks, desc="Generating shape-return videos", leave=True):
            records.append(process_task(task, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            iterator = executor.map(process_task, tasks, repeat(config))
            for record in tqdm(iterator, total=len(tasks), desc="Generating shape-return videos", leave=True):
                records.append(record)
    records.sort(key=lambda item: int(item["index"]))
    write_metadata(config, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a shape out-and-back video dataset.")
    parser.add_argument("--shape-name", type=str, default="ball")
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--shape-values", type=str, default="ball,square,triangle")
    parser.add_argument("--shape-counts", type=str, default="1000,1000,1000")
    parser.add_argument("--keep-shape-order", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--move-seconds", type=float, default=1.0)
    parser.add_argument("--offscreen-wait-seconds", type=float, default=1.0)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-dir", type=str, default="outputs/shape_return/videos")
    parser.add_argument("--struct-dir", type=str, default="outputs/shape_return/struct")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ShapeReturnConfig(
        shape_name=args.shape_name,
        num_videos=args.num_videos,
        shape_values=parse_shape_list(args.shape_values) if args.shape_values else (),
        shape_counts=parse_int_list(args.shape_counts) if args.shape_counts else (),
        shuffle_shape_values=not args.keep_shape_order,
        fps=args.fps,
        move_seconds=args.move_seconds,
        offscreen_wait_seconds=args.offscreen_wait_seconds,
        frame_size=args.frame_size,
        output_dir=args.output_dir,
        struct_dir=args.struct_dir,
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
