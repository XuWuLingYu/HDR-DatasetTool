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
class BounceBallConfig:
    bounce_count: int = 1
    num_videos: int = 3000
    bounce_values: tuple[int, ...] = ()
    bounce_counts: tuple[int, ...] = ()
    shuffle_bounce_values: bool = True
    fps: int = 20
    pixels_per_frame: float = 4.0
    frame_size: int = 208
    output_dir: str = "bounce_ball_dataset/bounce_ball_videos"
    struct_dir: str = "bounce_ball_dataset/bounce_ball_npy"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260412
    workers: int = 1
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7

    def validate(self) -> None:
        if self.bounce_count < 1:
            raise ValueError("bounce_count must be >= 1")
        if self.num_videos < 1:
            raise ValueError("num_videos must be >= 1")
        if self.fps < 1:
            raise ValueError("fps must be >= 1")
        if self.pixels_per_frame <= 0:
            raise ValueError("pixels_per_frame must be > 0")
        if self.frame_size < 96:
            raise ValueError("frame_size must be >= 96")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if bool(self.bounce_values) != bool(self.bounce_counts):
            raise ValueError("bounce_values and bounce_counts must be provided together")
        if self.bounce_values and len(self.bounce_values) != len(self.bounce_counts):
            raise ValueError("bounce_values and bounce_counts must have the same length")
        if self.bounce_values and sum(self.bounce_counts) != self.num_videos:
            raise ValueError("sum(bounce_counts) must equal num_videos")
        if self.bounce_values and any(value < 1 for value in self.bounce_values):
            raise ValueError("all bounce_values must be >= 1")


@dataclass(frozen=True)
class VideoTask:
    index: int
    bounce_count: int
    seed: int


ARENA_MARGIN = 20
BALL_RADIUS = 5
CANDIDATE_RADIUS = 6
CANDIDATE_COUNT = 5
START_ZONE_X = (0.08, 0.28)
START_ZONE_Y = (0.72, 0.90)
CANDIDATE_ZONE_X = (0.58, 0.84)
CANDIDATE_ZONE_Y = (0.14, 0.34)
SAFE_MARGIN = 10.0
PATH_CLEARANCE = BALL_RADIUS + CANDIDATE_RADIUS + 4.0
CANDIDATE_WALL_CLEARANCE = CANDIDATE_RADIUS + 4.0
CANDIDATE_SPACING_MIN = PATH_CLEARANCE + 3.0


def video_name(index: int) -> str:
    return f"bounce_ball_{index:04d}.mp4"


def npy_name(index: int) -> str:
    return f"bounce_ball_{index:04d}.npy"


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of integers")
    return tuple(int(value) for value in values)


def expand_bounce_schedule(config: BounceBallConfig) -> list[int]:
    if config.bounce_values:
        schedule: list[int] = []
        for bounce_count, count in zip(config.bounce_values, config.bounce_counts):
            schedule.extend([bounce_count] * count)
    else:
        schedule = [config.bounce_count] * config.num_videos
    if config.shuffle_bounce_values:
        random.Random(config.seed).shuffle(schedule)
    return schedule


def build_tasks(config: BounceBallConfig) -> list[VideoTask]:
    schedule = expand_bounce_schedule(config)
    return [VideoTask(index=index, bounce_count=bounce_count, seed=config.seed + index) for index, bounce_count in enumerate(schedule)]


def arena_size(frame_size: int) -> float:
    return float(frame_size - 2 * (ARENA_MARGIN + BALL_RADIUS))


def to_frame_coords(point: np.ndarray) -> np.ndarray:
    return point + np.asarray([ARENA_MARGIN + BALL_RADIUS, ARENA_MARGIN + BALL_RADIUS], dtype=np.float32)


def fold_coordinate(value: float, length: float) -> float:
    quotient = math.floor(value / length)
    remainder = value - quotient * length
    if quotient % 2 == 0:
        return remainder
    return length - remainder


def reflect_target_coordinate(value: float, length: float, reflections: int) -> float:
    if reflections % 2 == 0:
        return reflections * length + value
    return reflections * length + (length - value)


def folded_point(point: np.ndarray, width: float, height: float) -> np.ndarray:
    return np.asarray(
        [
            fold_coordinate(float(point[0]), width),
            fold_coordinate(float(point[1]), height),
        ],
        dtype=np.float32,
    )


def distance_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    length_sq = float(np.dot(ab, ab))
    if length_sq <= 1e-8:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / length_sq)
    t = max(0.0, min(1.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def sample_start_point(width: float, height: float, rng: random.Random) -> np.ndarray:
    return np.asarray(
        [
            rng.uniform(width * START_ZONE_X[0], width * START_ZONE_X[1]),
            rng.uniform(height * START_ZONE_Y[0], height * START_ZONE_Y[1]),
        ],
        dtype=np.float32,
    )


def sample_candidate_points(width: float, height: float, rng: random.Random) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < CANDIDATE_COUNT and attempts < 1000:
        attempts += 1
        x = rng.uniform(width * CANDIDATE_ZONE_X[0], width * CANDIDATE_ZONE_X[1])
        y = rng.uniform(height * CANDIDATE_ZONE_Y[0], height * CANDIDATE_ZONE_Y[1])
        candidate = np.asarray([x, y], dtype=np.float32)
        if any(float(np.linalg.norm(candidate - point)) < 2.1 * CANDIDATE_RADIUS for point in points):
            continue
        points.append(candidate)
    if len(points) != CANDIDATE_COUNT:
        raise RuntimeError("failed to sample candidate points")
    points.sort(key=lambda point: (float(point[0]), float(point[1])))
    return points


def max_offset_to_bounds(point: np.ndarray, direction: np.ndarray, width: float, height: float, clearance: float) -> float:
    limit = float("inf")
    if abs(float(direction[0])) > 1e-8:
        if float(direction[0]) > 0:
            limit = min(limit, (width - clearance - float(point[0])) / float(direction[0]))
        else:
            limit = min(limit, (clearance - float(point[0])) / float(direction[0]))
    if abs(float(direction[1])) > 1e-8:
        if float(direction[1]) > 0:
            limit = min(limit, (height - clearance - float(point[1])) / float(direction[1]))
        else:
            limit = min(limit, (clearance - float(point[1])) / float(direction[1]))
    return max(0.0, float(limit))


def build_candidate_line(
    target: np.ndarray,
    final_vector: np.ndarray,
    width: float,
    height: float,
    rng: random.Random,
) -> tuple[list[np.ndarray], int] | None:
    norm = float(np.linalg.norm(final_vector))
    if norm <= 1e-8:
        return None
    tangent = final_vector / norm
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    correct_index = rng.randrange(CANDIDATE_COUNT)
    steps_negative = correct_index
    steps_positive = CANDIDATE_COUNT - 1 - correct_index
    positive_limit = max_offset_to_bounds(target, normal, width, height, CANDIDATE_WALL_CLEARANCE)
    negative_limit = max_offset_to_bounds(target, -normal, width, height, CANDIDATE_WALL_CLEARANCE)
    max_spacing = float("inf")
    if steps_positive > 0:
        max_spacing = min(max_spacing, positive_limit / steps_positive)
    if steps_negative > 0:
        max_spacing = min(max_spacing, negative_limit / steps_negative)
    if max_spacing < CANDIDATE_SPACING_MIN:
        return None

    spacing = rng.uniform(CANDIDATE_SPACING_MIN, max_spacing)
    points = [
        (target + (idx - correct_index) * spacing * normal).astype(np.float32)
        for idx in range(CANDIDATE_COUNT)
    ]
    return points, correct_index


def boundary_crossings(start: np.ndarray, target_reflected: np.ndarray, width: float, height: float) -> list[tuple[float, str]]:
    delta = target_reflected - start
    crossings: list[tuple[float, str]] = []
    if abs(float(delta[0])) > 1e-6:
        x_min = math.floor(min(float(start[0]), float(target_reflected[0])) / width) + 1
        x_max = math.floor(max(float(start[0]), float(target_reflected[0])) / width)
        for multiple in range(x_min, x_max + 1):
            boundary_x = multiple * width
            t = (boundary_x - float(start[0])) / float(delta[0])
            if 1e-6 < t < 1.0 - 1e-6:
                crossings.append((t, "vertical"))
    if abs(float(delta[1])) > 1e-6:
        y_min = math.floor(min(float(start[1]), float(target_reflected[1])) / height) + 1
        y_max = math.floor(max(float(start[1]), float(target_reflected[1])) / height)
        for multiple in range(y_min, y_max + 1):
            boundary_y = multiple * height
            t = (boundary_y - float(start[1])) / float(delta[1])
            if 1e-6 < t < 1.0 - 1e-6:
                crossings.append((t, "horizontal"))
    crossings.sort(key=lambda item: item[0])
    for (t_a, kind_a), (t_b, kind_b) in zip(crossings, crossings[1:]):
        if abs(t_a - t_b) < 1e-6 and kind_a != kind_b:
            raise ValueError("corner hit detected")
    return crossings


def choose_reflections(bounce_count: int, rng: random.Random) -> tuple[int, int]:
    splits = list(range(bounce_count + 1))
    rng.shuffle(splits)
    for vertical in splits:
        horizontal = bounce_count - vertical
        if vertical == 0 and horizontal == 0:
            continue
        return vertical, horizontal
    raise RuntimeError("failed to choose reflections")


def build_path(start: np.ndarray, target: np.ndarray, bounce_count: int, width: float, height: float, rng: random.Random) -> tuple[list[np.ndarray], tuple[int, int]]:
    choices = list(range(bounce_count + 1))
    rng.shuffle(choices)
    last_error: Exception | None = None
    for vertical_bounces in choices:
        horizontal_bounces = bounce_count - vertical_bounces
        reflected = np.asarray(
            [
                reflect_target_coordinate(float(target[0]), width, vertical_bounces),
                reflect_target_coordinate(float(target[1]), height, horizontal_bounces),
            ],
            dtype=np.float32,
        )
        try:
            crossings = boundary_crossings(start, reflected, width, height)
        except ValueError as error:
            last_error = error
            continue
        if len(crossings) != bounce_count:
            continue
        points = [start]
        delta = reflected - start
        for t, _kind in crossings:
            unfolded = start + float(t) * delta
            points.append(folded_point(unfolded, width, height))
        points.append(target)
        return points, (vertical_bounces, horizontal_bounces)
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to build bounce path")


def path_is_clear(path_points: list[np.ndarray], candidate_points: list[np.ndarray], correct_index: int) -> bool:
    for idx, candidate in enumerate(candidate_points):
        if idx == correct_index:
            continue
        for segment_index, (start, end) in enumerate(zip(path_points, path_points[1:])):
            if segment_index == len(path_points) - 2:
                continue
            if distance_point_to_segment(candidate, start, end) <= PATH_CLEARANCE:
                return False
    return True


def sample_case(bounce_count: int, frame_size: int, rng: random.Random) -> dict[str, object]:
    width = arena_size(frame_size)
    height = arena_size(frame_size)
    for _attempt in range(2000):
        candidate_seed_points = sample_candidate_points(width, height, rng)
        target = candidate_seed_points[rng.randrange(CANDIDATE_COUNT)]
        start = sample_start_point(width, height, rng)
        if float(np.linalg.norm(start - target)) < 0.35 * width:
            continue
        try:
            path_points, reflections = build_path(start, target, bounce_count, width, height, rng)
        except Exception:
            continue
        final_vector = path_points[-1] - path_points[-2]
        candidate_line = build_candidate_line(target, final_vector, width, height, rng)
        if candidate_line is None:
            continue
        candidate_points, correct_index = candidate_line
        if not path_is_clear(path_points, candidate_points, correct_index):
            continue
        initial_vector = path_points[1] - path_points[0]
        if float(np.linalg.norm(initial_vector)) < 1e-6:
            continue
        return {
            "start": start,
            "target": target,
            "candidate_points": candidate_points,
            "correct_index": correct_index,
            "path_points": path_points,
            "reflections": reflections,
            "initial_vector": initial_vector / float(np.linalg.norm(initial_vector)),
            "final_vector": final_vector / float(np.linalg.norm(final_vector)),
        }
    raise RuntimeError("failed to sample a valid bounce-ball case")


def polyline_length(points: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(end - start) for start, end in zip(points, points[1:])))


def sample_polyline(points: list[np.ndarray], pixels_per_frame: float) -> list[np.ndarray]:
    if len(points) < 2:
        return [points[0].copy()]
    total_length = polyline_length(points)
    move_frames = max(1, int(math.ceil(total_length / pixels_per_frame)))
    distances = [0.0]
    for start, end in zip(points, points[1:]):
        distances.append(distances[-1] + float(np.linalg.norm(end - start)))
    samples = np.linspace(0.0, total_length, move_frames + 1)
    positions: list[np.ndarray] = []
    segment_index = 0
    for sample in samples.tolist():
        if sample >= total_length - 1e-8:
            positions.append(points[-1].copy())
            continue
        while segment_index + 1 < len(distances) and sample > distances[segment_index + 1]:
            segment_index += 1
        start = points[segment_index]
        end = points[segment_index + 1]
        segment_start = distances[segment_index]
        segment_end = distances[segment_index + 1]
        if segment_end - segment_start <= 1e-8:
            positions.append(end.copy())
            continue
        alpha = (sample - segment_start) / (segment_end - segment_start)
        positions.append(((1.0 - alpha) * start + alpha * end).astype(np.float32))
    return positions


def draw_arrow(draw: ImageDraw.ImageDraw, center: np.ndarray, direction: np.ndarray) -> None:
    start = to_frame_coords(center)
    unit = direction / float(np.linalg.norm(direction))
    length = 22.0
    end = start + unit * length
    draw.line([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))], fill=(90, 90, 90), width=2)
    orth = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    head_back = end - unit * 6.0
    left = head_back + orth * 3.0
    right = head_back - orth * 3.0
    draw.polygon(
        [
            (float(end[0]), float(end[1])),
            (float(left[0]), float(left[1])),
            (float(right[0]), float(right[1])),
        ],
        fill=(90, 90, 90),
    )


def render_frame(frame_size: int, candidate_points: list[np.ndarray], ball_center: np.ndarray, arrow_direction: np.ndarray | None) -> np.ndarray:
    image = Image.new("RGB", (frame_size, frame_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [(ARENA_MARGIN, ARENA_MARGIN), (frame_size - ARENA_MARGIN, frame_size - ARENA_MARGIN)],
        outline=(180, 180, 180),
        width=2,
    )
    for candidate in candidate_points:
        cx, cy = to_frame_coords(candidate)
        draw.ellipse(
            [(cx - CANDIDATE_RADIUS, cy - CANDIDATE_RADIUS), (cx + CANDIDATE_RADIUS, cy + CANDIDATE_RADIUS)],
            outline=(205, 205, 205),
            width=2,
        )
    if arrow_direction is not None:
        draw_arrow(draw, ball_center, arrow_direction)
    bx, by = to_frame_coords(ball_center)
    draw.ellipse(
        [(bx - BALL_RADIUS, by - BALL_RADIUS), (bx + BALL_RADIUS, by + BALL_RADIUS)],
        fill=(20, 20, 20),
    )
    return np.asarray(image, dtype=np.uint8)


def sample_payload(task: VideoTask, config: BounceBallConfig) -> tuple[dict[str, object], dict[str, object], list[np.ndarray]]:
    rng = random.Random(task.seed)
    case = sample_case(task.bounce_count, config.frame_size, rng)
    positions = sample_polyline(case["path_points"], config.pixels_per_frame)
    frames: list[np.ndarray] = []
    for frame_index, position in enumerate(positions):
        frames.append(
            render_frame(
                config.frame_size,
                case["candidate_points"],
                position,
                case["initial_vector"] if frame_index == 0 else None,
            )
        )

    candidate_array = np.asarray(case["candidate_points"], dtype=np.float32)
    path_array = np.asarray(case["path_points"], dtype=np.float32)
    positions_array = np.asarray(positions, dtype=np.float32)
    bounce_points = path_array[1:-1]
    initial_angle_deg = math.degrees(math.atan2(float(case["initial_vector"][1]), float(case["initial_vector"][0])))
    final_angle_deg = math.degrees(math.atan2(float(case["final_vector"][1]), float(case["final_vector"][0])))

    metadata = {
        "index": task.index,
        "seed": task.seed,
        "file": video_name(task.index),
        "npy_file": npy_name(task.index),
        "bounce_count": task.bounce_count,
        "frame_height": config.frame_size,
        "frame_width": config.frame_size,
        "fps": config.fps,
        "pixels_per_frame": config.pixels_per_frame,
        "num_frames": len(frames),
        "ball_radius": BALL_RADIUS,
        "candidate_radius": CANDIDATE_RADIUS,
        "candidate_positions": candidate_array.tolist(),
        "correct_candidate_index": int(case["correct_index"]),
        "start_position": path_array[0].tolist(),
        "final_position": path_array[-1].tolist(),
        "bounce_points": bounce_points.tolist(),
        "initial_angle_deg": round(initial_angle_deg, 4),
        "final_angle_deg": round(final_angle_deg, 4),
        "vertical_bounces": int(case["reflections"][0]),
        "horizontal_bounces": int(case["reflections"][1]),
    }
    struct = {
        "index": task.index,
        "seed": task.seed,
        "file": video_name(task.index),
        "bounce_count": np.int16(task.bounce_count),
        "frame_size": np.int16(config.frame_size),
        "fps": np.int16(config.fps),
        "pixels_per_frame": np.float32(config.pixels_per_frame),
        "ball_radius": np.int16(BALL_RADIUS),
        "candidate_radius": np.int16(CANDIDATE_RADIUS),
        "candidate_positions": candidate_array,
        "correct_candidate_index": np.int16(case["correct_index"]),
        "path_points": path_array,
        "positions": positions_array,
        "initial_vector": np.asarray(case["initial_vector"], dtype=np.float32),
        "initial_angle_deg": np.float32(initial_angle_deg),
        "final_vector": np.asarray(case["final_vector"], dtype=np.float32),
        "final_angle_deg": np.float32(final_angle_deg),
        "vertical_bounces": np.int16(case["reflections"][0]),
        "horizontal_bounces": np.int16(case["reflections"][1]),
    }
    return metadata, struct, frames


def process_task(task: VideoTask, config: BounceBallConfig) -> dict[str, object]:
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
            "bounce_count": int(struct["bounce_count"]),
            "frame_height": int(struct["frame_size"]),
            "frame_width": int(struct["frame_size"]),
            "fps": int(struct["fps"]),
            "pixels_per_frame": float(struct["pixels_per_frame"]),
            "num_frames": int(len(struct["positions"])),
            "ball_radius": int(struct["ball_radius"]),
            "candidate_radius": int(struct["candidate_radius"]),
            "candidate_positions": np.asarray(struct["candidate_positions"]).tolist(),
            "correct_candidate_index": int(struct["correct_candidate_index"]),
            "start_position": np.asarray(struct["path_points"])[0].tolist(),
            "final_position": np.asarray(struct["path_points"])[-1].tolist(),
            "bounce_points": np.asarray(struct["path_points"])[1:-1].tolist(),
            "initial_angle_deg": float(struct["initial_angle_deg"]),
            "final_angle_deg": float(struct["final_angle_deg"]),
            "vertical_bounces": int(struct["vertical_bounces"]),
            "horizontal_bounces": int(struct["horizontal_bounces"]),
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


def write_metadata(config: BounceBallConfig, records: list[dict[str, object]]) -> None:
    path = Path(config.output_dir) / config.metadata_name
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(config: BounceBallConfig) -> list[dict[str, object]]:
    config.validate()
    tasks = build_tasks(config)
    records: list[dict[str, object]] = []
    if config.workers == 1:
        for task in tqdm(tasks, desc="Generating bounce-ball videos", leave=True):
            records.append(process_task(task, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            iterator = executor.map(process_task, tasks, repeat(config))
            for record in tqdm(iterator, total=len(tasks), desc="Generating bounce-ball videos", leave=True):
                records.append(record)
    records.sort(key=lambda item: int(item["index"]))
    write_metadata(config, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a bounce-ball multiple-choice dataset.")
    parser.add_argument("--bounce-count", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--bounce-values", type=str, default="1,2,3")
    parser.add_argument("--bounce-counts", type=str, default="1000,1000,1000")
    parser.add_argument("--keep-bounce-order", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--pixels-per-frame", type=float, default=4.0)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-dir", type=str, default="bounce_ball_dataset/bounce_ball_videos")
    parser.add_argument("--struct-dir", type=str, default="bounce_ball_dataset/bounce_ball_npy")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BounceBallConfig(
        bounce_count=args.bounce_count,
        num_videos=args.num_videos,
        bounce_values=parse_int_list(args.bounce_values) if args.bounce_values else (),
        bounce_counts=parse_int_list(args.bounce_counts) if args.bounce_counts else (),
        shuffle_bounce_values=not args.keep_bounce_order,
        fps=args.fps,
        pixels_per_frame=args.pixels_per_frame,
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
