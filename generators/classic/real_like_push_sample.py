#!/usr/bin/env python3
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


OUT_DIR = Path("real_like_push_sample")
OUT_VIDEO = OUT_DIR / "real_like_push_sample_0000.mp4"
OUT_CONTACT = OUT_DIR / "real_like_push_contact_sheet.png"

FRAME_SIZE = 224
NUM_FRAMES = 81
FPS = 16
BOARD_RECT = (14.0, 24.0, 210.0, 216.0)
BALL_RADIUS = 4.0
GRID_STEP = 2
CLEARANCE_RADIUS = BALL_RADIUS * 2.0 + 1.0
SAFE_PASSAGE = CLEARANCE_RADIUS
START = (34, 50)
GOAL = (194, 194)

Point = tuple[float, float]
GridPoint = tuple[int, int]


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0

    @property
    def aabb(self) -> tuple[float, float, float, float]:
        if self.angle == 0:
            return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)
        radians = math.radians(self.angle)
        corners = []
        for x, y in [(-self.w / 2, -self.h / 2), (self.w / 2, -self.h / 2), (self.w / 2, self.h / 2), (-self.w / 2, self.h / 2)]:
            corners.append((self.cx + x * math.cos(radians) - y * math.sin(radians), self.cy + x * math.sin(radians) + y * math.cos(radians)))
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class Layout:
    walls: tuple[Box, ...]
    block_start: GridPoint
    block_w: float
    block_h: float
    orientation: str


@dataclass(frozen=True)
class PlannedStep:
    ball: GridPoint
    block: GridPoint


@dataclass(frozen=True)
class MotionSegment:
    ball_start: Point
    ball_end: Point
    block_start: Point
    block_end: Point


def aabb_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float = 0.0) -> bool:
    return not (a[2] + margin <= b[0] or a[0] - margin >= b[2] or a[3] + margin <= b[1] or a[1] - margin >= b[3])


def point_in_board(point: Point, margin: float) -> bool:
    x, y = point
    left, top, right, bottom = BOARD_RECT
    return left + margin <= x <= right - margin and top + margin <= y <= bottom - margin


def box_in_board(box: Box, margin: float = 0.0) -> bool:
    left, top, right, bottom = BOARD_RECT
    bx0, by0, bx1, by1 = box.aabb
    return left + margin <= bx0 and top + margin <= by0 and bx1 <= right - margin and by1 <= bottom - margin


def distance_point_to_box(point: Point, box: Box) -> float:
    px, py = point
    radians = -math.radians(box.angle)
    local_x = (px - box.cx) * math.cos(radians) - (py - box.cy) * math.sin(radians)
    local_y = (px - box.cx) * math.sin(radians) + (py - box.cy) * math.cos(radians)
    dx = max(abs(local_x) - box.w / 2, 0.0)
    dy = max(abs(local_y) - box.h / 2, 0.0)
    return math.hypot(dx, dy)


def segment_to_box_clearance(start: Point, end: Point, box: Box, sample_step: float = 1.0) -> float:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    samples = max(1, math.ceil(length / sample_step))
    best = float("inf")
    for index in range(samples + 1):
        t = index / samples
        point = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        best = min(best, distance_point_to_box(point, box))
    return best


def block_box(center: GridPoint | Point, layout: Layout) -> Box:
    return Box(float(center[0]), float(center[1]), layout.block_w, layout.block_h)


def static_ball_clear(point: GridPoint | Point, walls: tuple[Box, ...], clearance: float = CLEARANCE_RADIUS) -> bool:
    p = (float(point[0]), float(point[1]))
    if not point_in_board(p, clearance):
        return False
    return all(distance_point_to_box(p, wall) >= clearance - 1e-6 for wall in walls)


def ball_clear(point: GridPoint | Point, block: GridPoint | Point, layout: Layout, clearance: float = CLEARANCE_RADIUS) -> bool:
    p = (float(point[0]), float(point[1]))
    if not static_ball_clear(p, layout.walls, clearance):
        return False
    return distance_point_to_box(p, block_box(block, layout)) >= clearance - 1e-6


def block_clear(center: GridPoint | Point, layout: Layout) -> bool:
    box = block_box(center, layout)
    if not box_in_board(box, 0.0):
        return False
    return all(not aabb_overlap(box.aabb, wall.aabb, margin=0.25) for wall in layout.walls)


def static_ball_clear_with_blocks(
    point: GridPoint | Point,
    walls: tuple[Box, ...],
    blocks: tuple[Box, ...],
    wall_clearance: float = CLEARANCE_RADIUS,
    block_clearance: float = CLEARANCE_RADIUS,
) -> bool:
    p = (float(point[0]), float(point[1]))
    if not static_ball_clear(p, walls, wall_clearance):
        return False
    return all(distance_point_to_box(p, block) >= block_clearance - 1e-6 for block in blocks)


def static_neighbors(
    point: GridPoint,
    walls: tuple[Box, ...],
    blocks: tuple[Box, ...],
    wall_clearance: float = CLEARANCE_RADIUS,
    block_clearance: float = CLEARANCE_RADIUS,
) -> list[GridPoint]:
    result: list[GridPoint] = []
    for dx, dy in [(GRID_STEP, 0), (-GRID_STEP, 0), (0, GRID_STEP), (0, -GRID_STEP)]:
        candidate = (point[0] + dx, point[1] + dy)
        if static_ball_clear_with_blocks(candidate, walls, blocks, wall_clearance, block_clearance):
            result.append(candidate)
    return result


def static_astar(
    start: GridPoint,
    goal: GridPoint,
    walls: tuple[Box, ...],
    blocks: tuple[Box, ...],
    wall_clearance: float = CLEARANCE_RADIUS,
    block_clearance: float = CLEARANCE_RADIUS,
) -> list[GridPoint] | None:
    if not static_ball_clear_with_blocks(start, walls, blocks, wall_clearance, block_clearance):
        return None
    if not static_ball_clear_with_blocks(goal, walls, blocks, wall_clearance, block_clearance):
        return None

    queue: list[tuple[int, int, GridPoint]] = []
    counter = 0
    heapq.heappush(queue, (0, counter, start))
    came_from: dict[GridPoint, GridPoint | None] = {start: None}
    cost_so_far: dict[GridPoint, int] = {start: 0}

    while queue:
        _, _, current = heapq.heappop(queue)
        if current == goal:
            break
        for next_point in static_neighbors(current, walls, blocks, wall_clearance, block_clearance):
            new_cost = cost_so_far[current] + 1
            if next_point in cost_so_far and new_cost >= cost_so_far[next_point]:
                continue
            cost_so_far[next_point] = new_cost
            came_from[next_point] = current
            heuristic = abs(next_point[0] - goal[0]) + abs(next_point[1] - goal[1])
            counter += 1
            heapq.heappush(queue, (new_cost * GRID_STEP + heuristic, counter, next_point))

    if goal not in came_from:
        return None

    path: list[GridPoint] = []
    current: GridPoint | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def make_plan(layout: Layout, push_distance: int = 40) -> list[PlannedStep] | None:
    dx, dy = 1, 0
    contact_distance = layout.block_w / 2 + BALL_RADIUS
    contact_offset = int(math.ceil(contact_distance / GRID_STEP) * GRID_STEP)
    contact_start = (layout.block_start[0] - contact_offset, layout.block_start[1])
    block_end = (layout.block_start[0] + push_distance, layout.block_start[1])
    contact_end = (contact_start[0] + push_distance, contact_start[1])

    if not block_clear(layout.block_start, layout) or not block_clear(block_end, layout):
        return None
    for step in range(0, push_distance + GRID_STEP, GRID_STEP):
        block = (layout.block_start[0] + step * dx, layout.block_start[1] + step * dy)
        ball = (contact_start[0] + step * dx, contact_start[1] + step * dy)
        if not block_clear(block, layout) or not static_ball_clear(ball, layout.walls, CLEARANCE_RADIUS):
            return None
        if distance_point_to_box(ball, block_box(block, layout)) < BALL_RADIUS - 1e-6:
            return None

    path_a = static_astar(START, contact_start, layout.walls, (block_box(layout.block_start, layout),), CLEARANCE_RADIUS, BALL_RADIUS)
    path_b = static_astar(contact_end, GOAL, layout.walls, (block_box(block_end, layout),), CLEARANCE_RADIUS, BALL_RADIUS)
    if path_a is None or path_b is None:
        return None

    plan: list[PlannedStep] = [PlannedStep(point, layout.block_start) for point in path_a]
    for step in range(GRID_STEP, push_distance + GRID_STEP, GRID_STEP):
        plan.append(PlannedStep((contact_start[0] + step, contact_start[1]), (layout.block_start[0] + step, layout.block_start[1])))
    plan.extend(PlannedStep(point, block_end) for point in path_b[1:])
    return plan


def reachable_without_push(layout: Layout, push_distance: int = 40) -> bool:
    return static_astar(START, GOAL, layout.walls, (block_box(layout.block_start, layout),), CLEARANCE_RADIUS, CLEARANCE_RADIUS) is not None


def add_wall(walls: list[Box], box: Box, protected: list[Box]) -> bool:
    if not box_in_board(box):
        return False
    if any(aabb_overlap(box.aabb, other.aabb, margin=1.5) for other in walls):
        return False
    if any(aabb_overlap(box.aabb, area.aabb, margin=0.0) for area in protected):
        return False
    walls.append(box)
    return True


def generate_layout(seed: int = 42) -> tuple[Layout, list[PlannedStep]]:
    rng = random.Random(seed)
    for attempt in range(80):
        orientation = "horizontal" if rng.random() < 0.5 else "vertical"
        block_w, block_h = (32.0, 18.0) if orientation == "horizontal" else (18.0, 32.0)
        block_start = (106, 126)
        block_sweep = Box(126, 126, 76, max(58.0, block_h + SAFE_PASSAGE * 2))
        protected = [
            Box(START[0], START[1], 36, 36),
            Box(GOAL[0], GOAL[1], 36, 36),
            block_sweep,
            Box(74, 126, 78, 34),
            Box(166, 126, 76, 60),
        ]

        walls: list[Box] = [
            Box(112, 29, 184, 7),
            Box(112, 211, 184, 7),
            Box(18, 120, 7, 174),
            Box(206, 120, 7, 174),
            Box(106, 69, 7, 80),
            Box(106, 183, 7, 56),
        ]

        # Build a full-height gate first; the black block is the movable door in the middle.
        gate_extra = [
            Box(77, 33, 42, 7),
            Box(154, 207, 42, 7),
        ]
        for box in gate_extra:
            add_wall(walls, box, protected[:3])

        layout = Layout(tuple(walls), block_start, block_w, block_h, orientation)
        plan = make_plan(layout)
        if plan is None or reachable_without_push(layout):
            continue

        for _ in range(140):
            horizontal = rng.random() < 0.5
            if horizontal:
                box = Box(rng.randrange(36, 192, 2), rng.randrange(42, 198, 2), rng.randrange(22, 46, 2), 7)
            else:
                box = Box(rng.randrange(36, 192, 2), rng.randrange(42, 198, 2), 7, rng.randrange(22, 50, 2))
            trial_walls = walls.copy()
            if not add_wall(trial_walls, box, protected):
                continue
            trial_layout = Layout(tuple(trial_walls), block_start, block_w, block_h, orientation)
            trial_plan = make_plan(trial_layout)
            if trial_plan is None or reachable_without_push(trial_layout):
                continue
            walls = trial_walls
            layout = trial_layout
            plan = trial_plan
            if len(walls) >= 32:
                break

        push_steps = sum(1 for prev, cur in zip(plan, plan[1:]) if prev.block != cur.block)
        if push_steps >= 12 and len(plan) >= 70 and len(walls) >= 24:
            return layout, plan

    raise RuntimeError("failed to generate a solvable real-like push layout")


def compress_plan(plan: list[PlannedStep]) -> list[MotionSegment]:
    segments: list[MotionSegment] = []
    if len(plan) < 2:
        return segments

    start = plan[0]
    prev = plan[0]
    prev_ball_delta: GridPoint | None = None
    prev_block_delta: GridPoint | None = None
    for current in plan[1:]:
        ball_delta = (current.ball[0] - prev.ball[0], current.ball[1] - prev.ball[1])
        block_delta = (current.block[0] - prev.block[0], current.block[1] - prev.block[1])
        if prev_ball_delta is None:
            prev_ball_delta = ball_delta
            prev_block_delta = block_delta
        elif ball_delta != prev_ball_delta or block_delta != prev_block_delta:
            segments.append(MotionSegment(start.ball, prev.ball, start.block, prev.block))
            start = prev
            prev_ball_delta = ball_delta
            prev_block_delta = block_delta
        prev = current
    segments.append(MotionSegment(start.ball, prev.ball, start.block, prev.block))
    return [segment for segment in segments if segment.ball_start != segment.ball_end]


def interpolate(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def segment_length(segment: MotionSegment) -> float:
    return math.hypot(segment.ball_end[0] - segment.ball_start[0], segment.ball_end[1] - segment.ball_start[1])


def resample_motion(segments: list[MotionSegment]) -> tuple[list[Point], list[Point]]:
    lengths = [segment_length(segment) for segment in segments]
    total_length = sum(lengths)
    intervals = [max(1, int(round(length / total_length * (NUM_FRAMES - 1)))) for length in lengths]
    while sum(intervals) > NUM_FRAMES - 1:
        index = max(range(len(intervals)), key=lambda i: intervals[i])
        if intervals[index] > 1:
            intervals[index] -= 1
        else:
            break
    while sum(intervals) < NUM_FRAMES - 1:
        index = max(range(len(intervals)), key=lambda i: lengths[i] / max(intervals[i], 1))
        intervals[index] += 1

    balls = [segments[0].ball_start]
    blocks = [segments[0].block_start]
    for segment, interval_count in zip(segments, intervals):
        for step in range(1, interval_count + 1):
            t = step / interval_count
            balls.append(interpolate(segment.ball_start, segment.ball_end, t))
            blocks.append(interpolate(segment.block_start, segment.block_end, t))
    if len(balls) != NUM_FRAMES:
        raise RuntimeError(f"expected {NUM_FRAMES} frames, got {len(balls)}")
    return balls, blocks


LAYOUT, PLAN = generate_layout()
SEGMENTS = compress_plan(PLAN)
BALL_POSITIONS, BLOCK_POSITIONS = resample_motion(SEGMENTS)


def polygon_points(box: Box) -> list[Point]:
    radians = math.radians(box.angle)
    points = []
    for x, y in [(-box.w / 2, -box.h / 2), (box.w / 2, -box.h / 2), (box.w / 2, box.h / 2), (-box.w / 2, box.h / 2)]:
        points.append((box.cx + x * math.cos(radians) - y * math.sin(radians), box.cy + x * math.sin(radians) + y * math.cos(radians)))
    return points


def draw_wood(draw: ImageDraw.ImageDraw, box: Box) -> None:
    points = polygon_points(box)
    draw.polygon(points, fill=(164, 125, 78))
    draw.line(points + [points[0]], fill=(120, 86, 46), width=1)
    if box.w >= box.h:
        y = (points[0][1] + points[2][1]) / 2
        draw.line((points[0][0] + 4, y, points[1][0] - 4, y), fill=(190, 150, 96), width=1)
    else:
        x = (points[0][0] + points[2][0]) / 2
        draw.line((x, points[0][1] + 4, x, points[2][1] - 4), fill=(190, 150, 96), width=1)


def draw_background() -> Image.Image:
    image = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (204, 207, 198))
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(round(value) for value in BOARD_RECT), fill=(238, 239, 232), outline=(215, 216, 209))
    for x in range(28, 202, 18):
        draw.line((x, BOARD_RECT[1] + 5, x, BOARD_RECT[3] - 5), fill=(231, 232, 224), width=1)
    for box in LAYOUT.walls:
        draw_wood(draw, box)
    return image


def render_frame(frame_index: int) -> np.ndarray:
    image = draw_background()
    draw = ImageDraw.Draw(image)
    block_x, block_y = BLOCK_POSITIONS[frame_index]
    block = Box(block_x, block_y, LAYOUT.block_w, LAYOUT.block_h)
    draw.rounded_rectangle(
        (round(block.cx - block.w / 2), round(block.cy - block.h / 2), round(block.cx + block.w / 2), round(block.cy + block.h / 2)),
        radius=1,
        fill=(8, 10, 9),
        outline=(0, 0, 0),
        width=1,
    )
    ball_x, ball_y = BALL_POSITIONS[frame_index]
    draw.ellipse((ball_x - BALL_RADIUS, ball_y - BALL_RADIUS, ball_x + BALL_RADIUS, ball_y + BALL_RADIUS), fill=(255, 0, 0))
    return np.asarray(image)


def assert_motion_valid() -> None:
    for index, (ball, block) in enumerate(zip(BALL_POSITIONS, BLOCK_POSITIONS)):
        if not static_ball_clear(ball, LAYOUT.walls):
            raise RuntimeError(f"red ball touches wall at frame {index}")
        if distance_point_to_box(ball, block_box(block, LAYOUT)) < BALL_RADIUS - 0.5:
            raise RuntimeError(f"red ball overlaps black block at frame {index}")
        if not block_clear(block, LAYOUT):
            raise RuntimeError(f"black block collision at frame {index}")
        if index > 0:
            prev = BALL_POSITIONS[index - 1]
            if abs(ball[0] - prev[0]) > 1e-6 and abs(ball[1] - prev[1]) > 1e-6:
                raise RuntimeError(f"diagonal red-ball motion at frame {index}")
            prev_block = BLOCK_POSITIONS[index - 1]
            if block != prev_block and abs(block[0] - prev_block[0]) > 1e-6 and abs(block[1] - prev_block[1]) > 1e-6:
                raise RuntimeError(f"diagonal black-block motion at frame {index}")

    for segment in SEGMENTS:
        for wall in LAYOUT.walls:
            if segment_to_box_clearance(segment.ball_start, segment.ball_end, wall) < BALL_RADIUS - 0.5:
                raise RuntimeError(f"planned red-ball corridor too narrow near {wall}")


def write_contact_sheet(frames: list[np.ndarray]) -> None:
    ids = [0, 10, 20, 32, 40, 45, 52, 65, 80]
    tiles = []
    for frame_index in ids:
        tile = Image.fromarray(frames[frame_index]).resize((224, 224))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 6), f"f{frame_index}", fill=(0, 180, 0))
        tiles.append(tile)
    canvas = Image.new("RGB", (3 * 224, 3 * 224), (255, 255, 255))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 3) * 224, (index // 3) * 224))
    canvas.save(OUT_CONTACT)


def main() -> None:
    assert_motion_valid()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [render_frame(index) for index in range(NUM_FRAMES)]
    with imageio.get_writer(OUT_VIDEO, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    write_contact_sheet(frames)
    push_steps = sum(1 for prev, cur in zip(PLAN, PLAN[1:]) if prev.block != cur.block)
    print(OUT_VIDEO)
    print(OUT_CONTACT)
    print(f"walls={len(LAYOUT.walls)} block_orientation={LAYOUT.orientation} plan_steps={len(PLAN)} push_steps={push_steps}")


if __name__ == "__main__":
    main()
