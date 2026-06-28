#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import multiprocessing as mp
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


MAZE_WIDTH_CM = 27
MAZE_HEIGHT_CM = 21
LOGICAL_W = 9
LOGICAL_H = 7
CELL_CM = 3
WALL_THICK_CM = 1
WALL_LONG_CM = 3
BALL_RADIUS_CM = 1
RENDER_BALL_RADIUS_CM = BALL_RADIUS_CM * 0.5
PASSAGE_CLEARANCE_CM = BALL_RADIUS_CM * 2
MIN_VISIBLE_GAP_CM = BALL_RADIUS_CM * 2
PUSH_BLOCK_SHORT_CM = 2
PUSH_BLOCK_LONG_CM = 4
FPS = 10
FRAMES_PER_CM = 5 / 3
FRAME_SIZE = 224
SCALE = FRAME_SIZE / MAZE_WIDTH_CM
Y_OFFSET_PX = (FRAME_SIZE - round(MAZE_HEIGHT_CM * SCALE)) / 2

Point = tuple[float, float]
GridPoint = tuple[int, int]
Dir = tuple[int, int]
DIRS: tuple[Dir, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
LONG_CORRIDOR_4DIR_ORDER: tuple[str, ...] = ("up", "down", "left", "right")
LONG_CORRIDOR_DIRECTION_VECTORS: dict[str, Dir] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class PushBlock:
    x: float
    y: float
    w: float
    h: float
    orientation: str


@dataclass
class LogicalCell:
    visited: bool
    walls: dict[str, bool]


@dataclass(frozen=True)
class MazeSample:
    walls: list[Rect]
    push_blocks: list[PushBlock]
    start: Point
    goal: Point
    solution_ball_path: list[Point]
    solution_block_paths: list[list[Point]]
    no_push_reachable: bool
    branch_score: int
    raw_path_steps: int = 0
    raw_push_steps: int = 0


@dataclass(frozen=True)
class RequiredBlockCandidate:
    block: PushBlock
    direction: str
    single_path_steps: int
    single_push_steps: int


def rect_overlap(a: Rect, b: Rect, margin: int = 0) -> bool:
    return not (a.x + a.w + margin <= b.x or b.x + b.w + margin <= a.x or a.y + a.h + margin <= b.y or b.y + b.h + margin <= a.y)


def inflate(rect: Rect, radius: int) -> Rect:
    return Rect(rect.x - radius, rect.y - radius, rect.w + radius * 2, rect.h + radius * 2)


def point_inside_rect(point: Point, rect: Rect) -> bool:
    x, y = point
    return rect.x < x < rect.x + rect.w and rect.y < y < rect.y + rect.h


def inside_maze_rect(rect: Rect) -> bool:
    return rect.x >= 0 and rect.y >= 0 and rect.x + rect.w <= MAZE_WIDTH_CM and rect.y + rect.h <= MAZE_HEIGHT_CM


def inside_ball(point: Point, clearance: int = BALL_RADIUS_CM) -> bool:
    x, y = point
    return clearance <= x <= MAZE_WIDTH_CM - clearance and clearance <= y <= MAZE_HEIGHT_CM - clearance


def ball_area_free(point: Point, walls: list[Rect], blocks: list[PushBlock], wall_clearance: int = BALL_RADIUS_CM, block_clearance: int = BALL_RADIUS_CM) -> bool:
    if not inside_ball(point, wall_clearance):
        return False
    for wall in walls:
        if point_inside_rect(point, inflate(wall, wall_clearance)):
            return False
    for block in blocks:
        if point_inside_rect(point, inflate(Rect(block.x, block.y, block.w, block.h), block_clearance)):
            return False
    return True


def rect_free(rect: Rect, walls: list[Rect], blocks: list[PushBlock], ignore_block_index: int | None = None) -> bool:
    if not inside_maze_rect(rect):
        return False
    for wall in walls:
        if rect_overlap(rect, wall):
            return False
    for index, block in enumerate(blocks):
        if index == ignore_block_index:
            continue
        if rect_overlap(rect, Rect(block.x, block.y, block.w, block.h)):
            return False
    return True


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def creates_micro_gap(candidate: Rect, obstacles: list[Rect], min_gap: float = MIN_VISIBLE_GAP_CM) -> bool:
    for obstacle in obstacles:
        if candidate.x + candidate.w <= obstacle.x:
            gap = obstacle.x - (candidate.x + candidate.w)
            if 0 < gap < min_gap and overlap_1d(candidate.y, candidate.y + candidate.h, obstacle.y, obstacle.y + obstacle.h) > 0:
                return True
        if obstacle.x + obstacle.w <= candidate.x:
            gap = candidate.x - (obstacle.x + obstacle.w)
            if 0 < gap < min_gap and overlap_1d(candidate.y, candidate.y + candidate.h, obstacle.y, obstacle.y + obstacle.h) > 0:
                return True
        if candidate.y + candidate.h <= obstacle.y:
            gap = obstacle.y - (candidate.y + candidate.h)
            if 0 < gap < min_gap and overlap_1d(candidate.x, candidate.x + candidate.w, obstacle.x, obstacle.x + obstacle.w) > 0:
                return True
        if obstacle.y + obstacle.h <= candidate.y:
            gap = candidate.y - (obstacle.y + obstacle.h)
            if 0 < gap < min_gap and overlap_1d(candidate.x, candidate.x + candidate.w, obstacle.x, obstacle.x + obstacle.w) > 0:
                return True
    return False


def make_logical_grid() -> list[list[LogicalCell]]:
    return [
        [LogicalCell(False, {"up": True, "down": True, "left": True, "right": True}) for _ in range(LOGICAL_W)]
        for _ in range(LOGICAL_H)
    ]


def remove_wall_between(grid: list[list[LogicalCell]], a: Point, b: Point) -> None:
    ax, ay = a
    bx, by = b
    if bx == ax + 1:
        grid[ay][ax].walls["right"] = False
        grid[by][bx].walls["left"] = False
    elif bx == ax - 1:
        grid[ay][ax].walls["left"] = False
        grid[by][bx].walls["right"] = False
    elif by == ay + 1:
        grid[ay][ax].walls["down"] = False
        grid[by][bx].walls["up"] = False
    elif by == ay - 1:
        grid[ay][ax].walls["up"] = False
        grid[by][bx].walls["down"] = False


def generate_maze_skeleton_dfs(rng: random.Random) -> list[list[LogicalCell]]:
    grid = make_logical_grid()
    stack = [(0, 0)]
    grid[0][0].visited = True
    while stack:
        x, y = stack[-1]
        candidates: list[GridPoint] = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < LOGICAL_W and 0 <= ny < LOGICAL_H and not grid[ny][nx].visited:
                candidates.append((nx, ny))
        if not candidates:
            stack.pop()
            continue
        nxt = rng.choice(candidates)
        remove_wall_between(grid, (x, y), nxt)
        grid[nxt[1]][nxt[0]].visited = True
        stack.append(nxt)
    return grid


def add_misleading_branches(grid: list[list[LogicalCell]], rng: random.Random, branch_count: int) -> None:
    for _ in range(branch_count):
        x = rng.randrange(LOGICAL_W)
        y = rng.randrange(LOGICAL_H)
        length = rng.randint(1, 3)
        dx, dy = rng.choice(DIRS)
        for _ in range(length):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < LOGICAL_W and 0 <= ny < LOGICAL_H):
                break
            remove_wall_between(grid, (x, y), (nx, ny))
            x, y = nx, ny
            if rng.random() < 0.35:
                dx, dy = rng.choice([(dy, dx), (-dy, -dx)] if dx == 0 else [(dy, dx), (-dy, -dx)])


def skeleton_to_walls(grid: list[list[LogicalCell]]) -> list[Rect]:
    walls: list[Rect] = []
    for y in range(LOGICAL_H):
        for x in range(LOGICAL_W):
            cell = grid[y][x]
            rx = x * CELL_CM
            ry = y * CELL_CM
            if cell.walls["up"]:
                walls.append(Rect(rx, ry, WALL_LONG_CM, WALL_THICK_CM))
            if cell.walls["left"]:
                walls.append(Rect(rx, ry, WALL_THICK_CM, WALL_LONG_CM))
            if x == LOGICAL_W - 1 and cell.walls["right"]:
                walls.append(Rect(rx + CELL_CM - WALL_THICK_CM, ry, WALL_THICK_CM, WALL_LONG_CM))
            if y == LOGICAL_H - 1 and cell.walls["down"]:
                walls.append(Rect(rx, ry + CELL_CM - WALL_THICK_CM, WALL_LONG_CM, WALL_THICK_CM))
    return walls


def add_irregular_decorative_walls(walls: list[Rect], rng: random.Random, protected: list[Rect], attempts: int = 120) -> list[Rect]:
    result = list(walls)
    for _ in range(attempts):
        horizontal = rng.random() < 0.5
        if horizontal:
            wall = Rect(rng.randint(1, MAZE_WIDTH_CM - WALL_LONG_CM - 1), rng.randint(1, MAZE_HEIGHT_CM - WALL_THICK_CM - 1), WALL_LONG_CM, WALL_THICK_CM)
        else:
            wall = Rect(rng.randint(1, MAZE_WIDTH_CM - WALL_THICK_CM - 1), rng.randint(1, MAZE_HEIGHT_CM - WALL_LONG_CM - 1), WALL_THICK_CM, WALL_LONG_CM)
        if any(rect_overlap(wall, existing) for existing in result):
            continue
        if creates_micro_gap(wall, result):
            continue
        if any(rect_overlap(wall, area, margin=1) for area in protected):
            continue
        result.append(wall)
        if len(result) >= len(walls) + 10:
            break
    return result


def prune_to_keep_solvable(walls: list[Rect], blocks: list[PushBlock], start: Point, goal: Point, rng: random.Random) -> list[Rect]:
    shuffled = list(walls)
    rng.shuffle(shuffled)
    kept: list[Rect] = []
    for wall in shuffled:
        trial = kept + [wall]
        if bfs_path(start, goal, trial, blocks, BALL_RADIUS_CM) is not None:
            kept.append(wall)
    return kept


def carve_clear_areas(walls: list[Rect], clear_areas: list[Rect]) -> list[Rect]:
    return [wall for wall in walls if not any(rect_overlap(wall, area, margin=1) for area in clear_areas)]


def seal_micro_gaps(walls: list[Rect], blocks: list[PushBlock], protected: list[Rect], min_gap: float = MIN_VISIBLE_GAP_CM) -> list[Rect]:
    obstacles = list(walls) + [Rect(block.x, block.y, block.w, block.h) for block in blocks]
    new_walls = list(walls)

    def add_wall_if_valid(candidate: Rect) -> None:
        if candidate.w <= 0 or candidate.h <= 0:
            return
        if not inside_maze_rect(candidate):
            return
        if any(rect_overlap(candidate, area) for area in protected):
            return
        if any(rect_overlap(candidate, existing) for existing in new_walls):
            return
        new_walls.append(candidate)

    for index, a in enumerate(obstacles):
        for b in obstacles[index + 1 :]:
            if a.x + a.w <= b.x:
                gap = b.x - (a.x + a.w)
                y0 = max(a.y, b.y)
                y1 = min(a.y + a.h, b.y + b.h)
                if 0 < gap < min_gap and y1 > y0:
                    add_wall_if_valid(Rect(a.x + a.w, y0, gap, y1 - y0))
            elif b.x + b.w <= a.x:
                gap = a.x - (b.x + b.w)
                y0 = max(a.y, b.y)
                y1 = min(a.y + a.h, b.y + b.h)
                if 0 < gap < min_gap and y1 > y0:
                    add_wall_if_valid(Rect(b.x + b.w, y0, gap, y1 - y0))

            if a.y + a.h <= b.y:
                gap = b.y - (a.y + a.h)
                x0 = max(a.x, b.x)
                x1 = min(a.x + a.w, b.x + b.w)
                if 0 < gap < min_gap and x1 > x0:
                    add_wall_if_valid(Rect(x0, a.y + a.h, x1 - x0, gap))
            elif b.y + b.h <= a.y:
                gap = a.y - (b.y + b.h)
                x0 = max(a.x, b.x)
                x1 = min(a.x + a.w, b.x + b.w)
                if 0 < gap < min_gap and x1 > x0:
                    add_wall_if_valid(Rect(x0, b.y + b.h, x1 - x0, gap))
    return new_walls


def bfs_path(start: GridPoint, goal: GridPoint, walls: list[Rect], blocks: list[PushBlock], block_clearance: int = BALL_RADIUS_CM) -> list[GridPoint] | None:
    queue: list[tuple[int, int, GridPoint]] = []
    heapq.heappush(queue, (0, 0, start))
    parent: dict[GridPoint, GridPoint | None] = {start: None}
    cost: dict[GridPoint, int] = {start: 0}
    counter = 0
    while queue:
        _, _, current = heapq.heappop(queue)
        if current == goal:
            break
        for dx, dy in DIRS:
            nxt = (current[0] + dx, current[1] + dy)
            if not ball_area_free(nxt, walls, blocks, BALL_RADIUS_CM, block_clearance):
                continue
            new_cost = cost[current] + 1
            if nxt in cost and new_cost >= cost[nxt]:
                continue
            cost[nxt] = new_cost
            parent[nxt] = current
            counter += 1
            heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
            heapq.heappush(queue, (new_cost + heuristic, counter, nxt))
    if goal not in parent:
        return None
    path: list[GridPoint] = []
    cur: GridPoint | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def can_push_block(block: PushBlock, direction: Dir, walls: list[Rect], blocks: list[PushBlock], index: int) -> bool:
    dx, dy = direction
    moved = Rect(block.x + dx, block.y + dy, block.w, block.h)
    return rect_free(moved, walls, blocks, index)


def can_touch_and_push(ball: GridPoint, block: PushBlock, direction: Dir) -> bool:
    dx, dy = direction
    x, y = ball
    if (dx, dy) == (1, 0):
        return x <= block.x and block.y - BALL_RADIUS_CM <= y <= block.y + block.h + BALL_RADIUS_CM
    if (dx, dy) == (-1, 0):
        return x >= block.x + block.w and block.y - BALL_RADIUS_CM <= y <= block.y + block.h + BALL_RADIUS_CM
    if (dx, dy) == (0, 1):
        return y <= block.y and block.x - BALL_RADIUS_CM <= x <= block.x + block.w + BALL_RADIUS_CM
    if (dx, dy) == (0, -1):
        return y >= block.y + block.h and block.x - BALL_RADIUS_CM <= x <= block.x + block.w + BALL_RADIUS_CM
    return False


def push_direction_allowed(block: PushBlock, direction: Dir, push_side: str) -> bool:
    if push_side == "any":
        return True
    side = classify_push_side(block, (0, 0), direction)
    if push_side == "long_only":
        return side == "long"
    if push_side == "short_only":
        return side == "short"
    raise ValueError(f"unknown push_side: {push_side}")


def solve_with_pushing(
    start: GridPoint,
    goal: GridPoint,
    walls: list[Rect],
    blocks: list[PushBlock],
    max_expansions: int = 50000,
    push_side: str = "any",
) -> tuple[list[GridPoint], list[list[GridPoint]]] | None:
    block_tuple = tuple((b.x, b.y, b.w, b.h, b.orientation) for b in blocks)
    start_state = (start, block_tuple)
    queue: list[tuple[int, int, object]] = []
    heapq.heappush(queue, (0, 0, start_state))
    parent: dict[object, object | None] = {start_state: None}
    cost: dict[object, int] = {start_state: 0}
    counter = 0
    expansions = 0
    goal_state: object | None = None
    while queue and expansions < max_expansions:
        _, _, state = heapq.heappop(queue)
        expansions += 1
        ball, block_state = state
        cur_blocks = [PushBlock(x, y, w, h, orientation) for x, y, w, h, orientation in block_state]
        if ball == goal:
            goal_state = state
            break
        for direction in DIRS:
            dx, dy = direction
            nxt = (ball[0] + dx, ball[1] + dy)
            if ball_area_free(nxt, walls, cur_blocks):
                next_state = (nxt, block_state)
            else:
                pushed_index = None
                for index, block in enumerate(cur_blocks):
                    if point_inside_rect(nxt, inflate(Rect(block.x, block.y, block.w, block.h), BALL_RADIUS_CM)) and can_touch_and_push(ball, block, direction):
                        pushed_index = index
                        break
                if pushed_index is None or not can_push_block(cur_blocks[pushed_index], direction, walls, cur_blocks, pushed_index):
                    continue
                if not push_direction_allowed(cur_blocks[pushed_index], direction, push_side):
                    continue
                new_blocks = list(cur_blocks)
                old = cur_blocks[pushed_index]
                new_blocks[pushed_index] = PushBlock(old.x + dx, old.y + dy, old.w, old.h, old.orientation)
                if not ball_area_free(nxt, walls, new_blocks):
                    continue
                next_block_state = tuple((b.x, b.y, b.w, b.h, b.orientation) for b in new_blocks)
                next_state = (nxt, next_block_state)
            new_cost = cost[state] + 1
            if next_state in cost and new_cost >= cost[next_state]:
                continue
            cost[next_state] = new_cost
            parent[next_state] = state
            counter += 1
            heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
            heapq.heappush(queue, (new_cost + heuristic, counter, next_state))

    if goal_state is None:
        return None
    states = []
    cur = goal_state
    while cur is not None:
        states.append(cur)
        cur = parent[cur]
    states.reverse()
    ball_path = [state[0] for state in states]
    block_paths = [[(state[1][i][0], state[1][i][1]) for state in states] for i in range(len(blocks))]
    return ball_path, block_paths


def add_boundary_walls(walls: list[Rect]) -> list[Rect]:
    result = list(walls)
    for wall in [Rect(0, 0, MAZE_WIDTH_CM, 1), Rect(0, MAZE_HEIGHT_CM - 1, MAZE_WIDTH_CM, 1), Rect(0, 0, 1, MAZE_HEIGHT_CM), Rect(MAZE_WIDTH_CM - 1, 0, 1, MAZE_HEIGHT_CM)]:
        if wall not in result:
            result.append(wall)
    return result


def candidate_blocks_for_path(path: list[GridPoint], rng: random.Random) -> list[PushBlock]:
    candidates: list[PushBlock] = []
    for x, y in path[len(path) // 5 : -len(path) // 5]:
        for block in [
            PushBlock(x - 1, y - 2, PUSH_BLOCK_SHORT_CM, PUSH_BLOCK_LONG_CM, "vertical"),
            PushBlock(x - 2, y - 1, PUSH_BLOCK_LONG_CM, PUSH_BLOCK_SHORT_CM, "horizontal"),
        ]:
            if inside_maze_rect(Rect(block.x, block.y, block.w, block.h)):
                candidates.append(block)
    rng.shuffle(candidates)
    return candidates


def candidate_blocks_for_path_direction(path: list[GridPoint], push_direction: str, rng: random.Random) -> list[PushBlock]:
    direction = LONG_CORRIDOR_DIRECTION_VECTORS[push_direction]
    candidates: list[PushBlock] = []
    margin = max(2, len(path) // 5)
    for start, end in zip(path[margin:-margin], path[margin + 1 : -margin + 1]):
        if (end[0] - start[0], end[1] - start[1]) != direction:
            continue
        x, y = start
        if push_direction == "down":
            block = PushBlock(x - 2, y + 1, PUSH_BLOCK_LONG_CM, PUSH_BLOCK_SHORT_CM, "horizontal")
        elif push_direction == "up":
            block = PushBlock(x - 2, y - 3, PUSH_BLOCK_LONG_CM, PUSH_BLOCK_SHORT_CM, "horizontal")
        elif push_direction == "right":
            block = PushBlock(x + 1, y - 2, PUSH_BLOCK_SHORT_CM, PUSH_BLOCK_LONG_CM, "vertical")
        elif push_direction == "left":
            block = PushBlock(x - 3, y - 2, PUSH_BLOCK_SHORT_CM, PUSH_BLOCK_LONG_CM, "vertical")
        else:
            raise ValueError(f"unknown push direction: {push_direction}")
        if inside_maze_rect(Rect(block.x, block.y, block.w, block.h)):
            candidates.append(block)
    rng.shuffle(candidates)
    return candidates[:12]


def add_extra_blocks(required: PushBlock, walls: list[Rect], start: GridPoint, goal: GridPoint, rng: random.Random, count: int) -> list[PushBlock]:
    blocks = [required]
    attempts = 0
    while len(blocks) < count and attempts < 100:
        attempts += 1
        horizontal = rng.random() < 0.5
        if horizontal:
            candidate = PushBlock(rng.randint(3, MAZE_WIDTH_CM - PUSH_BLOCK_LONG_CM - 3), rng.randint(3, MAZE_HEIGHT_CM - PUSH_BLOCK_SHORT_CM - 3), PUSH_BLOCK_LONG_CM, PUSH_BLOCK_SHORT_CM, "horizontal")
        else:
            candidate = PushBlock(rng.randint(3, MAZE_WIDTH_CM - PUSH_BLOCK_SHORT_CM - 3), rng.randint(3, MAZE_HEIGHT_CM - PUSH_BLOCK_LONG_CM - 3), PUSH_BLOCK_SHORT_CM, PUSH_BLOCK_LONG_CM, "vertical")
        if not rect_free(Rect(candidate.x, candidate.y, candidate.w, candidate.h), walls, blocks):
            continue
        if not ball_area_free(start, walls, blocks + [candidate]) or not ball_area_free(goal, walls, blocks + [candidate]):
            continue
        blocks.append(candidate)
    return blocks


def add_path_biased_extra_blocks(
    required: PushBlock,
    walls: list[Rect],
    start: GridPoint,
    goal: GridPoint,
    rng: random.Random,
    count: int,
) -> list[PushBlock]:
    blocks = [required]
    while len(blocks) < count:
        current_path = bfs_path(start, goal, walls, blocks, block_clearance=0)
        if current_path is None:
            break
        added = False
        for candidate in candidate_blocks_for_path(current_path, rng):
            if not rect_free(Rect(candidate.x, candidate.y, candidate.w, candidate.h), walls, blocks):
                continue
            if not ball_area_free(start, walls, blocks + [candidate]) or not ball_area_free(goal, walls, blocks + [candidate]):
                continue
            blocks.append(candidate)
            added = True
            break
        if not added:
            break
    if len(blocks) < count:
        blocks = add_extra_blocks(required, walls, start, goal, rng, count)
    return blocks


def place_required_blocks_by_cutting_path(walls: list[Rect], start: GridPoint, goal: GridPoint, rng: random.Random, count: int, push_side: str) -> list[PushBlock] | None:
    base_path = bfs_path(start, goal, walls, [])
    if base_path is None:
        return None
    for required in candidate_blocks_for_path(base_path, rng):
        if not rect_free(Rect(required.x, required.y, required.w, required.h), walls, []):
            continue
        blocks = [required]
        if not ball_area_free(start, walls, blocks) or not ball_area_free(goal, walls, blocks):
            continue
        if bfs_path(start, goal, walls, blocks) is not None:
            continue
        solved = solve_with_pushing(start, goal, walls, blocks, max_expansions=150000, push_side=push_side)
        if solved is None:
            continue
        return add_path_biased_extra_blocks(required, walls, start, goal, rng, count)
    return None


def place_required_blocks_by_direction(
    walls: list[Rect],
    start: GridPoint,
    goal: GridPoint,
    rng: random.Random,
    count: int,
    push_direction: str,
) -> list[PushBlock] | None:
    base_path = bfs_path(start, goal, walls, [])
    if base_path is None:
        return None
    candidates = candidate_blocks_for_path(base_path, rng)
    if push_direction in ("up", "down"):
        candidates = [block for block in candidates if block.orientation == "horizontal"]
    else:
        candidates = [block for block in candidates if block.orientation == "vertical"]
    for required in candidates[:36]:
        if not rect_free(Rect(required.x, required.y, required.w, required.h), walls, []):
            continue
        blocks = [required]
        if not ball_area_free(start, walls, blocks) or not ball_area_free(goal, walls, blocks):
            continue
        if bfs_path(start, goal, walls, blocks) is not None:
            continue
        solved = solve_with_pushing(start, goal, walls, blocks, max_expansions=40000, push_side="long_only")
        if solved is None:
            continue
        directions = pushed_directions(solved[1])
        expected_direction = LONG_CORRIDOR_DIRECTION_VECTORS[push_direction]
        if not directions or any(direction != expected_direction for direction in directions):
            continue
        return add_extra_blocks(required, walls, start, goal, rng, count)
    return None


def enumerate_required_block_candidates(
    walls: list[Rect],
    start: GridPoint,
    goal: GridPoint,
    rng: random.Random,
    push_side: str = "any",
    max_candidates: int = 80,
) -> list[RequiredBlockCandidate]:
    base_path = bfs_path(start, goal, walls, [])
    if base_path is None:
        return []
    result: list[RequiredBlockCandidate] = []
    seen: set[tuple[float, float, float, float]] = set()
    for block in candidate_blocks_for_path(base_path, rng):
        key = (block.x, block.y, block.w, block.h)
        if key in seen:
            continue
        seen.add(key)
        if not rect_free(Rect(block.x, block.y, block.w, block.h), walls, []):
            continue
        blocks = [block]
        if not ball_area_free(start, walls, blocks) or not ball_area_free(goal, walls, blocks):
            continue
        if bfs_path(start, goal, walls, blocks) is not None:
            continue
        solved = solve_with_pushing(start, goal, walls, blocks, max_expansions=80000, push_side=push_side)
        if solved is None:
            continue
        ball_path, block_paths = solved
        directions = solution_moved_directions(block_paths)
        if len(directions) != 1:
            continue
        push_moves = sum(1 for path in block_paths for prev, cur in zip(path, path[1:]) if prev != cur)
        result.append(RequiredBlockCandidate(block, directions[0], len(ball_path), push_moves))
        if len(result) >= max_candidates:
            break
    rng.shuffle(result)
    return result


def blocks_pairwise_compatible(blocks: list[PushBlock], walls: list[Rect], start: GridPoint, goal: GridPoint) -> bool:
    placed: list[PushBlock] = []
    for block in blocks:
        rect = Rect(block.x, block.y, block.w, block.h)
        if not rect_free(rect, walls, placed):
            return False
        placed.append(block)
    return ball_area_free(start, walls, placed) and ball_area_free(goal, walls, placed)


def choose_required_blocks(
    candidates: list[RequiredBlockCandidate],
    walls: list[Rect],
    start: GridPoint,
    goal: GridPoint,
    rng: random.Random,
    block_count: int,
    target_direction: str | None = None,
) -> list[PushBlock] | None:
    ordered = list(candidates)
    rng.shuffle(ordered)
    if target_direction is not None:
        ordered.sort(key=lambda candidate: candidate.direction != target_direction)

    chosen: list[PushBlock] = []
    used_keys: set[tuple[float, float, float, float]] = set()
    for candidate in ordered:
        if target_direction is not None and not chosen and candidate.direction != target_direction:
            continue
        key = (candidate.block.x, candidate.block.y, candidate.block.w, candidate.block.h)
        if key in used_keys:
            continue
        trial = chosen + [candidate.block]
        if not blocks_pairwise_compatible(trial, walls, start, goal):
            continue
        chosen.append(candidate.block)
        used_keys.add(key)
        if len(chosen) == block_count:
            return chosen
    return None


def reachable_points(walls: list[Rect], blocks: list[PushBlock], start: GridPoint) -> set[GridPoint]:
    reachable = set()
    stack = [start]
    while stack:
        p = stack.pop()
        if p in reachable:
            continue
        reachable.add(p)
        for dx, dy in DIRS:
            nxt = (p[0] + dx, p[1] + dy)
            if ball_area_free(nxt, walls, blocks):
                stack.append(nxt)
    return reachable


def branch_score_v2(walls: list[Rect], blocks: list[PushBlock], start: GridPoint, goal: GridPoint, solution_path: list[GridPoint]) -> int:
    reachable = reachable_points(walls, blocks, start)
    solution_set = set(solution_path)
    off_path = reachable - solution_set
    visited: set[GridPoint] = set()
    score = 0
    for point in list(off_path):
        if point in visited:
            continue
        stack = [point]
        component: set[GridPoint] = set()
        entrances = 0
        while stack:
            current = stack.pop()
            if current in visited or current not in off_path:
                continue
            visited.add(current)
            component.add(current)
            for dx, dy in DIRS:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in solution_set:
                    entrances += 1
                elif nxt in off_path and nxt not in visited:
                    stack.append(nxt)
        if not component:
            continue
        min_dist_to_goal = min(abs(p[0] - goal[0]) + abs(p[1] - goal[1]) for p in component)
        if entrances <= 2 and len(component) >= 6:
            score += 2
        if min_dist_to_goal <= 10 and len(component) >= 4:
            score += 3
    return score


def branch_score(walls: list[Rect], blocks: list[PushBlock], start: Point, goal: Point) -> int:
    reachable = reachable_points(walls, blocks, start)  # kept for backward-compatible diagnostics
    score = 0
    for p in reachable:
        degree = sum(1 for dx, dy in DIRS if (p[0] + dx, p[1] + dy) in reachable)
        if degree >= 3:
            score += 1
    return score


def classify_push_side(block: PushBlock, start: Point, end: Point) -> str | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    side_length = block.h if abs(dx) > abs(dy) else block.w
    return "long" if side_length == max(block.w, block.h) else "short"


def push_side_counts(blocks: list[PushBlock], block_paths: list[list[GridPoint]]) -> tuple[int, int]:
    long_count = 0
    short_count = 0
    for block, path in zip(blocks, block_paths):
        for start, end in zip(path, path[1:]):
            side = classify_push_side(block, start, end)
            if side == "long":
                long_count += 1
            elif side == "short":
                short_count += 1
    return long_count, short_count


def push_side_allowed(blocks: list[PushBlock], block_paths: list[list[GridPoint]], push_side: str) -> bool:
    long_count, short_count = push_side_counts(blocks, block_paths)
    if push_side == "any":
        return long_count + short_count > 0
    if push_side == "long_only":
        return long_count > 0 and short_count == 0
    if push_side == "short_only":
        return short_count > 0 and long_count == 0
    raise ValueError(f"unknown push_side: {push_side}")


def pushed_directions(block_paths: list[list[GridPoint]]) -> list[Dir]:
    directions: list[Dir] = []
    for path in block_paths:
        for start, end in zip(path, path[1:]):
            dx = int(round(end[0] - start[0]))
            dy = int(round(end[1] - start[1]))
            if dx != 0 or dy != 0:
                directions.append((dx, dy))
    return directions


def moved_block_indices(block_paths: list[list[GridPoint]]) -> list[int]:
    return [index for index, path in enumerate(block_paths) if any(start != end for start, end in zip(path, path[1:]))]


def block_total_direction(block_path: list[GridPoint]) -> str | None:
    dx = sum(end[0] - start[0] for start, end in zip(block_path, block_path[1:]))
    dy = sum(end[1] - start[1] for start, end in zip(block_path, block_path[1:]))
    if dx > 0 and dy == 0:
        return "right"
    if dx < 0 and dy == 0:
        return "left"
    if dy > 0 and dx == 0:
        return "down"
    if dy < 0 and dx == 0:
        return "up"
    return None


def solution_moved_directions(block_paths: list[list[GridPoint]]) -> list[str]:
    directions: list[str] = []
    for path in block_paths:
        direction = block_total_direction(path)
        if direction is not None:
            directions.append(direction)
    return directions


def shortest_goal_lengths_by_push_validity(
    start: GridPoint,
    goal: GridPoint,
    walls: list[Rect],
    blocks: list[PushBlock],
    expected_direction: Dir,
    max_expansions: int = 200000,
) -> tuple[int | None, int | None]:
    block_tuple = tuple((b.x, b.y, b.w, b.h, b.orientation) for b in blocks)
    start_state = (start, block_tuple, False)
    queue = deque([start_state])
    distance: dict[object, int] = {start_state: 0}
    best_valid: int | None = None
    best_invalid: int | None = None
    expansions = 0
    while queue and expansions < max_expansions:
        state = queue.popleft()
        expansions += 1
        ball, block_state, invalid_push_used = state
        state_distance = distance[state]
        if ball == goal:
            if invalid_push_used:
                best_invalid = state_distance if best_invalid is None else min(best_invalid, state_distance)
            else:
                best_valid = state_distance if best_valid is None else min(best_valid, state_distance)
            if best_valid is not None and best_invalid is not None:
                return best_valid + 1, best_invalid + 1
            continue
        cur_blocks = [PushBlock(x, y, w, h, orientation) for x, y, w, h, orientation in block_state]
        for direction in DIRS:
            dx, dy = direction
            nxt = (ball[0] + dx, ball[1] + dy)
            next_invalid = invalid_push_used
            if ball_area_free(nxt, walls, cur_blocks):
                next_block_state = block_state
            else:
                pushed_index = None
                for index, block in enumerate(cur_blocks):
                    if point_inside_rect(nxt, inflate(Rect(block.x, block.y, block.w, block.h), BALL_RADIUS_CM)) and can_touch_and_push(ball, block, direction):
                        pushed_index = index
                        break
                if pushed_index is None or not can_push_block(cur_blocks[pushed_index], direction, walls, cur_blocks, pushed_index):
                    continue
                side = classify_push_side(cur_blocks[pushed_index], (0, 0), direction)
                next_invalid = next_invalid or side != "long" or direction != expected_direction
                new_blocks = list(cur_blocks)
                old = cur_blocks[pushed_index]
                new_blocks[pushed_index] = PushBlock(old.x + dx, old.y + dy, old.w, old.h, old.orientation)
                if not ball_area_free(nxt, walls, new_blocks):
                    continue
                next_block_state = tuple((b.x, b.y, b.w, b.h, b.orientation) for b in new_blocks)
            next_state = (nxt, next_block_state, next_invalid)
            if next_state in distance:
                continue
            distance[next_state] = state_distance + 1
            queue.append(next_state)
    return best_valid + 1 if best_valid is not None else None, best_invalid + 1 if best_invalid is not None else None


def generate_one(
    seed: int,
    block_count: int | None = None,
    max_attempts: int = 200,
    push_side: str = "any",
    push_direction: str | None = None,
    min_moved_blocks: int = 1,
) -> MazeSample:
    rng = random.Random(seed)
    start = (3, 3)
    goal = (26, 20)
    for _ in range(max_attempts):
        count = block_count if block_count is not None else (1 if push_side != "any" else rng.randint(1, 3))
        grid = generate_maze_skeleton_dfs(rng)
        add_misleading_branches(grid, rng, branch_count=rng.randint(5, 10))
        walls = skeleton_to_walls(grid)
        walls = carve_clear_areas(walls, [Rect(1, 1, 6, 6), Rect(23, 17, 4, 4)])
        walls = add_boundary_walls(walls)
        walls = prune_to_keep_solvable(walls, [], start, goal, rng)
        protected = [
            Rect(0, 0, 5, 5),
            Rect(23, 17, 4, 4),
        ]
        walls = add_irregular_decorative_walls(walls, rng, protected)
        walls = prune_to_keep_solvable(walls, [], start, goal, rng)
        if push_direction is None:
            blocks = place_required_blocks_by_cutting_path(walls, start, goal, rng, count, push_side)
        else:
            blocks = place_required_blocks_by_direction(walls, start, goal, rng, count, push_direction)
        if blocks is None:
            continue
        walls = seal_micro_gaps(
            walls,
            [],
            [
                Rect(start[0] - 2, start[1] - 2, 4, 4),
                Rect(goal[0] - 2, goal[1] - 2, 4, 4),
            ],
        )
        no_push = bfs_path(start, goal, walls, blocks) is not None
        if no_push:
            continue
        solved = solve_with_pushing(start, goal, walls, blocks, max_expansions=200000, push_side=push_side)
        if solved is None:
            continue
        ball_path, block_paths = solved
        push_moves = sum(1 for path in block_paths for prev, cur in zip(path, path[1:]) if prev != cur)
        if len(moved_block_indices(block_paths)) < min_moved_blocks:
            continue
        if not push_side_allowed(blocks, block_paths, push_side):
            continue
        if push_direction is not None:
            directions = pushed_directions(block_paths)
            expected_direction = LONG_CORRIDOR_DIRECTION_VECTORS[push_direction]
            if not directions or any(direction != expected_direction for direction in directions):
                continue
            validate_long_corridor_solution(walls, blocks, start, goal, solved, push_direction)
        score = branch_score_v2(walls, blocks, start, goal, ball_path)
        if push_moves < 1 or score < 3:
            continue
        return MazeSample(walls, blocks, start, goal, ball_path, block_paths, no_push, score, len(ball_path), push_moves)
    raise RuntimeError(f"failed to generate irregular push maze after {max_attempts} attempts")


def generate_required_blocks_one(
    seed: int,
    block_count: int = 3,
    min_moved_blocks: int = 1,
    max_attempts: int = 200,
    target_direction: str | None = None,
    push_side: str = "any",
) -> MazeSample:
    rng = random.Random(seed)
    start = (3, 3)
    goal = (26, 20)
    for _ in range(max_attempts):
        grid = generate_maze_skeleton_dfs(rng)
        add_misleading_branches(grid, rng, branch_count=rng.randint(5, 10))
        walls = skeleton_to_walls(grid)
        walls = carve_clear_areas(walls, [Rect(1, 1, 6, 6), Rect(23, 17, 4, 4)])
        walls = add_boundary_walls(walls)
        walls = prune_to_keep_solvable(walls, [], start, goal, rng)
        protected = [Rect(0, 0, 5, 5), Rect(23, 17, 4, 4)]
        walls = add_irregular_decorative_walls(walls, rng, protected)
        walls = prune_to_keep_solvable(walls, [], start, goal, rng)

        candidates = enumerate_required_block_candidates(walls, start, goal, rng, push_side=push_side)
        if not candidates:
            continue
        blocks = choose_required_blocks(candidates, walls, start, goal, rng, block_count, target_direction=target_direction)
        if blocks is None:
            continue
        walls = seal_micro_gaps(
            walls,
            [],
            [
                Rect(start[0] - 2, start[1] - 2, 4, 4),
                Rect(goal[0] - 2, goal[1] - 2, 4, 4),
            ],
        )
        if bfs_path(start, goal, walls, blocks) is not None:
            continue
        solved = solve_with_pushing(start, goal, walls, blocks, max_expansions=250000, push_side=push_side)
        if solved is None:
            continue
        ball_path, block_paths = solved
        moved = moved_block_indices(block_paths)
        if len(moved) < min_moved_blocks:
            continue
        if not push_side_allowed(blocks, block_paths, push_side):
            continue
        directions = solution_moved_directions(block_paths)
        if target_direction is not None and target_direction not in directions:
            continue
        push_moves = sum(1 for path in block_paths for prev, cur in zip(path, path[1:]) if prev != cur)
        score = branch_score_v2(walls, blocks, start, goal, ball_path)
        if push_moves < 1 or score < 3:
            continue
        return MazeSample(walls, blocks, start, goal, ball_path, block_paths, False, score, len(ball_path), push_moves)
    raise RuntimeError(f"failed to generate required-block maze after {max_attempts} attempts")


def solve_scripted_corridor(
    start: GridPoint,
    contact_start: GridPoint,
    push_dir: Dir,
    push_distance: int,
    goal: GridPoint,
    block: PushBlock,
    waypoints: list[GridPoint],
) -> tuple[list[GridPoint], list[list[GridPoint]]]:
    ball_path: list[GridPoint] = []
    block_path: list[GridPoint] = []

    def append_point(point: GridPoint, block_pos: GridPoint) -> None:
        if ball_path and ball_path[-1] == point and block_path[-1] == block_pos:
            return
        ball_path.append(point)
        block_path.append(block_pos)

    block_start = (int(block.x), int(block.y))
    append_point(start, block_start)
    x, y = start
    cx, cy = contact_start
    while x != cx:
        x += 1 if cx > x else -1
        append_point((x, y), block_start)
    while y != cy:
        y += 1 if cy > y else -1
        append_point((x, y), block_start)

    dx, dy = push_dir
    bx, by = block_start
    for _ in range(push_distance):
        x += dx
        y += dy
        bx += dx
        by += dy
        append_point((x, y), (bx, by))

    for wx, wy in waypoints:
        while x != wx:
            x += 1 if wx > x else -1
            append_point((x, y), (bx, by))
        while y != wy:
            y += 1 if wy > y else -1
            append_point((x, y), (bx, by))

    gx, gy = goal
    while x != gx:
        x += 1 if gx > x else -1
        append_point((x, y), (bx, by))
    while y != gy:
        y += 1 if gy > y else -1
        append_point((x, y), (bx, by))
    return ball_path, [block_path]


def long_corridor_walls(orientation: str, rng: random.Random, decorate: bool = True) -> list[Rect]:
    walls = add_boundary_walls([])
    if orientation == "horizontal":
        # Horizontal block is pushed downward from its long top edge through the only gate.
        walls.extend(
            [
                Rect(1, 8, 10, 1),
                Rect(15, 8, 11, 1),
                Rect(1, 15, 10, 1),
                Rect(15, 15, 11, 1),
                Rect(6, 1, 1, 5),
                Rect(3, 5, 5, 1),
                Rect(18, 3, 1, 4),
                Rect(20, 5, 4, 1),
                Rect(4, 17, 1, 3),
                Rect(6, 18, 5, 1),
                Rect(18, 18, 5, 1),
            ]
        )
        protected = [Rect(1, 1, 15, 6), Rect(10, 5, 6, 14), Rect(11, 16, 15, 4), Rect(21, 17, 5, 4)]
    else:
        # Vertical block is pushed rightward from its long left edge through the only gate.
        walls.extend(
            [
                Rect(10, 1, 1, 8),
                Rect(10, 13, 1, 7),
                Rect(17, 1, 1, 8),
                Rect(17, 13, 1, 7),
                Rect(3, 4, 5, 1),
                Rect(6, 5, 1, 4),
                Rect(3, 16, 5, 1),
                Rect(6, 17, 1, 3),
                Rect(20, 3, 4, 1),
                Rect(22, 4, 1, 4),
                Rect(20, 16, 4, 1),
            ]
        )
        protected = [Rect(1, 1, 7, 13), Rect(6, 8, 16, 6), Rect(18, 5, 8, 5), Rect(21, 17, 5, 4)]
    if decorate:
        walls = add_irregular_decorative_walls(walls, rng, protected, attempts=40)
    return seal_micro_gaps(walls, [], [Rect(1, 1, 5, 5), Rect(22, 16, 5, 5)])


def lock_short_side_pushes(walls: list[Rect], block: PushBlock) -> list[Rect]:
    result = list(walls)
    if block.orientation == "horizontal":
        blockers = [
            Rect(block.x - 1, block.y, 1, block.h),
            Rect(block.x + block.w, block.y, 1, block.h),
        ]
    elif block.orientation == "vertical":
        blockers = [
            Rect(block.x, block.y - 1, block.w, 1),
            Rect(block.x, block.y + block.h, block.w, 1),
        ]
    else:
        raise ValueError(f"unknown block orientation: {block.orientation}")
    for blocker in blockers:
        if blocker not in result:
            result.append(blocker)
    return result


def fixed_endpoint_reverse_long_corridor(push_direction: str) -> tuple[list[Rect], PushBlock, GridPoint, GridPoint]:
    start = (3, 3)
    goal = (24, 19)
    walls = add_boundary_walls([])
    if push_direction == "up":
        block = PushBlock(13, 12, PUSH_BLOCK_LONG_CM, PUSH_BLOCK_SHORT_CM, "horizontal")
        walls.extend(
            [
                Rect(17, 1, 1, 11),
                Rect(17, 14, 1, 6),
                Rect(12, 12, 1, 2),
                Rect(13, 14, 1, 1),
                Rect(16, 14, 1, 1),
            ]
        )
    elif push_direction == "left":
        block = PushBlock(12, 13, PUSH_BLOCK_SHORT_CM, PUSH_BLOCK_LONG_CM, "vertical")
        walls.extend(
            [
                Rect(1, 17, 11, 1),
                Rect(14, 17, 12, 1),
                Rect(12, 12, 2, 1),
                Rect(14, 13, 1, 1),
                Rect(14, 16, 1, 1),
            ]
        )
    else:
        raise ValueError(f"reverse long corridor only supports up/left, got {push_direction}")
    return walls, block, start, goal


def mirror_rect_for_push_direction(rect: Rect, push_direction: str) -> Rect:
    if push_direction in ("down", "right"):
        return rect
    if push_direction == "up":
        return Rect(rect.x, MAZE_HEIGHT_CM - rect.y - rect.h, rect.w, rect.h)
    if push_direction == "left":
        return Rect(MAZE_WIDTH_CM - rect.x - rect.w, rect.y, rect.w, rect.h)
    raise ValueError(f"unknown long-corridor push direction: {push_direction}")


def mirror_point_for_push_direction(point: GridPoint, push_direction: str) -> GridPoint:
    x, y = point
    if push_direction in ("down", "right"):
        return point
    if push_direction == "up":
        return (x, MAZE_HEIGHT_CM - y)
    if push_direction == "left":
        return (MAZE_WIDTH_CM - x, y)
    raise ValueError(f"unknown long-corridor push direction: {push_direction}")


def long_corridor_orientation_for_direction(push_direction: str) -> str:
    if push_direction in ("up", "down"):
        return "horizontal"
    if push_direction in ("left", "right"):
        return "vertical"
    raise ValueError(f"unknown long-corridor push direction: {push_direction}")


def long_corridor_base_block(orientation: str) -> PushBlock:
    if orientation == "horizontal":
        return PushBlock(11, 6, PUSH_BLOCK_LONG_CM, PUSH_BLOCK_SHORT_CM, "horizontal")
    if orientation == "vertical":
        return PushBlock(8, 9, PUSH_BLOCK_SHORT_CM, PUSH_BLOCK_LONG_CM, "vertical")
    raise ValueError(f"unknown long-corridor orientation: {orientation}")


def transform_long_corridor_layout(
    walls: list[Rect],
    block: PushBlock,
    start: GridPoint,
    goal: GridPoint,
    push_direction: str,
) -> tuple[list[Rect], PushBlock, GridPoint, GridPoint]:
    transformed_walls = [mirror_rect_for_push_direction(wall, push_direction) for wall in walls]
    transformed_block_rect = mirror_rect_for_push_direction(Rect(block.x, block.y, block.w, block.h), push_direction)
    transformed_block = PushBlock(transformed_block_rect.x, transformed_block_rect.y, block.w, block.h, block.orientation)
    transformed_start = mirror_point_for_push_direction(start, push_direction)
    transformed_goal = mirror_point_for_push_direction(goal, push_direction)
    return transformed_walls, transformed_block, transformed_start, transformed_goal


def validate_long_corridor_solution(
    walls: list[Rect],
    blocks: list[PushBlock],
    start: GridPoint,
    goal: GridPoint,
    solved: tuple[list[GridPoint], list[list[GridPoint]]],
    push_direction: str,
) -> None:
    ball_path, block_paths = solved
    long_count, short_count = push_side_counts(blocks, block_paths)
    if long_count <= 0 or short_count != 0:
        raise RuntimeError("long corridor generator produced invalid push side")
    directions = pushed_directions(block_paths)
    expected_direction = LONG_CORRIDOR_DIRECTION_VECTORS[push_direction]
    if not directions or any(direction != expected_direction for direction in directions):
        raise RuntimeError(f"long corridor generator produced non-{push_direction} push directions: {sorted(set(directions))}")
    unrestricted = solve_with_pushing(start, goal, walls, blocks, max_expansions=200000, push_side="any")
    if unrestricted is None:
        raise RuntimeError("long corridor generator produced layout without unrestricted solution")
    if len(unrestricted[0]) != len(ball_path):
        raise RuntimeError(
            f"long corridor long-only path is not shortest: long_only={len(ball_path)} unrestricted={len(unrestricted[0])}"
        )
    unrestricted_long_count, unrestricted_short_count = push_side_counts(blocks, unrestricted[1])
    if unrestricted_long_count <= 0 or unrestricted_short_count != 0:
        raise RuntimeError("long corridor unrestricted shortest path is not strictly long-side pushing")
    unrestricted_directions = pushed_directions(unrestricted[1])
    if not unrestricted_directions or any(direction != expected_direction for direction in unrestricted_directions):
        raise RuntimeError("long corridor unrestricted shortest path uses different push directions")
    best_valid, best_invalid = shortest_goal_lengths_by_push_validity(start, goal, walls, blocks, expected_direction)
    if best_valid != len(ball_path):
        raise RuntimeError(f"long corridor shortest valid path mismatch: valid={best_valid} expected={len(ball_path)}")
    if best_invalid is not None and best_invalid <= len(ball_path):
        raise RuntimeError(f"long corridor has non-strict alternative at {best_invalid} steps; expected longer than {len(ball_path)}")


def long_corridor_direction_for_index(index: int, count: int | None) -> str:
    if count is not None and count % len(LONG_CORRIDOR_4DIR_ORDER) == 0:
        per_direction = count // len(LONG_CORRIDOR_4DIR_ORDER)
        return LONG_CORRIDOR_4DIR_ORDER[min(index // per_direction, len(LONG_CORRIDOR_4DIR_ORDER) - 1)]
    return LONG_CORRIDOR_4DIR_ORDER[index % len(LONG_CORRIDOR_4DIR_ORDER)]


def generate_long_corridor_one(seed: int, index: int, push_direction: str | None = None) -> MazeSample:
    rng = random.Random(seed)
    if push_direction is None:
        push_direction = "down" if index % 2 == 0 else "right"
    orientation = long_corridor_orientation_for_direction(push_direction)
    base_start = (3, 3)
    base_goal = (24, 19)
    base_block = long_corridor_base_block(orientation)

    last_error: Exception | None = None
    for decorate in (True, False):
        if push_direction in ("up", "left"):
            walls, block, start, goal = fixed_endpoint_reverse_long_corridor(push_direction)
        else:
            walls = long_corridor_walls(orientation, rng, decorate=decorate)
            walls, block, start, goal = transform_long_corridor_layout(walls, base_block, base_start, base_goal, push_direction)
            walls = lock_short_side_pushes(walls, block)
        blocks = [block]
        if bfs_path(start, goal, walls, blocks) is not None:
            last_error = RuntimeError("long corridor generator produced a no-push shortcut")
            continue
        solved = solve_with_pushing(start, goal, walls, blocks, max_expansions=200000, push_side="long_only")
        if solved is None:
            last_error = RuntimeError("long corridor generator produced an unsolved layout")
            continue
        try:
            validate_long_corridor_solution(walls, blocks, start, goal, solved, push_direction)
        except RuntimeError as exc:
            last_error = exc
            continue
        ball_path, block_paths = solved
        long_count, _short_count = push_side_counts(blocks, block_paths)
        return MazeSample(walls, blocks, start, goal, ball_path, block_paths, False, 6, len(ball_path), long_count)
    if last_error is not None:
        raise last_error
    raise RuntimeError("long corridor generator produced no candidate layouts")


def x_cm_to_px(value: float) -> int:
    return round(value * SCALE)


def y_cm_to_px(value: float) -> int:
    return round(Y_OFFSET_PX + value * SCALE)


def draw_rect(draw: ImageDraw.ImageDraw, rect: Rect, fill: tuple[int, int, int], outline: tuple[int, int, int] | None = None) -> None:
    draw.rectangle((x_cm_to_px(rect.x), y_cm_to_px(rect.y), x_cm_to_px(rect.x + rect.w), y_cm_to_px(rect.y + rect.h)), fill=fill, outline=outline)


def draw_frame(sample: MazeSample, frame_index: int) -> np.ndarray:
    image = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    maze_rect = (x_cm_to_px(0), y_cm_to_px(0), x_cm_to_px(MAZE_WIDTH_CM), y_cm_to_px(MAZE_HEIGHT_CM))
    draw.rectangle(maze_rect, fill=(238, 239, 232), outline=(210, 210, 205))
    for wall in sample.walls:
        draw_rect(draw, wall, (165, 124, 75), (115, 82, 43))
    for block_index, block in enumerate(sample.push_blocks):
        bx, by = sample.solution_block_paths[block_index][frame_index]
        draw_rect(draw, Rect(bx, by, block.w, block.h), (8, 9, 8), (0, 0, 0))
    x, y = sample.solution_ball_path[frame_index]
    draw.ellipse(
        (
            x_cm_to_px(x - RENDER_BALL_RADIUS_CM),
            y_cm_to_px(y - RENDER_BALL_RADIUS_CM),
            x_cm_to_px(x + RENDER_BALL_RADIUS_CM),
            y_cm_to_px(y + RENDER_BALL_RADIUS_CM),
        ),
        fill=(255, 0, 0),
    )
    return np.asarray(image)


def adjusted_contact_position(ball: Point, block: Point, block_shape: PushBlock, direction: Dir) -> Point:
    dx, dy = direction
    bx, by = block
    if (dx, dy) == (1, 0):
        return (bx - RENDER_BALL_RADIUS_CM, ball[1])
    if (dx, dy) == (-1, 0):
        return (bx + block_shape.w + RENDER_BALL_RADIUS_CM, ball[1])
    if (dx, dy) == (0, 1):
        return (ball[0], by - RENDER_BALL_RADIUS_CM)
    if (dx, dy) == (0, -1):
        return (ball[0], by + block_shape.h + RENDER_BALL_RADIUS_CM)
    return ball


def adjust_ball_for_visual_push(ball: Point, block: Point | None, block_shape: PushBlock | None, step_start: Point, step_end: Point) -> Point:
    if block is None or block_shape is None:
        return ball
    direction = (int(round(step_end[0] - step_start[0])), int(round(step_end[1] - step_start[1])))
    if direction == (0, 0):
        return ball
    return adjusted_contact_position(ball, block, block_shape, direction)


def expand_path_to_frames(sample: MazeSample) -> MazeSample:
    balls: list[Point] = []
    block_paths: list[list[Point]] = [[] for _ in sample.push_blocks]
    for step_index, (start_ball, end_ball) in enumerate(zip(sample.solution_ball_path, sample.solution_ball_path[1:])):
        distance = abs(end_ball[0] - start_ball[0]) + abs(end_ball[1] - start_ball[1])
        frame_count = max(1, int(round(distance * FRAMES_PER_CM)))
        moving_block_index = next(
            (
                index
                for index in range(len(sample.push_blocks))
                if sample.solution_block_paths[index][step_index] != sample.solution_block_paths[index][step_index + 1]
            ),
            None,
        )
        for substep in range(frame_count):
            t = substep / frame_count
            ball = (start_ball[0] + (end_ball[0] - start_ball[0]) * t, start_ball[1] + (end_ball[1] - start_ball[1]) * t)
            for block_index in range(len(sample.push_blocks)):
                start_block = sample.solution_block_paths[block_index][step_index]
                end_block = sample.solution_block_paths[block_index][step_index + 1]
                block = (start_block[0] + (end_block[0] - start_block[0]) * t, start_block[1] + (end_block[1] - start_block[1]) * t)
                block_paths[block_index].append(block)
            moving_block = block_paths[moving_block_index][-1] if moving_block_index is not None else None
            moving_shape = sample.push_blocks[moving_block_index] if moving_block_index is not None else None
            balls.append(adjust_ball_for_visual_push(ball, moving_block, moving_shape, start_ball, end_ball))
    balls.append(sample.solution_ball_path[-1])
    for block_index in range(len(sample.push_blocks)):
        block_paths[block_index].append(sample.solution_block_paths[block_index][-1])
    return MazeSample(
        sample.walls,
        sample.push_blocks,
        sample.start,
        sample.goal,
        balls,
        block_paths,
        sample.no_push_reachable,
        sample.branch_score,
        sample.raw_path_steps,
        sample.raw_push_steps,
    )


def save_sample(sample: MazeSample, output_dir: Path, index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = expand_path_to_frames(sample)
    video_path = output_dir / f"irregular_push_maze_{index:04d}.mp4"
    json_path = output_dir / f"irregular_push_maze_{index:04d}.json"
    with imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_index in range(len(sample.solution_ball_path)):
            writer.append_data(draw_frame(sample, frame_index))
    with json_path.open("w") as f:
        json.dump(asdict(sample), f, indent=2)


def search_long_direction_for_index(index: int, seed: int, count: int, min_moved_blocks: int = 1, attempts: int = 300) -> MazeSample:
    target_direction = long_corridor_direction_for_index(index, count)
    for offset in range(attempts):
        try:
            sample = generate_one(seed + offset * count, block_count=1, max_attempts=30, push_side="long_only", min_moved_blocks=min_moved_blocks)
        except RuntimeError:
            continue
        directions = pushed_directions(sample.solution_block_paths)
        expected_direction = LONG_CORRIDOR_DIRECTION_VECTORS[target_direction]
        if directions and all(direction == expected_direction for direction in directions):
            return sample
    raise RuntimeError(f"failed to generate search long push direction {target_direction} after {attempts} attempts")


def generate_and_save_task(args: tuple[int, int, int, str, int | None, bool, str, str, int, int, str | None]) -> dict[str, object]:
    index, seed, count, output_dir_string, block_count, resume, push_side, generator, min_moved_blocks, max_attempts, target_direction_arg = args
    output_dir = Path(output_dir_string)
    video_path = output_dir / f"irregular_push_maze_{index:04d}.mp4"
    json_path = output_dir / f"irregular_push_maze_{index:04d}.json"
    if resume and video_path.exists() and json_path.exists():
        return {"index": index, "status": "skipped"}
    push_direction = None
    if generator == "long_corridor":
        sample = generate_long_corridor_one(seed, index)
        push_direction = "down" if index % 2 == 0 else "right"
    elif generator == "long_corridor_4dir":
        push_direction = long_corridor_direction_for_index(index, count)
        sample = generate_long_corridor_one(seed, index, push_direction)
    elif generator == "search_long_4dir":
        push_direction = long_corridor_direction_for_index(index, count)
        sample = search_long_direction_for_index(index, seed, count, min_moved_blocks=min_moved_blocks)
    elif generator == "search_multi_block":
        sample = generate_one(
            seed,
            block_count=block_count or max(3, min_moved_blocks),
            max_attempts=max_attempts,
            push_side=push_side,
            min_moved_blocks=min_moved_blocks,
        )
    elif generator == "search_required_blocks":
        push_direction = target_direction_arg if target_direction_arg != "balanced" else long_corridor_direction_for_index(index, count)
        sample = generate_required_blocks_one(
            seed,
            block_count=block_count or max(3, min_moved_blocks),
            min_moved_blocks=min_moved_blocks,
            max_attempts=max_attempts,
            target_direction=push_direction,
            push_side=push_side,
        )
    else:
        sample = generate_one(seed, block_count, max_attempts=max_attempts, push_side=push_side, min_moved_blocks=min_moved_blocks)
    save_sample(sample, output_dir, index)
    return {
        "index": index,
        "status": "generated",
        "walls": len(sample.walls),
        "blocks": len(sample.push_blocks),
        "raw_path_steps": sample.raw_path_steps,
        "raw_push_steps": sample.raw_push_steps,
        "moved_blocks": len(moved_block_indices(sample.solution_block_paths)),
        "branch_score": sample.branch_score,
        "push_side": push_side,
        "generator": generator,
        "push_direction": push_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate irregular 27cm x 21cm push mazes with cuboid walls.")
    parser.add_argument("--output-dir", type=Path, default=Path("irregular_push_maze_sample"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-count", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--push-side", type=str, default="any", choices=["any", "long_only", "short_only"])
    parser.add_argument(
        "--generator",
        type=str,
        default="search",
        choices=["search", "long_corridor", "long_corridor_4dir", "search_long_4dir", "search_multi_block", "search_required_blocks"],
    )
    parser.add_argument("--min-moved-blocks", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--target-direction", type=str, default=None, choices=[None, "balanced", "up", "down", "left", "right"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            index,
            args.seed + index,
            args.count,
            str(args.output_dir),
            args.block_count,
            args.resume,
            args.push_side,
            args.generator,
            args.min_moved_blocks,
            args.max_attempts,
            args.target_direction,
        )
        for index in range(args.count)
    ]
    if args.workers <= 1:
        for offset, task in enumerate(tasks, 1):
            result = generate_and_save_task(task)
            if offset == 1 or offset % args.progress_every == 0 or offset == len(tasks):
                print(f"{offset}/{len(tasks)} {result}")
        return

    with mp.Pool(processes=args.workers) as pool:
        for offset, result in enumerate(pool.imap_unordered(generate_and_save_task, tasks), 1):
            if offset == 1 or offset % args.progress_every == 0 or offset == len(tasks):
                print(f"{offset}/{len(tasks)} {result}", flush=True)


if __name__ == "__main__":
    main()
