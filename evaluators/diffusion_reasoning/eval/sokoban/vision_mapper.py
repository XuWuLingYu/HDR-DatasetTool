from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from evaluators.diffusion_reasoning.utils.video_io import load_video_frames

Cell = tuple[int, int]
State = tuple[Cell, Cell]

BACKGROUND = (32, 34, 38)
PANEL = (16, 19, 26)
FLOOR = (18, 22, 28)
FLOOR_GRID = (62, 67, 77)
WALL_FILL = (124, 134, 164)
TARGET = (130, 225, 127)
PLAYER = (255, 92, 122)
BOX = (244, 183, 78)

RESOLVED_ERROR_THRESHOLD = 0.14
RESOLVED_MARGIN_THRESHOLD = 0.001
ENTITY_ERROR_THRESHOLD = 0.24


@dataclass(frozen=True)
class Layout:
    board_size: int
    wall_rows: tuple[str, ...]
    floor_cells: tuple[Cell, ...]
    wall_cells: tuple[Cell, ...]
    target: Cell
    cell_rects: dict[Cell, tuple[int, int, int, int]]
    templates: dict[str, np.ndarray]
    floor_templates: dict[Cell, np.ndarray]
    target_distance: dict[Cell, int]


def extract_state_trace(
    video_path: str | Path,
    metadata: dict[str, Any],
    *,
    gt_video_path: str | Path | None = None,
) -> dict[str, Any]:
    frames = load_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    layout = _build_layout(metadata)
    initial_player = tuple(int(value) for value in metadata["puzzle"]["initial_player"])
    initial_box = tuple(int(value) for value in metadata["puzzle"]["initial_box"])

    frame_states: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        frame_states.append(_decode_frame(frame, frame_index, layout))

    resolved_states = _build_resolved_state_sequence(frame_states)
    trace_rule_adherence_ratio = sum(
        1
        for frame in frame_states
        if frame["resolved"]
        and frame["player_on_floor"]
        and frame["box_on_floor"]
        and not frame["overlap_detected"]
    ) / max(len([frame for frame in frame_states if frame["resolved"]]), 1)
    trace_observability_ratio = min(len(resolved_states) / max(int(metadata["solution"]["action_count"]) + 1, 1), 1.0)

    return {
        "video_path": str(video_path),
        "frame_count": len(frames),
        "layout": {
            "board_size": layout.board_size,
            "wall_rows": list(layout.wall_rows),
            "floor_cells": [list(cell) for cell in layout.floor_cells],
            "wall_cells": [list(cell) for cell in layout.wall_cells],
            "target": list(layout.target),
            "initial_box_distance": layout.target_distance[initial_box],
        },
        "frame_states": frame_states,
        "resolved_states": resolved_states,
        "trace_rule_adherence_ratio": trace_rule_adherence_ratio,
        "trace_observability_ratio": trace_observability_ratio,
        "initial_state": {"player": list(initial_player), "box": list(initial_box)},
    }


def _build_layout(metadata: dict[str, Any]) -> Layout:
    board_size = int(metadata["puzzle"]["board_size"][0])
    wall_rows = tuple(str(row) for row in metadata["puzzle"]["wall_rows"])
    target = tuple(int(value) for value in metadata["puzzle"]["target"])
    width, height = metadata["render"]["resolution"]
    geometry = _board_geometry(width, height, board_size)

    floor_cells: list[Cell] = []
    wall_cells: list[Cell] = []
    cell_rects: dict[Cell, tuple[int, int, int, int]] = {}
    for row in range(board_size):
        for col in range(board_size):
            cell = (row, col)
            cell_rects[cell] = _cell_rect(cell, geometry)
            if wall_rows[row][col] == ".":
                floor_cells.append(cell)
            else:
                wall_cells.append(cell)

    templates = _build_templates(cell_rects[target][2] - cell_rects[target][0], cell_rects[target][3] - cell_rects[target][1])
    floor_templates = {cell: templates["empty_target"] if cell == target else templates["empty_floor"] for cell in floor_cells}
    target_distance = _compute_target_distance(target, wall_rows)
    return Layout(
        board_size=board_size,
        wall_rows=wall_rows,
        floor_cells=tuple(floor_cells),
        wall_cells=tuple(wall_cells),
        target=target,
        cell_rects=cell_rects,
        templates=templates,
        floor_templates=floor_templates,
        target_distance=target_distance,
    )


def _decode_frame(frame: np.ndarray, frame_index: int, layout: Layout) -> dict[str, Any]:
    cell_errors: dict[Cell, dict[str, float]] = {}
    for cell in layout.floor_cells:
        patch = _crop_patch(frame, layout.cell_rects[cell])
        statuses = ("empty_target", "player_target", "box_target") if cell == layout.target else ("empty_floor", "player_floor", "box_floor")
        cell_errors[cell] = {status: _normalized_error(patch, layout.templates[status]) for status in statuses}

    empty_total = sum(cell_errors[cell][("empty_target" if cell == layout.target else "empty_floor")] for cell in layout.floor_cells)
    best_cost = None
    second_cost = None
    best_state: State | None = None
    best_entity_errors = (1.0, 1.0)

    for player in layout.floor_cells:
        player_key = "player_target" if player == layout.target else "player_floor"
        player_delta = cell_errors[player][player_key] - cell_errors[player][("empty_target" if player == layout.target else "empty_floor")]
        for box in layout.floor_cells:
            if box == player:
                continue
            box_key = "box_target" if box == layout.target else "box_floor"
            cost = empty_total + player_delta
            cost += cell_errors[box][box_key] - cell_errors[box][("empty_target" if box == layout.target else "empty_floor")]
            if best_cost is None or cost < best_cost:
                second_cost = best_cost
                best_cost = cost
                best_state = (player, box)
                best_entity_errors = (cell_errors[player][player_key], cell_errors[box][box_key])
            elif second_cost is None or cost < second_cost:
                second_cost = cost

    if best_state is None or best_cost is None:
        return _unresolved_frame(frame_index)

    average_error = best_cost / max(len(layout.floor_cells), 1)
    margin = 0.0 if second_cost is None else (second_cost - best_cost) / max(len(layout.floor_cells), 1)
    player, box = best_state
    player_on_floor = layout.wall_rows[player[0]][player[1]] == "."
    box_on_floor = layout.wall_rows[box[0]][box[1]] == "."
    overlap_detected = player == box
    resolved = (
        average_error <= RESOLVED_ERROR_THRESHOLD
        and margin >= RESOLVED_MARGIN_THRESHOLD
        and best_entity_errors[0] <= ENTITY_ERROR_THRESHOLD
        and best_entity_errors[1] <= ENTITY_ERROR_THRESHOLD
        and player_on_floor
        and box_on_floor
        and not overlap_detected
    )
    box_distance = layout.target_distance.get(box)

    return {
        "frame_index": frame_index,
        "player_cell": list(player) if resolved else None,
        "box_cell": list(box) if resolved else None,
        "resolved": resolved,
        "in_motion": not resolved,
        "player_on_floor": player_on_floor,
        "box_on_floor": box_on_floor,
        "overlap_detected": overlap_detected,
        "box_on_target": resolved and box == layout.target,
        "player_box_adjacent": resolved and _adjacent(player, box),
        "box_target_distance": box_distance if resolved else None,
        "state": {"player": list(player), "box": list(box)} if resolved else None,
        "avg_error": average_error,
        "state_margin": margin,
        "player_error": float(best_entity_errors[0]),
        "box_error": float(best_entity_errors[1]),
    }


def _unresolved_frame(frame_index: int) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "player_cell": None,
        "box_cell": None,
        "resolved": False,
        "in_motion": True,
        "player_on_floor": False,
        "box_on_floor": False,
        "overlap_detected": False,
        "box_on_target": False,
        "player_box_adjacent": False,
        "box_target_distance": None,
        "state": None,
        "avg_error": 1.0,
        "state_margin": 0.0,
        "player_error": 1.0,
        "box_error": 1.0,
    }


def _build_resolved_state_sequence(frame_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    previous: State | None = None
    for frame in frame_states:
        state = frame["state"]
        if state is None:
            continue
        current: State = (
            tuple(int(value) for value in state["player"]),
            tuple(int(value) for value in state["box"]),
        )
        if current == previous:
            continue
        resolved.append(
            {
                "frame_index": frame["frame_index"],
                "state": {"player": list(current[0]), "box": list(current[1])},
            }
        )
        previous = current
    return resolved


def _board_geometry(width: int, height: int, board_size: int) -> dict[str, float]:
    panel_margin = 6.0
    board_padding = 18.0
    cell_gap = 2.0
    board_extent = min(width, height) - 2 * board_padding
    cell_size = (board_extent - (board_size - 1) * cell_gap) / board_size
    board_size_px = board_size * cell_size + (board_size - 1) * cell_gap
    origin_x = (width - board_size_px) / 2
    origin_y = (height - board_size_px) / 2
    return {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "cell_size": cell_size,
        "cell_gap": cell_gap,
        "panel_margin": panel_margin,
    }


def _cell_rect(cell: Cell, geometry: dict[str, float]) -> tuple[int, int, int, int]:
    row, col = cell
    x0 = geometry["origin_x"] + col * (geometry["cell_size"] + geometry["cell_gap"])
    y0 = geometry["origin_y"] + row * (geometry["cell_size"] + geometry["cell_gap"])
    x1 = x0 + geometry["cell_size"]
    y1 = y0 + geometry["cell_size"]
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))


def _build_templates(width: int, height: int) -> dict[str, np.ndarray]:
    templates = {
        "empty_floor": _render_patch(width, height, include_target=False, include_player=False, include_box=False),
        "empty_target": _render_patch(width, height, include_target=True, include_player=False, include_box=False),
        "player_floor": _render_patch(width, height, include_target=False, include_player=True, include_box=False),
        "player_target": _render_patch(width, height, include_target=True, include_player=True, include_box=False),
        "box_floor": _render_patch(width, height, include_target=False, include_player=False, include_box=True),
        "box_target": _render_patch(width, height, include_target=True, include_player=False, include_box=True),
    }
    return templates


def _render_patch(width: int, height: int, *, include_target: bool, include_player: bool, include_box: bool) -> np.ndarray:
    image = Image.new("RGB", (width, height), FLOOR)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=FLOOR_GRID, width=1, fill=FLOOR)
    if include_target:
        inset = width * 0.22
        ring_width = max(2, int(width * 0.08))
        draw.ellipse((inset, inset, width - inset, height - inset), outline=TARGET, width=ring_width)
    if include_box:
        inset = width * 0.20
        draw.rectangle((inset, inset, width - inset, height - inset), fill=BOX)
    if include_player:
        inset_x = width * 0.18
        inset_y = height * 0.14
        draw.polygon(
            [
                (width / 2, inset_y),
                (width - inset_x, height - inset_y),
                (inset_x, height - inset_y),
            ],
            fill=PLAYER,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


def _crop_patch(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return frame[y0:y1, x0:x1, :3].astype(np.float32) / 255.0


def _normalized_error(patch: np.ndarray, template: np.ndarray) -> float:
    if patch.shape != template.shape:
        patch = _resize_patch(patch, template.shape[:2])
    return float(np.mean(np.abs(patch - template)))


def _resize_patch(patch: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.clip(patch * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _compute_target_distance(target: Cell, wall_rows: tuple[str, ...]) -> dict[Cell, int]:
    queue: list[Cell] = [target]
    distance = {target: 0}
    index = 0
    while index < len(queue):
        row, col = queue[index]
        index += 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if nxt in distance:
                continue
            if not (0 <= nxt[0] < len(wall_rows) and 0 <= nxt[1] < len(wall_rows[0])):
                continue
            if wall_rows[nxt[0]][nxt[1]] != ".":
                continue
            distance[nxt] = distance[(row, col)] + 1
            queue.append(nxt)
    return distance


def _adjacent(left: Cell, right: Cell) -> bool:
    return abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1
