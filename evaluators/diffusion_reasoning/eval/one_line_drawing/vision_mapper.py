from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from evaluators.diffusion_reasoning.utils.video_io import load_video_frames

Cell = tuple[int, int]

DEFAULT_BOARD_MARGIN = 16
GREEN_THRESHOLD = 135
WHITE_THRESHOLD = 215
HEAD_MAX_DISTANCE_RATIO = 0.75
CELL_VISIT_THRESHOLD = 0.08
TRACE_ON_SHAPE_MIN_RATIO = 0.97


@dataclass(frozen=True)
class Layout:
    cell_size: int
    cell_centers: dict[Cell, tuple[int, int]]
    cell_masks: dict[Cell, np.ndarray]
    valid_path_mask: np.ndarray
    shape: tuple[int, int]


def extract_state_trace(video_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    frames = load_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    layout = _build_layout(metadata)
    static_white_mask = _compute_static_white_mask(frames[0], layout.shape)
    occupied_cells = {tuple(cell) for cell in metadata["graph"]["occupied_cells"]}
    start_cell = tuple(metadata["graph"]["start_cell"])

    frame_states: list[dict[str, Any]] = []
    for frame in frames:
        green_mask = _green_trace_mask(frame, layout.shape)
        moving_head_mask = _moving_head_mask(frame, static_white_mask, layout.shape)
        head_point = _mask_centroid(moving_head_mask)
        head_cell = _resolve_head_cell(head_point, layout)
        visited_cells = _resolve_visited_cells(green_mask, layout, start_cell)
        trace_ratio = _mask_inside_ratio(green_mask, layout.valid_path_mask)
        head_on_shape = _point_inside_mask(head_point, layout.valid_path_mask)
        frame_states.append(
            {
                "head_point": head_point,
                "head_cell": head_cell,
                "visited_cells": sorted(visited_cells),
                "trace_pixels": int(green_mask.sum()),
                "trace_on_shape_ratio": trace_ratio,
                "trace_on_shape": trace_ratio >= TRACE_ON_SHAPE_MIN_RATIO,
                "head_on_shape": head_on_shape,
                "on_shape": (trace_ratio >= TRACE_ON_SHAPE_MIN_RATIO) and head_on_shape,
            }
        )

    return {
        "video_path": str(video_path),
        "frame_count": len(frames),
        "layout": {
            "cell_size": layout.cell_size,
            "cell_centers": {f"{cell[0]},{cell[1]}": center for cell, center in layout.cell_centers.items()},
        },
        "frame_states": frame_states,
    }


def _build_layout(metadata: dict[str, Any]) -> Layout:
    occupied_cells = [tuple(cell) for cell in metadata["graph"]["occupied_cells"]]
    width, height = metadata["render"]["resolution"]
    xs = [cell[0] for cell in occupied_cells]
    ys = [cell[1] for cell in occupied_cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cols = max_x - min_x + 1
    rows = max_y - min_y + 1

    usable_w = width - DEFAULT_BOARD_MARGIN * 2
    usable_h = height - DEFAULT_BOARD_MARGIN * 2
    cell_size = max(10, int(min(usable_w / max(cols, 1), usable_h / max(rows, 1))))
    board_w = cols * cell_size
    board_h = rows * cell_size
    origin_x = int((width - board_w) / 2)
    origin_y = int((height - board_h) / 2)

    cell_centers: dict[Cell, tuple[int, int]] = {}
    for cell in occupied_cells:
        shifted_x = cell[0] - min_x
        shifted_y = cell[1] - min_y
        center_x = origin_x + shifted_x * cell_size + cell_size // 2
        center_y = origin_y + shifted_y * cell_size + cell_size // 2
        cell_centers[cell] = (center_x, center_y)

    shape = (height, width)
    cell_masks = {
        cell: _build_cell_mask(shape, center, max(5, int(round(cell_size * 0.3))))
        for cell, center in cell_centers.items()
    }
    valid_path_mask = _build_valid_path_mask(shape, cell_centers, cell_size)
    return Layout(
        cell_size=cell_size,
        cell_centers=cell_centers,
        cell_masks=cell_masks,
        valid_path_mask=valid_path_mask,
        shape=shape,
    )


def _build_cell_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=1)
    return np.asarray(image, dtype=bool)


def _build_valid_path_mask(
    shape: tuple[int, int],
    cell_centers: dict[Cell, tuple[int, int]],
    cell_size: int,
) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    half_cell = max(6, int(round(cell_size * 0.5)))
    line_thickness = max(6, int(round(cell_size * 0.52)))

    for center in cell_centers.values():
        cx, cy = center
        draw.rectangle((cx - half_cell, cy - half_cell, cx + half_cell, cy + half_cell), fill=1)

    cells = set(cell_centers)
    for cell, center in cell_centers.items():
        x, y = cell
        for neighbor in ((x + 1, y), (x, y + 1)):
            if neighbor in cells:
                draw.line((center, cell_centers[neighbor]), fill=1, width=line_thickness)

    return np.asarray(image, dtype=bool)


def _compute_static_white_mask(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    white_mask = _white_mask(frame, shape)
    image = Image.fromarray((white_mask.astype(np.uint8) * 255), mode="L")
    expanded = image.filter(ImageFilter.MaxFilter(size=5))
    return np.asarray(expanded, dtype=bool)


def _white_mask(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rgb = frame[..., :3]
    mask = (
        (rgb[..., 0] >= WHITE_THRESHOLD)
        & (rgb[..., 1] >= WHITE_THRESHOLD)
        & (rgb[..., 2] >= WHITE_THRESHOLD)
    )
    return mask.reshape(shape)


def _green_trace_mask(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rgb = frame[..., :3].astype(np.int16)
    mask = (
        (rgb[..., 1] >= GREEN_THRESHOLD)
        & ((rgb[..., 1] - rgb[..., 0]) >= 18)
        & ((rgb[..., 1] - rgb[..., 2]) >= 6)
    )
    return mask.reshape(shape)


def _moving_head_mask(frame: np.ndarray, static_white_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    current_white_mask = _white_mask(frame, shape)
    moving = current_white_mask & ~static_white_mask
    if moving.any():
        return moving
    return np.zeros(shape, dtype=bool)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(mask)
    if len(points) == 0:
        return None
    y, x = points.mean(axis=0)
    return (float(x), float(y))


def _resolve_head_cell(head_point: tuple[float, float] | None, layout: Layout) -> Cell | None:
    if head_point is None:
        return None

    best_cell = None
    best_distance = None
    for cell, center in layout.cell_centers.items():
        distance = float(np.hypot(head_point[0] - center[0], head_point[1] - center[1]))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_cell = cell

    if best_distance is None or best_distance > layout.cell_size * HEAD_MAX_DISTANCE_RATIO:
        return None
    return best_cell


def _resolve_visited_cells(green_mask: np.ndarray, layout: Layout, start_cell: Cell) -> set[Cell]:
    visited = {start_cell}
    for cell, mask in layout.cell_masks.items():
        total = int(mask.sum())
        if total == 0:
            continue
        covered = int(np.count_nonzero(green_mask & mask))
        if covered / total >= CELL_VISIT_THRESHOLD:
            visited.add(cell)
    return visited


def _mask_inside_ratio(mask: np.ndarray, valid_mask: np.ndarray) -> float:
    total = int(mask.sum())
    if total == 0:
        return 1.0
    inside = int(np.count_nonzero(mask & valid_mask))
    return inside / total


def _point_inside_mask(point: tuple[float, float] | None, mask: np.ndarray) -> bool:
    if point is None:
        return True
    x = int(round(point[0]))
    y = int(round(point[1]))
    if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])
