from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from evaluators.diffusion_reasoning.utils.video_io import load_video_frames

ColorId = int
TubeState = tuple[ColorId, ...]
State = tuple[TubeState, ...]
Move = tuple[int, int, int]

EMPTY_DISTANCE_THRESHOLD = 34.0
COLOR_DISTANCE_THRESHOLD = 62.0
EMPTY_RATIO_THRESHOLD = 0.55
COLOR_RATIO_THRESHOLD = 0.35
MOVING_BLOCK_MIN_PIXELS = 28
MOVING_BLOCK_MAX_PIXELS = 2400
MOVING_BLOCK_MIN_DOMINANT_RATIO = 0.72
UNKNOWN_SLOT_RATIO_WARNING = 0.15


@dataclass(frozen=True)
class Layout:
    shape: tuple[int, int]
    tube_geometry: list[dict[str, float]]
    slot_centers: list[list[tuple[float, float]]]
    slot_sample_masks: list[list[np.ndarray]]
    tube_inner_masks: list[np.ndarray]
    tube_body_union_mask: np.ndarray
    palette_rgb: list[tuple[int, int, int]]
    empty_reference_rgb: tuple[int, int, int]


def extract_state_trace(video_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    frames = load_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    layout = _build_layout(metadata)
    expected_inventory = _count_inventory(metadata["puzzle"]["initial_tubes"])
    expected_block_count = sum(expected_inventory.values())

    frame_states: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        raw_slots = _decode_slots(frame, layout)
        stable_state, unknown_slots = _collapse_slots_to_state(raw_slots)
        moving_block = _detect_moving_block(frame, layout)
        if moving_block is not None and moving_block["color"] is not None:
            moving_color = int(moving_block["color"])
        else:
            moving_color = None
        in_motion = moving_block is not None

        tube_block_count = sum(len(tube) for tube in stable_state) if stable_state is not None else 0
        conservation_ok = False
        counts_match = False
        state_counts: dict[int, int] = {}
        if stable_state is not None:
            state_counts = _count_inventory(stable_state)
            effective_counts = dict(state_counts)
            if moving_color is not None:
                effective_counts[moving_color] = effective_counts.get(moving_color, 0) + 1
            counts_match = effective_counts == expected_inventory
            conservation_ok = _inventory_matches_with_motion_tolerance(
                state_counts=state_counts,
                expected_inventory=expected_inventory,
                moving_color=moving_color,
                expected_block_count=expected_block_count,
                tube_block_count=tube_block_count,
            )

        capacity_ok = stable_state is not None and all(
            len(tube) <= int(metadata["puzzle"]["capacity"]) for tube in stable_state
        )
        placement_ok = stable_state is not None and _is_state_stack_valid(stable_state)
        resolved = stable_state is not None and conservation_ok and capacity_ok and placement_ok and moving_block is None
        solved_tube_count = 0
        monochrome_tube_count = 0
        block_progress = 0.0
        if stable_state is not None:
            solved_tube_count = _solved_tube_count(stable_state, int(metadata["puzzle"]["capacity"]))
            monochrome_tube_count = _monochrome_tube_count(stable_state)
            block_progress = _block_progress_ratio(stable_state, int(metadata["puzzle"]["capacity"]))

        resolved_move = None

        frame_state = {
            "frame_index": frame_index,
            "raw_slots": raw_slots,
            "tube_state": _state_to_lists(stable_state),
            "resolved": resolved,
            "unknown_slots": unknown_slots,
            "unknown_ratio": unknown_slots / max(len(raw_slots) * len(raw_slots[0]), 1),
            "moving_block": moving_block,
            "in_motion": in_motion,
            "conservation_ok": conservation_ok,
            "counts_match": counts_match,
            "capacity_ok": capacity_ok,
            "placement_ok": placement_ok,
            "solved_tube_count": solved_tube_count,
            "monochrome_tube_count": monochrome_tube_count,
            "block_progress_ratio": block_progress,
            "resolved_move": list(resolved_move) if resolved_move is not None else None,
        }
        frame_states.append(frame_state)

    _fill_motion_gaps(frame_states)
    resolved_states = _build_resolved_state_sequence(frame_states, int(metadata["puzzle"]["capacity"]))

    return {
        "video_path": str(video_path),
        "frame_count": len(frames),
        "layout": {
            "tube_count": len(layout.tube_geometry),
            "tube_geometry": layout.tube_geometry,
            "slot_centers": layout.slot_centers,
        },
        "expected_inventory": {str(key): value for key, value in expected_inventory.items()},
        "frame_states": frame_states,
        "resolved_states": resolved_states,
    }


def _build_layout(metadata: dict[str, Any]) -> Layout:
    width, height = metadata["render"]["resolution"]
    tube_count = int(metadata["puzzle"]["tube_count"])
    capacity = int(metadata["puzzle"]["capacity"])
    layout_variant = metadata["render"]["layout_variant"]
    palette_rgb = [_hex_to_rgb(value) for value in metadata["render"]["palette"]]

    style = _render_style_defaults()
    tube_geometry = _tube_layout(width, height, tube_count, layout_variant, style)
    slot_centers: list[list[tuple[float, float]]] = []
    slot_sample_masks: list[list[np.ndarray]] = []
    tube_inner_masks: list[np.ndarray] = []
    union_mask = np.zeros((height, width), dtype=bool)
    for geom in tube_geometry:
        centers: list[tuple[float, float]] = []
        slot_masks: list[np.ndarray] = []
        inner_mask = _build_tube_body_mask((height, width), geom)
        tube_inner_masks.append(inner_mask)
        union_mask |= inner_mask
        slot_width = _block_width(geom["width"])
        slot_height = _block_height(geom["height"], capacity)
        sample_width = max(4, int(round(slot_width * 0.32)))
        sample_height = max(4, int(round(slot_height * 0.32)))
        for block_idx in range(capacity):
            center = _block_center(geom, block_idx, capacity)
            centers.append(center)
            slot_masks.append(_build_rect_mask((height, width), center, sample_width, sample_height))
        slot_centers.append(centers)
        slot_sample_masks.append(slot_masks)

    return Layout(
        shape=(height, width),
        tube_geometry=tube_geometry,
        slot_centers=slot_centers,
        slot_sample_masks=slot_sample_masks,
        tube_inner_masks=tube_inner_masks,
        tube_body_union_mask=union_mask,
        palette_rgb=palette_rgb,
        empty_reference_rgb=_hex_to_rgb(style["tube_inner"]),
    )


def _decode_slots(frame: np.ndarray, layout: Layout) -> list[list[int | None | str]]:
    decoded: list[list[int | None | str]] = []
    for slot_masks in layout.slot_sample_masks:
        tube_slots: list[int | None | str] = []
        for mask in slot_masks:
            slot_rgb = frame[mask]
            label = _classify_slot(slot_rgb, layout.palette_rgb, layout.empty_reference_rgb)
            tube_slots.append(label)
        decoded.append(tube_slots)
    return decoded


def _classify_slot(
    slot_rgb: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    empty_reference_rgb: tuple[int, int, int],
) -> int | None | str:
    if slot_rgb.size == 0:
        return "unknown"
    pixels = slot_rgb.astype(np.float32)
    empty = np.asarray(empty_reference_rgb, dtype=np.float32)
    palette = np.asarray(palette_rgb, dtype=np.float32)
    empty_distances = np.linalg.norm(pixels - empty, axis=1)
    color_distances = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
    best_indices = np.argmin(color_distances, axis=1)
    best_distances = np.min(color_distances, axis=1)

    empty_ratio = float(np.mean(empty_distances <= EMPTY_DISTANCE_THRESHOLD))
    color_close = best_distances <= COLOR_DISTANCE_THRESHOLD * 1.35
    if len(palette_rgb) == 0:
        return None if empty_ratio >= EMPTY_RATIO_THRESHOLD else "unknown"

    bincount = np.bincount(best_indices[color_close], minlength=len(palette_rgb)) if np.any(color_close) else np.zeros(len(palette_rgb), dtype=int)
    best_idx = int(np.argmax(bincount))
    color_ratio = float(bincount[best_idx] / max(len(pixels), 1))

    if color_ratio >= COLOR_RATIO_THRESHOLD and color_ratio > empty_ratio:
        return best_idx
    if empty_ratio >= EMPTY_RATIO_THRESHOLD:
        return None
    return "unknown"


def _collapse_slots_to_state(raw_slots: list[list[int | None | str]]) -> tuple[State | None, int]:
    tubes: list[TubeState] = []
    unknown_slots = 0
    for tube_slots in raw_slots:
        blocks: list[int] = []
        empty_seen = False
        invalid = False
        for slot in tube_slots:
            if slot == "unknown":
                unknown_slots += 1
                invalid = True
                break
            if slot is None:
                empty_seen = True
                continue
            if empty_seen:
                invalid = True
                break
            blocks.append(int(slot))
        if invalid:
            return None, unknown_slots
        tubes.append(tuple(blocks))
    return tuple(tubes), unknown_slots


def _detect_moving_block(frame: np.ndarray, layout: Layout) -> dict[str, Any] | None:
    frame_rgb = frame[..., :3].astype(np.float32)
    palette = np.asarray(layout.palette_rgb, dtype=np.float32)
    if len(palette) == 0:
        return None

    distance_stack = np.stack(
        [np.linalg.norm(frame_rgb - color.reshape(1, 1, 3), axis=2) for color in palette],
        axis=2,
    )
    best_color = np.argmin(distance_stack, axis=2)
    best_distance = np.min(distance_stack, axis=2)
    color_mask = best_distance <= COLOR_DISTANCE_THRESHOLD
    candidate_mask = color_mask & ~layout.tube_body_union_mask
    if not np.any(candidate_mask):
        return None

    component = _largest_component(candidate_mask)
    pixel_count = int(component.sum())
    if pixel_count < MOVING_BLOCK_MIN_PIXELS or pixel_count > MOVING_BLOCK_MAX_PIXELS:
        return None

    ys, xs = np.where(component)
    dominant_colors = best_color[component]
    bincount = np.bincount(dominant_colors, minlength=len(layout.palette_rgb))
    color = int(np.argmax(bincount)) if bincount.sum() > 0 else None
    dominant_ratio = float(bincount[color] / max(pixel_count, 1)) if color is not None else 0.0
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    tube_centers = [geom["x"] + geom["width"] / 2 for geom in layout.tube_geometry]
    left_bound = min(tube_centers) - layout.tube_geometry[0]["width"] * 0.3
    right_bound = max(tube_centers) + layout.tube_geometry[0]["width"] * 0.3
    if color is None or dominant_ratio < MOVING_BLOCK_MIN_DOMINANT_RATIO:
        return None
    if center_x < left_bound or center_x > right_bound:
        return None
    return {
        "color": color,
        "pixel_count": pixel_count,
        "center": [center_x, center_y],
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    best_component = np.zeros_like(mask, dtype=bool)
    best_size = 0
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component_points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component_points.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        if len(component_points) > best_size:
            best_size = len(component_points)
            best_component = np.zeros_like(mask, dtype=bool)
            for y, x in component_points:
                best_component[y, x] = True
    return best_component


def _infer_move(previous: State, current: State, capacity: int) -> Move | None:
    source = None
    dest = None
    moved_blocks: tuple[int, ...] | None = None
    for idx, (left, right) in enumerate(zip(previous, current)):
        if left == right:
            continue
        if len(left) > len(right) and left[: len(right)] == right:
            source = idx
            moved_blocks = left[len(right) :]
        elif len(left) < len(right) and left == right[: len(left)]:
            dest = idx
            if moved_blocks is None:
                moved_blocks = right[len(left) :]
        else:
            return None
    if source is None or dest is None or moved_blocks is None or source == dest:
        return None
    moved_count = len(moved_blocks)
    if moved_count <= 0 or len(current[dest]) > capacity:
        return None
    moved_color = moved_blocks[0]
    if any(block != moved_color for block in moved_blocks):
        return None
    if previous[source][-moved_count:] != moved_blocks:
        return None
    if current[dest][-moved_count:] != moved_blocks:
        return None
    if previous[dest] and previous[dest][-1] != moved_color:
        return None
    source_run_length = _top_run_length(previous[source])
    if moved_count != min(source_run_length, capacity - len(previous[dest])):
        return None
    return (source, dest, moved_count)


def _is_state_stack_valid(state: State) -> bool:
    for tube in state:
        if len(tube) != len([block for block in tube if block is not None]):
            return False
    return True


def _solved_tube_count(state: State, capacity: int) -> int:
    return sum(1 for tube in state if len(tube) == capacity and len(set(tube)) == 1)


def _monochrome_tube_count(state: State) -> int:
    return sum(1 for tube in state if tube and len(set(tube)) == 1)


def _block_progress_ratio(state: State, capacity: int) -> float:
    total_blocks = sum(len(tube) for tube in state)
    if total_blocks == 0:
        return 1.0
    good_blocks = 0
    for tube in state:
        if not tube:
            continue
        runs: list[int] = [1]
        for left, right in zip(tube[:-1], tube[1:]):
            if left == right:
                runs[-1] += 1
            else:
                runs.append(1)
        good_blocks += max(runs)
        if len(tube) == capacity and len(set(tube)) == 1:
            good_blocks += 0
    return min(good_blocks / total_blocks, 1.0)


def _count_inventory(state: list[list[int]] | State) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tube in state:
        for block in tube:
            counts[int(block)] = counts.get(int(block), 0) + 1
    return counts


def _inventory_matches_with_motion_tolerance(
    state_counts: dict[int, int],
    expected_inventory: dict[int, int],
    moving_color: int | None,
    expected_block_count: int,
    tube_block_count: int,
) -> bool:
    if moving_color is None:
        return state_counts == expected_inventory and tube_block_count == expected_block_count

    if tube_block_count < expected_block_count - max(expected_inventory.values(), default=0) or tube_block_count > expected_block_count:
        return False

    color_keys = set(state_counts) | set(expected_inventory)
    off_color_keys = [key for key in color_keys if key != moving_color and state_counts.get(key, 0) != expected_inventory.get(key, 0)]
    if off_color_keys:
        return False

    moving_delta = state_counts.get(moving_color, 0) - expected_inventory.get(moving_color, 0)
    max_missing = expected_inventory.get(moving_color, 0)
    return moving_delta in range(-max_missing, 1)


def _state_to_lists(state: State | None) -> list[list[int]] | None:
    if state is None:
        return None
    return [list(tube) for tube in state]


def _top_run_length(tube: State | TubeState) -> int:
    if not tube:
        return 0
    top = tube[-1]
    run_length = 0
    for block in reversed(tube):
        if block != top:
            break
        run_length += 1
    return run_length


def _fill_motion_gaps(frame_states: list[dict[str, Any]], radius: int = 2) -> None:
    motion_indices = [index for index, frame in enumerate(frame_states) if frame["in_motion"]]
    if not motion_indices:
        return
    for index, frame in enumerate(frame_states):
        if frame["resolved"] or frame["in_motion"]:
            continue
        if any(abs(index - motion_index) <= radius for motion_index in motion_indices):
            frame["in_motion"] = True


def _build_resolved_state_sequence(frame_states: list[dict[str, Any]], capacity: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for frame in frame_states:
        if not frame["resolved"] or frame["tube_state"] is None:
            continue
        state = tuple(tuple(tube) for tube in frame["tube_state"])
        if runs and runs[-1]["state"] == state:
            runs[-1]["end_frame"] = frame["frame_index"]
            runs[-1]["length"] += 1
            continue
        runs.append(
            {
                "state": state,
                "start_frame": frame["frame_index"],
                "end_frame": frame["frame_index"],
                "length": 1,
            }
        )

    filtered: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if 0 < index < len(runs) - 1 and run["length"] == 1:
            previous_state = runs[index - 1]["state"]
            next_state = runs[index + 1]["state"]
            if _infer_move(previous_state, next_state, capacity) is not None:
                continue
        filtered.append(run)

    resolved_states: list[dict[str, Any]] = []
    previous_state: State | None = None
    previous_frame = -1
    for run in filtered:
        move = None if previous_state is None else _infer_move(previous_state, run["state"], capacity)
        resolved_states.append(
            {
                "frame_index": run["start_frame"],
                "state": _state_to_lists(run["state"]),
                "move_from_previous": list(move) if move is not None else None,
                "frame_gap": run["start_frame"] - previous_frame if previous_frame >= 0 else 0,
                "run_length": run["length"],
            }
        )
        previous_state = run["state"]
        previous_frame = run["start_frame"]
    return resolved_states


def _render_style_defaults() -> dict[str, Any]:
    return {
        "margin_x": 14,
        "margin_y": 28,
        "tube_gap_x": 14,
        "tube_gap_y": 22,
        "tube_width": 34,
        "tube_height": 126,
        "tube_height_two_row": 68,
        "two_row_carry_reserve": 28,
        "tube_wall": 4,
        "tube_radius": 10,
        "tube_inner": "#242834",
        "tube_inner_padding": 8,
        "block_gap": 6,
        "block_side_margin": 7,
    }


def _tube_layout(
    width: int,
    height: int,
    tube_count: int,
    layout_variant: str,
    style: dict[str, Any],
) -> list[dict[str, float]]:
    margin_x = float(style["margin_x"])
    margin_y = float(style["margin_y"])
    if layout_variant == "two_row":
        cols = int(np.ceil(tube_count / 2))
        rows = 2
    else:
        cols = tube_count
        rows = 1
    gap_x = float(style["tube_gap_x"])
    gap_y = float(style["tube_gap_y"])
    tube_width = min(float(style["tube_width"]), (width - 2 * margin_x - (cols - 1) * gap_x) / max(cols, 1))
    default_tube_height = float(style["tube_height_two_row"] if rows == 2 else style["tube_height"])
    top_reserve = float(style["two_row_carry_reserve"]) if rows == 2 else 0.0
    tube_height = min(
        default_tube_height,
        (height - 2 * margin_y - (rows - 1) * gap_y - top_reserve) / max(rows, 1),
    )
    total_width = cols * tube_width + (cols - 1) * gap_x
    total_height = rows * tube_height + (rows - 1) * gap_y + top_reserve
    origin_x = (width - total_width) / 2
    origin_y = (height - total_height) / 2 + top_reserve
    layout: list[dict[str, float]] = []
    for index in range(tube_count):
        row = index // cols
        col = index % cols
        x = origin_x + col * (tube_width + gap_x)
        y = origin_y + row * (tube_height + gap_y)
        layout.append({"x": x, "y": y, "width": tube_width, "height": tube_height})
    return layout


def _build_tube_body_mask(shape: tuple[int, int], geom: dict[str, float]) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            int(np.floor(geom["x"])),
            int(np.floor(geom["y"])),
            int(np.ceil(geom["x"] + geom["width"])),
            int(np.ceil(geom["y"] + geom["height"])),
        ),
        fill=1,
    )
    return np.asarray(image, dtype=bool)


def _build_rect_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    width: int,
    height: int,
) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    cx, cy = center
    draw.rectangle(
        (
            int(round(cx - width / 2)),
            int(round(cy - height / 2)),
            int(round(cx + width / 2)),
            int(round(cy + height / 2)),
        ),
        fill=1,
    )
    return np.asarray(image, dtype=bool)


def _block_center(geom: dict[str, float], block_idx: int, capacity: int) -> tuple[float, float]:
    padding = 8.0
    gap = 6.0
    block_height = _block_height(geom["height"], capacity)
    bottom = geom["y"] + geom["height"] - padding - block_height / 2
    return (geom["x"] + geom["width"] / 2, bottom - block_idx * (block_height + gap))


def _block_width(tube_width: float) -> float:
    return max(10.0, tube_width - 14.0)


def _block_height(tube_height: float, capacity: int) -> float:
    usable = tube_height - 16.0 - 6.0 * (max(capacity, 1) - 1)
    return max(2.0, usable / max(capacity, 1))


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    normalized = value.lstrip("#")
    return tuple(int(normalized[index:index + 2], 16) for index in (0, 2, 4))
