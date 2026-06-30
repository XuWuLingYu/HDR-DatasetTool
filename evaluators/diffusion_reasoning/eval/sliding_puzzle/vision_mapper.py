from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluators.diffusion_reasoning.utils.video_io import load_video_frames

DEFAULT_COLORS = {
    "background": (14, 17, 24),
    "panel": (37, 39, 43),
    "slot": (67, 72, 92),
    "tile": (48, 66, 109),
    "tile_border": (184, 209, 255),
    "text": (241, 244, 251),
}
MAX_TEMPLATE_ERROR = 0.20
MIN_TEMPLATE_MARGIN = 0.015


@dataclass(frozen=True)
class Layout:
    board_size: int
    shape: tuple[int, int]
    tile_rects: list[tuple[int, int, int, int]]
    blank_template: np.ndarray
    tile_template: np.ndarray
    templates: list[np.ndarray]
    glyph_templates: list[np.ndarray]
    template_shape: tuple[int, int]
    uses_gt_templates: bool
    expected_states: list[tuple[int, ...]]
    state_slot_templates: list[list[np.ndarray]]


def extract_state_trace(
    video_path: str | Path,
    metadata: dict[str, Any],
    *,
    gt_video_path: str | Path | None = None,
) -> dict[str, Any]:
    frames = load_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    gt_frames = None
    if gt_video_path is not None:
        gt_frames = load_video_frames(gt_video_path)
    layout = _build_layout(metadata, gt_frames=gt_frames)
    goal_state = flatten_board(metadata["puzzle"]["goal_board"])

    frame_states: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        if layout.uses_gt_templates:
            decoded_labels, errors, margins, expected_state_index = _decode_frame_with_expected_states(frame, layout)
        else:
            decoded_labels, errors, margins = _decode_frame(frame, layout)
            expected_state_index = None
        if _is_frame_resolved(decoded_labels, errors, margins, layout):
            decoded_state = tuple(int(value) for value in decoded_labels)
            duplicate_tile_detected = len(set(decoded_state)) != len(decoded_state)
            blank_count_ok = decoded_state.count(0) == 1
            expected_tiles = set(range(layout.board_size * layout.board_size))
            tile_count_match = set(decoded_state) == expected_tiles
            resolved = tile_count_match and blank_count_ok and not duplicate_tile_detected
        else:
            decoded_state = None
            duplicate_tile_detected = True
            blank_count_ok = False
            tile_count_match = False
            resolved = False

        if decoded_state is None:
            current_distance = None
            current_correct_ratio = 0.0
        else:
            current_distance = manhattan_distance(decoded_state, layout.board_size)
            current_correct_ratio = _correct_tile_ratio(decoded_state, goal_state)

        frame_states.append(
            {
                "frame_index": frame_index,
                "decoded_labels": list(decoded_labels),
                "decoded_state": list(decoded_state) if decoded_state is not None else None,
                "resolved": resolved,
                "in_motion": not resolved,
                "tile_errors": [float(value) for value in errors],
                "tile_margins": [float(value) for value in margins],
                "tile_count_match": tile_count_match,
                "duplicate_tile_detected": duplicate_tile_detected,
                "blank_count_ok": blank_count_ok,
                "manhattan_to_goal": current_distance,
                "correct_tile_ratio": current_correct_ratio,
                "expected_state_index": expected_state_index,
            }
        )

    resolved_states = _build_resolved_state_sequence(frame_states)
    return {
        "video_path": str(video_path),
        "frame_count": len(frames),
        "layout": {
            "board_size": layout.board_size,
            "tile_rects": [list(rect) for rect in layout.tile_rects],
        },
        "frame_states": frame_states,
        "resolved_states": resolved_states,
    }


def compute_board_size(metadata: dict[str, Any]) -> int:
    return int(metadata["puzzle"]["board_size"][0])


def flatten_board(board: list[list[int]]) -> tuple[int, ...]:
    return tuple(int(value) for row in board for value in row)


def manhattan_distance(state: tuple[int, ...], board_size: int) -> int:
    distance = 0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        row, col = divmod(index, board_size)
        goal_row, goal_col = divmod(tile - 1, board_size)
        distance += abs(row - goal_row) + abs(col - goal_col)
    return distance


def _build_layout(metadata: dict[str, Any], gt_frames: list[np.ndarray] | None = None) -> Layout:
    board_size = compute_board_size(metadata)
    width, height = metadata["render"]["resolution"]
    geometry = _board_geometry(width, height, board_size)
    tile_rects = [_tile_rect(index, board_size, geometry) for index in range(board_size * board_size)]
    expected_states = _expected_state_sequence(metadata)
    uses_gt_templates = bool(gt_frames)
    if not gt_frames:
        templates = [_render_slot_template(label, board_size, geometry) for label in range(board_size * board_size)]
        blank_template = templates[0]
        state_slot_templates: list[list[np.ndarray]] = []
    else:
        templates = _extract_templates_from_gt(gt_frames[0], metadata, tile_rects)
        blank_template = _render_slot_template(0, board_size, geometry)
        state_slot_templates = _extract_state_slot_templates(gt_frames, metadata, tile_rects, expected_states)
    tile_template = np.mean(np.stack(templates[1:], axis=0), axis=0) if len(templates) > 1 else templates[0]
    return Layout(
        board_size=board_size,
        shape=(height, width),
        tile_rects=tile_rects,
        blank_template=blank_template,
        tile_template=tile_template,
        templates=templates,
        glyph_templates=[_render_glyph_template(label, board_size, geometry) for label in range(board_size * board_size)],
        template_shape=templates[0].shape[:2],
        uses_gt_templates=uses_gt_templates,
        expected_states=expected_states,
        state_slot_templates=state_slot_templates,
    )


def _decode_frame(frame: np.ndarray, layout: Layout) -> tuple[list[int], list[float], list[float]]:
    labels: list[int] = []
    errors: list[float] = []
    margins: list[float] = []
    for rect in layout.tile_rects:
        patch = _crop_patch(frame, rect)
        if layout.uses_gt_templates:
            raise RuntimeError("direct slot decoding should not be used with GT templates")

        blank_error = _normalized_error(patch, layout.blank_template)
        tile_error = _normalized_error(patch, layout.tile_template)
        if blank_error + 0.01 < tile_error:
            labels.append(0)
            errors.append(blank_error)
            margins.append(tile_error - blank_error)
            continue

        glyph_mask = _extract_glyph_mask(patch)
        patch_errors = [_glyph_error(glyph_mask, template) for template in layout.glyph_templates[1:]]
        order = np.argsort(patch_errors)
        best_offset = int(order[0])
        best_label = best_offset + 1
        second = float(patch_errors[int(order[1])]) if len(order) > 1 else 1.0
        labels.append(best_label)
        errors.append(float(patch_errors[best_offset]))
        margins.append(second - float(patch_errors[best_offset]))
    return labels, errors, margins


def _decode_frame_with_expected_states(
    frame: np.ndarray,
    layout: Layout,
) -> tuple[list[int], list[float], list[float], int | None]:
    patches = [_crop_patch(frame, rect) for rect in layout.tile_rects]
    best_state: tuple[int, ...] | None = None
    best_errors: list[float] = []
    best_total = None
    second_total = None
    best_index = None
    for state_index, state in enumerate(layout.expected_states):
        if state_index < len(layout.state_slot_templates):
            slot_templates = layout.state_slot_templates[state_index]
            per_slot = [_normalized_error(patch, template) for patch, template in zip(patches, slot_templates)]
        else:
            per_slot = [_normalized_error(patch, layout.templates[label]) for patch, label in zip(patches, state)]
        total = float(sum(per_slot) / max(len(per_slot), 1))
        if best_total is None or total < best_total:
            second_total = best_total
            best_total = total
            best_state = state
            best_errors = per_slot
            best_index = state_index
        elif second_total is None or total < second_total:
            second_total = total
    if best_state is None:
        return [], [], [], None
    margin = 0.0 if second_total is None else second_total - float(best_total)
    return list(best_state), [float(value) for value in best_errors], [margin] * len(best_state), best_index


def _is_frame_resolved(labels: list[int], errors: list[float], margins: list[float], layout: Layout) -> bool:
    if len(labels) != layout.board_size * layout.board_size:
        return False
    if layout.uses_gt_templates:
        if not errors:
            return False
        return (sum(errors) / len(errors)) <= 0.145
    if any(error > MAX_TEMPLATE_ERROR for error in errors):
        return False
    if any(margin < MIN_TEMPLATE_MARGIN for margin in margins):
        return False
    if labels.count(0) != 1:
        return False
    expected = set(range(layout.board_size * layout.board_size))
    return set(labels) == expected


def _crop_patch(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    patch = frame[y0:y1, x0:x1, :3]
    return patch.astype(np.float32) / 255.0


def _normalized_error(patch: np.ndarray, template: np.ndarray) -> float:
    if patch.shape != template.shape:
        patch = _resize_patch(patch, template.shape[:2])
    return float(np.mean(np.abs(patch - template)))


def _glyph_error(patch_mask: np.ndarray, template_mask: np.ndarray) -> float:
    if patch_mask.shape != template_mask.shape:
        image = Image.fromarray((patch_mask.astype(np.uint8) * 255), mode="L")
        image = image.resize((template_mask.shape[1], template_mask.shape[0]), Image.Resampling.NEAREST)
        patch_mask = np.asarray(image, dtype=np.uint8) > 0
    return float(np.mean(np.logical_xor(patch_mask, template_mask)))


def _resize_patch(patch: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if patch.shape[:2] == target_shape:
        return patch
    image = Image.fromarray(np.clip(patch * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    resized = image.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _build_resolved_state_sequence(frame_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved_states: list[dict[str, Any]] = []
    previous = None
    previous_state_index = -1
    for frame in frame_states:
        state = frame["decoded_state"]
        if state is None:
            continue
        state_tuple = tuple(int(value) for value in state)
        state_index = frame.get("expected_state_index")
        if state_index is not None and int(state_index) < previous_state_index:
            continue
        if previous == state_tuple:
            continue
        resolved_states.append({"frame_index": int(frame["frame_index"]), "state": list(state_tuple)})
        previous = state_tuple
        if state_index is not None:
            previous_state_index = int(state_index)
    return resolved_states


def _correct_tile_ratio(state: tuple[int, ...], goal_state: tuple[int, ...]) -> float:
    total = max(sum(1 for value in goal_state if value != 0), 1)
    correct = sum(1 for left, right in zip(state, goal_state) if left == right and right != 0)
    return correct / total


def _expected_state_sequence(metadata: dict[str, Any]) -> list[tuple[int, ...]]:
    board_size = compute_board_size(metadata)
    current = list(flatten_board(metadata["puzzle"]["initial_board"]))
    states = [tuple(current)]
    for move in metadata["solution"]["moves"]:
        from_row, from_col = move["from"]
        to_row, to_col = move["to"]
        from_index = int(from_row) * board_size + int(from_col)
        to_index = int(to_row) * board_size + int(to_col)
        current[from_index], current[to_index] = current[to_index], current[from_index]
        states.append(tuple(current))
    return states


def _extract_templates_from_gt(
    gt_frame: np.ndarray,
    metadata: dict[str, Any],
    tile_rects: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    initial_state = flatten_board(metadata["puzzle"]["initial_board"])
    base_shape = _crop_patch(gt_frame, tile_rects[0]).shape[:2]
    templates: list[np.ndarray] = [np.zeros((base_shape[0], base_shape[1], 3), dtype=np.float32) for _ in range(len(tile_rects))]
    for index, label in enumerate(initial_state):
        patch = _crop_patch(gt_frame, tile_rects[index])
        templates[int(label)] = _resize_patch(patch, base_shape)
    return templates


def _extract_state_slot_templates(
    gt_frames: list[np.ndarray],
    metadata: dict[str, Any],
    tile_rects: list[tuple[int, int, int, int]],
    expected_states: list[tuple[int, ...]],
) -> list[list[np.ndarray]]:
    representative_indices = _representative_gt_frame_indices(len(gt_frames), int(metadata["solution"]["move_count"]))
    base_shape = _crop_patch(gt_frames[0], tile_rects[0]).shape[:2]
    templates: list[list[np.ndarray]] = []
    for frame_index, _state in zip(representative_indices, expected_states):
        frame = gt_frames[min(max(frame_index, 0), len(gt_frames) - 1)]
        templates.append([_resize_patch(_crop_patch(frame, rect), base_shape) for rect in tile_rects])
    return templates


def _representative_gt_frame_indices(frame_count: int, move_count: int) -> list[int]:
    if move_count <= 0 or frame_count <= 1:
        return [0]
    start_hold = 2
    end_hold = 4
    per_move = max(1, (frame_count - start_hold - end_hold) // move_count)
    indices = [0]
    for step in range(1, move_count + 1):
        indices.append(min(frame_count - 1, start_hold + step * per_move - 1))
    return indices


def _board_geometry(width: int, height: int, board_size: int) -> dict[str, float]:
    outer_margin = 4.0 if board_size == 3 else 8.0
    panel_radius = 16.0
    slot_gap = 3.0 if board_size == 3 else 5.0
    inner_padding = 4.0 if board_size == 3 else 8.0
    panel_w = width - 2 * outer_margin
    panel_h = height - 2 * outer_margin
    available = min(panel_w - 2 * inner_padding, panel_h - 2 * inner_padding)
    tile_size = (available - (board_size - 1) * slot_gap) / max(board_size, 1)
    board_extent = board_size * tile_size + (board_size - 1) * slot_gap
    origin_x = (width - board_extent) / 2
    origin_y = (height - board_extent) / 2
    return {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "tile_size": tile_size,
        "slot_gap": slot_gap,
        "panel_radius": panel_radius,
    }


def _tile_rect(index: int, board_size: int, geometry: dict[str, float]) -> tuple[int, int, int, int]:
    row, col = divmod(index, board_size)
    tile_size = geometry["tile_size"]
    slot_gap = geometry["slot_gap"]
    x0 = geometry["origin_x"] + col * (tile_size + slot_gap)
    y0 = geometry["origin_y"] + row * (tile_size + slot_gap)
    return (
        int(round(x0)),
        int(round(y0)),
        int(round(x0 + tile_size)),
        int(round(y0 + tile_size)),
    )


def _render_slot_template(label: int, board_size: int, geometry: dict[str, float]) -> np.ndarray:
    tile_size = max(1, int(round(geometry["tile_size"])))
    rect = (0, 0, tile_size - 1, tile_size - 1)
    radius = max(6, int(tile_size * 0.12))
    image = Image.new("RGB", (tile_size, tile_size), DEFAULT_COLORS["slot"])
    draw = ImageDraw.Draw(image)
    if label == 0:
        draw.rounded_rectangle(rect, radius=radius, fill=DEFAULT_COLORS["slot"])
    else:
        draw.rounded_rectangle(
            rect,
            radius=radius,
            fill=DEFAULT_COLORS["tile"],
            outline=DEFAULT_COLORS["tile_border"],
            width=2,
        )
        if label > 0:
            glyph = _render_scaled_glyph(str(label), tile_size, DEFAULT_COLORS["text"], board_size)
            glyph_x = int(round((tile_size - glyph.width) / 2))
            glyph_y = int(round((tile_size - glyph.height) / 2))
            image.paste(glyph, (glyph_x, glyph_y), glyph)
    return np.asarray(image, dtype=np.float32) / 255.0


def _render_glyph_template(label: int, board_size: int, geometry: dict[str, float]) -> np.ndarray:
    tile_size = max(1, int(round(geometry["tile_size"])))
    image = Image.new("L", (tile_size, tile_size), 0)
    if label == 0:
        return np.asarray(image, dtype=np.uint8) > 0
    glyph = _render_scaled_glyph(str(label), tile_size, (255, 255, 255), board_size)
    glyph_x = int(round((tile_size - glyph.width) / 2))
    glyph_y = int(round((tile_size - glyph.height) / 2))
    alpha = glyph.getchannel("A")
    image.paste(alpha, (glyph_x, glyph_y), alpha)
    return np.asarray(image, dtype=np.uint8) >= 96


def _extract_glyph_mask(patch: np.ndarray) -> np.ndarray:
    image = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    if image.shape[0] > 12 and image.shape[1] > 12:
        margin_y = max(2, int(round(image.shape[0] * 0.12)))
        margin_x = max(2, int(round(image.shape[1] * 0.12)))
        image = image[margin_y:-margin_y, margin_x:-margin_x, :]
    mask = (
        (image[..., 0] >= 180)
        & (image[..., 1] >= 180)
        & (image[..., 2] >= 180)
    )
    return mask


def _render_scaled_glyph(
    text: str,
    tile_size: int,
    fill_color: tuple[int, int, int],
    board_size: int,
) -> Image.Image:
    target_h = max(1, int(round(tile_size * 0.8)))
    target_w_limit = max(1, int(round(tile_size * 0.8)))
    font = _pick_base_font(board_size, tile_size * 2)
    canvas_size = tile_size * 4
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    bbox = mask_draw.textbbox((0, 0), text, font=font, anchor="lt")
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    origin_x = (canvas_size - text_w) // 2 - bbox[0]
    origin_y = (canvas_size - text_h) // 2 - bbox[1]
    mask_draw.text((origin_x, origin_y), text, font=font, fill=255)
    crop_box = mask.getbbox()
    if crop_box is None:
        return Image.new("RGBA", (target_w_limit, target_h), fill_color + (0,))
    glyph_mask = mask.crop(crop_box)
    glyph_w, glyph_h = glyph_mask.size
    scaled_w = max(1, int(round(glyph_w * (target_h / max(glyph_h, 1)))))
    scaled_h = target_h
    if scaled_w > target_w_limit:
        scale = target_w_limit / max(scaled_w, 1)
        scaled_w = target_w_limit
        scaled_h = max(1, int(round(scaled_h * scale)))
    glyph_mask = glyph_mask.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    glyph_mask = glyph_mask.point(lambda value: 255 if value >= 96 else 0, mode="L")
    glyph = Image.new("RGBA", (scaled_w, scaled_h), fill_color + (0,))
    glyph.putalpha(glyph_mask)
    return glyph


def _pick_base_font(board_size: int, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = (
        "DejaVuSans-Bold.ttf",
        "DejaVuSansCondensed-Bold.ttf",
        "DejaVuSans.ttf",
    )
    if board_size >= 4:
        font_candidates = (
            "DejaVuSansCondensed-Bold.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
        )
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
