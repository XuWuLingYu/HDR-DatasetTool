from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluators.diffusion_reasoning.utils.video_io import write_video_frames

BANNER_HEIGHT = 56
PANEL_GAP = 8
STATUS_BG = (24, 29, 34)
TEXT = (241, 244, 247)
MUTED = (182, 189, 198)
PASS = (75, 187, 96)
FAIL = (208, 78, 72)
LABEL_BG = (38, 46, 55)


def render_evaluation_video(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    state_trace: dict[str, Any],
    strict_result: dict[str, Any],
    relaxed_result: dict[str, Any],
    metadata: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    total_frames = min(len(gt_frames), len(pred_frames), int(state_trace["frame_count"]))
    if total_frames <= 0:
        raise ValueError("Cannot render evaluation video without overlapping frames.")

    font = ImageFont.load_default()
    gt_height, gt_width = gt_frames[0].shape[0], gt_frames[0].shape[1]
    width = gt_width * 2 + PANEL_GAP * 3
    height = gt_height + BANNER_HEIGHT + PANEL_GAP * 2
    fps = int(metadata["render"]["fps"])

    rendered_frames: list[np.ndarray] = []
    for frame_index in range(total_frames):
        frame_state = state_trace["frame_states"][frame_index]
        canvas = Image.new("RGB", (width, height), STATUS_BG)
        draw = ImageDraw.Draw(canvas)
        _draw_banner(draw, width, strict_result, relaxed_result, metadata, font)
        _paste_panel(canvas, gt_frames[frame_index], PANEL_GAP, BANNER_HEIGHT + PANEL_GAP)
        _paste_panel(canvas, pred_frames[frame_index], gt_width + PANEL_GAP * 2, BANNER_HEIGHT + PANEL_GAP)
        _draw_label(draw, PANEL_GAP, BANNER_HEIGHT + 4, "GT", font)
        _draw_label(draw, gt_width + PANEL_GAP * 2, BANNER_HEIGHT + 4, "PRED", font)
        _draw_overlay(draw, gt_width + PANEL_GAP * 2, BANNER_HEIGHT + PANEL_GAP, gt_width, frame_state, state_trace, font)
        rendered_frames.append(np.asarray(canvas, dtype=np.uint8))

    write_video_frames(output_path, rendered_frames, fps=fps)
    return {
        "path": str(output_path),
        "rendered_frames": total_frames,
        "gt_frame_count": len(gt_frames),
        "pred_frame_count": len(pred_frames),
        "frame_count_mismatch": len(gt_frames) != len(pred_frames),
    }


def _draw_banner(
    draw: ImageDraw.ImageDraw,
    width: int,
    strict_result: dict[str, Any],
    relaxed_result: dict[str, Any],
    metadata: dict[str, Any],
    font: ImageFont.ImageFont,
) -> None:
    draw.rectangle((0, 0, width, BANNER_HEIGHT), fill=(31, 37, 44))
    strict_text = "Strict: PASS" if strict_result["passed"] else "Strict: FAIL"
    strict_color = PASS if strict_result["passed"] else FAIL
    sample_text = (
        f"{metadata['sample_id']}  level={metadata['difficulty_level']}  "
        f"bucket={metadata['bucket']}  split={metadata['topology_split']}"
    )
    diag_text = (
        f"illegal={len(strict_result['illegal_transitions'])}  "
        f"invalid={strict_result['entity_validity_violations']}  "
        f"unresolved={strict_result['unresolved_frames']}"
    )
    draw.text((10, 7), sample_text, fill=TEXT, font=font)
    draw.text((10, 30), strict_text, fill=strict_color, font=font)
    draw.text((120, 30), f"Relaxed: {relaxed_result['score']:.3f}", fill=TEXT, font=font)
    draw.text((250, 30), diag_text, fill=MUTED, font=font)


def _paste_panel(canvas: Image.Image, frame: np.ndarray, x: int, y: int) -> None:
    canvas.paste(Image.fromarray(frame.astype(np.uint8), mode="RGB"), (x, y))


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=LABEL_BG)
    draw.text((x, y), label, fill=TEXT, font=font)


def _draw_overlay(
    draw: ImageDraw.ImageDraw,
    panel_x: int,
    panel_y: int,
    panel_w: int,
    frame_state: dict[str, Any],
    state_trace: dict[str, Any],
    font: ImageFont.ImageFont,
) -> None:
    lines = [
        f"frame={frame_state['frame_index'] + 1}",
        f"resolved={'yes' if frame_state['resolved'] else 'no'}  moving={'yes' if frame_state['in_motion'] else 'no'}",
        f"player={_fmt_cell(frame_state['player_cell'])}  box={_fmt_cell(frame_state['box_cell'])}",
        (
            f"on_floor={'ok' if frame_state['player_on_floor'] and frame_state['box_on_floor'] else 'bad'}  "
            f"overlap={'yes' if frame_state['overlap_detected'] else 'no'}"
        ),
        (
            f"on_target={'yes' if frame_state['box_on_target'] else 'no'}  "
            f"dist={frame_state['box_target_distance']}"
        ),
        (
            f"obs={state_trace['trace_observability_ratio']:.2f}  "
            f"rule={state_trace['trace_rule_adherence_ratio']:.2f}"
        ),
    ]

    base_x = panel_x + 8
    base_y = panel_y + 10
    for offset, line in enumerate(lines):
        draw.text((base_x, base_y + offset * 14), line, fill=TEXT, font=font)

    symbolic = frame_state["state"]
    if symbolic is None:
        return
    _draw_symbolic_state(draw, symbolic, state_trace["layout"], panel_x + panel_w - 84, panel_y + 8)


def _draw_symbolic_state(
    draw: ImageDraw.ImageDraw,
    state: dict[str, list[int]],
    layout: dict[str, Any],
    x: int,
    y: int,
) -> None:
    board_size = int(layout["board_size"])
    walls = {tuple(cell) for cell in layout["wall_cells"]}
    target = tuple(layout["target"])
    player = tuple(state["player"])
    box = tuple(state["box"])
    gap = 2
    cell_size = 8
    size = board_size * cell_size + (board_size - 1) * gap
    draw.rectangle((x - 4, y - 4, x + size + 4, y + size + 4), outline=(80, 88, 98), width=1)
    for row in range(board_size):
        for col in range(board_size):
            x0 = x + col * (cell_size + gap)
            y0 = y + row * (cell_size + gap)
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            cell = (row, col)
            fill = (22, 28, 36)
            if cell in walls:
                fill = (124, 134, 164)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(45, 49, 58))
            if cell == target:
                draw.ellipse((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=(130, 225, 127), width=1)
            if cell == box:
                draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=(244, 183, 78))
            if cell == player:
                draw.polygon(((x0 + cell_size / 2, y0 + 2), (x1 - 2, y1 - 2), (x0 + 2, y1 - 2)), fill=(255, 92, 122))


def _fmt_cell(cell: list[int] | None) -> str:
    if cell is None:
        return "none"
    return f"({cell[0]},{cell[1]})"
