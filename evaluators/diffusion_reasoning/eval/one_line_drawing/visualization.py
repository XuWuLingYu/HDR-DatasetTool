from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluators.diffusion_reasoning.utils.video_io import write_video_frames

BANNER_HEIGHT = 52
PANEL_GAP = 8
BG_COLOR = (24, 29, 34)
TEXT_COLOR = (242, 245, 248)
MUTED_TEXT = (186, 193, 201)
PASS_COLOR = (75, 187, 96)
FAIL_COLOR = (208, 78, 72)
PANEL_LABEL_BG = (38, 46, 55)


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

    fps = int(metadata["render"]["fps"])
    target_count = int(metadata["solution"]["cell_count"])
    font = ImageFont.load_default()
    width = gt_frames[0].shape[1] + pred_frames[0].shape[1] + PANEL_GAP * 3
    height = max(gt_frames[0].shape[0], pred_frames[0].shape[0]) + BANNER_HEIGHT + PANEL_GAP * 2

    rendered_frames: list[np.ndarray] = []
    for frame_index in range(total_frames):
        gt_frame = gt_frames[frame_index]
        pred_frame = pred_frames[frame_index]
        frame_state = state_trace["frame_states"][frame_index]
        canvas = Image.new("RGB", (width, height), BG_COLOR)
        draw = ImageDraw.Draw(canvas)

        _draw_banner(draw, width, strict_result, relaxed_result, metadata, font)
        _paste_panel(canvas, gt_frame, PANEL_GAP, BANNER_HEIGHT + PANEL_GAP)
        _paste_panel(canvas, pred_frame, gt_frame.shape[1] + PANEL_GAP * 2, BANNER_HEIGHT + PANEL_GAP)
        _draw_panel_label(draw, PANEL_GAP, BANNER_HEIGHT + 4, "GT", font)
        _draw_panel_label(draw, gt_frame.shape[1] + PANEL_GAP * 2, BANNER_HEIGHT + 4, "PRED", font)
        _draw_pred_overlay(
            draw,
            panel_x=gt_frame.shape[1] + PANEL_GAP * 2,
            panel_y=BANNER_HEIGHT + PANEL_GAP,
            frame_state=frame_state,
            target_count=target_count,
            frame_index=frame_index,
            total_frames=total_frames,
            font=font,
        )
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
    sample_text = f"{metadata['sample_id']}  level={metadata['difficulty_level']}  family={metadata['family']}"
    strict_text = "Strict: PASS" if strict_result["passed"] else "Strict: FAIL"
    relaxed_text = f"Relaxed: {relaxed_result['score']:.3f}"
    diag_text = (
        f"missing={len(strict_result['missing_cells'])}  "
        f"illegal={len(strict_result['illegal_transitions'])}  "
        f"repeat={len(strict_result['repeated_cells'])}"
    )
    strict_color = PASS_COLOR if strict_result["passed"] else FAIL_COLOR

    draw.text((10, 7), sample_text, fill=TEXT_COLOR, font=font)
    draw.text((10, 28), strict_text, fill=strict_color, font=font)
    draw.text((120, 28), relaxed_text, fill=TEXT_COLOR, font=font)
    draw.text((250, 28), diag_text, fill=MUTED_TEXT, font=font)


def _paste_panel(canvas: Image.Image, frame: np.ndarray, x: int, y: int) -> None:
    panel = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    canvas.paste(panel, (x, y))


def _draw_panel_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=PANEL_LABEL_BG)
    draw.text((x, y), label, fill=TEXT_COLOR, font=font)


def _draw_pred_overlay(
    draw: ImageDraw.ImageDraw,
    panel_x: int,
    panel_y: int,
    frame_state: dict[str, Any],
    target_count: int,
    frame_index: int,
    total_frames: int,
    font: ImageFont.ImageFont,
) -> None:
    head_cell = frame_state["head_cell"]
    head_text = "head=none" if head_cell is None else f"head={tuple(head_cell)}"
    visited_count = len(frame_state["visited_cells"])
    on_shape = "on_shape=yes" if frame_state["on_shape"] else "on_shape=no"
    ratio_text = f"trace={frame_state['trace_on_shape_ratio']:.2f}"
    progress_text = f"visited={visited_count}/{target_count}  frame={frame_index + 1}/{total_frames}"
    base_x = panel_x + 8
    base_y = panel_y + 16
    draw.text((base_x, base_y), head_text, fill=TEXT_COLOR, font=font)
    draw.text((base_x, base_y + 16), progress_text, fill=TEXT_COLOR, font=font)
    draw.text((base_x, base_y + 32), f"{on_shape}  {ratio_text}", fill=TEXT_COLOR, font=font)
