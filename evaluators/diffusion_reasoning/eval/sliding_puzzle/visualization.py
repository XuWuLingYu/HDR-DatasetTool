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
        _draw_overlay(
            draw=draw,
            panel_x=gt_width + PANEL_GAP * 2,
            panel_y=BANNER_HEIGHT + PANEL_GAP,
            panel_w=gt_width,
            frame_state=frame_state,
            metadata=metadata,
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
    strict_text = "Strict: PASS" if strict_result["passed"] else "Strict: FAIL"
    strict_color = PASS if strict_result["passed"] else FAIL
    sample_text = (
        f"{metadata['sample_id']}  level={metadata['difficulty_level']}  "
        f"bucket={metadata['bucket']}  size={metadata['puzzle']['board_size'][0]}x{metadata['puzzle']['board_size'][1]}"
    )
    diag_text = (
        f"illegal={len(strict_result['illegal_transitions'])}  "
        f"inventory={strict_result['inventory_violations']}  "
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
    metadata: dict[str, Any],
    font: ImageFont.ImageFont,
) -> None:
    lines = [
        f"frame={frame_state['frame_index'] + 1}",
        f"resolved={'yes' if frame_state['resolved'] else 'no'}  moving={'yes' if frame_state['in_motion'] else 'no'}",
        (
            f"tiles={'ok' if frame_state['tile_count_match'] else 'bad'}  "
            f"dup={'yes' if frame_state['duplicate_tile_detected'] else 'no'}  "
            f"blank={'ok' if frame_state['blank_count_ok'] else 'bad'}"
        ),
        (
            f"manhattan={frame_state['manhattan_to_goal']}  "
            f"correct={frame_state['correct_tile_ratio']:.2f}"
        ),
    ]
    base_x = panel_x + 8
    base_y = panel_y + 10
    for offset, line in enumerate(lines):
        draw.text((base_x, base_y + offset * 14), line, fill=TEXT, font=font)

    state = frame_state["decoded_state"]
    if state is None:
        return
    _draw_symbolic_board(
        draw=draw,
        state=state,
        board_size=int(metadata["puzzle"]["board_size"][0]),
        x=panel_x + panel_w - 92,
        y=panel_y + 8,
        size=84,
        font=font,
    )


def _draw_symbolic_board(
    draw: ImageDraw.ImageDraw,
    state: list[int],
    board_size: int,
    x: int,
    y: int,
    size: int,
    font: ImageFont.ImageFont,
) -> None:
    gap = 3
    tile_size = max(10, int((size - (board_size - 1) * gap) / max(board_size, 1)))
    draw.rectangle((x - 4, y - 4, x + size + 4, y + size + 4), outline=(80, 88, 98), width=1)
    for index, tile in enumerate(state):
        row, col = divmod(index, board_size)
        x0 = x + col * (tile_size + gap)
        y0 = y + row * (tile_size + gap)
        x1 = x0 + tile_size
        y1 = y0 + tile_size
        fill = (67, 72, 92) if tile == 0 else (48, 66, 109)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=fill)
        if tile != 0:
            text = str(tile)
            bbox = draw.textbbox((0, 0), text, font=font)
            tx = x0 + (tile_size - (bbox[2] - bbox[0])) / 2
            ty = y0 + (tile_size - (bbox[3] - bbox[1])) / 2
            draw.text((tx, ty), text, fill=TEXT, font=font)
