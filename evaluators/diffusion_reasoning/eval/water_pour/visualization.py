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
            panel_h=gt_height,
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
        f"bucket={metadata['bucket']}  split={metadata['topology_split']}"
    )
    diag_text = (
        f"illegal={len(strict_result['illegal_transitions'])}  "
        f"cons={strict_result['conservation_violations']}  "
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
    panel_h: int,
    frame_state: dict[str, Any],
    metadata: dict[str, Any],
    font: ImageFont.ImageFont,
) -> None:
    lines = [
        f"frame={frame_state['frame_index'] + 1}",
        f"resolved={'yes' if frame_state['resolved'] else 'no'}  unknown={frame_state['unknown_ratio']:.2f}",
        (
            f"cons={'ok' if frame_state['conservation_ok'] else 'bad'}  "
            f"place={'ok' if frame_state['placement_ok'] else 'bad'}  "
            f"cap={'ok' if frame_state['capacity_ok'] else 'bad'}"
        ),
        (
            f"solved={frame_state['solved_tube_count']}/{metadata['puzzle']['tube_count']}  "
            f"mono={frame_state['monochrome_tube_count']}/{metadata['puzzle']['tube_count']}"
        ),
    ]
    if frame_state["resolved_move"] is not None:
        lines.append(f"move={tuple(frame_state['resolved_move'])}")
    elif frame_state["moving_block"] is not None:
        lines.append(f"airborne=color{frame_state['moving_block']['color']}")
    else:
        lines.append("move=none")

    base_x = panel_x + 8
    base_y = panel_y + 10
    for offset, line in enumerate(lines):
        draw.text((base_x, base_y + offset * 14), line, fill=TEXT, font=font)

    state = frame_state["tube_state"]
    if state is None:
        return
    _draw_symbolic_state(
        draw=draw,
        state=state,
        metadata=metadata,
        x=panel_x + panel_w - 88,
        y=panel_y + 8,
        width=78,
        height=min(110, panel_h - 16),
    )


def _draw_symbolic_state(
    draw: ImageDraw.ImageDraw,
    state: list[list[int]],
    metadata: dict[str, Any],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    palette = [_hex_to_rgb(value) for value in metadata["render"]["palette"]]
    tube_count = int(metadata["puzzle"]["tube_count"])
    capacity = int(metadata["puzzle"]["capacity"])
    gap = 4
    tube_width = max(6, int((width - gap * (tube_count - 1)) / max(tube_count, 1)))
    slot_height = max(6, int(height / max(capacity, 1)))
    draw.rectangle((x - 4, y - 4, x + width + 4, y + height + 4), outline=(80, 88, 98), width=1)
    for tube_idx, tube in enumerate(state):
        tx = x + tube_idx * (tube_width + gap)
        for slot_idx in range(capacity):
            top = y + height - (slot_idx + 1) * slot_height
            bottom = top + slot_height - 1
            fill = (29, 34, 41)
            if slot_idx < len(tube):
                fill = palette[int(tube[slot_idx]) % len(palette)]
            draw.rectangle((tx, top, tx + tube_width - 1, bottom), fill=fill, outline=(60, 68, 78))


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    normalized = value.lstrip("#")
    return tuple(int(normalized[index:index + 2], 16) for index in (0, 2, 4))
