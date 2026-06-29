#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare an original drawing-copy episode against an exact replay.")
    parser.add_argument("source_hdf5", type=Path)
    parser.add_argument("replay_hdf5", type=Path)
    parser.add_argument("--max-sampled-frame-mean-abs-diff", type=float, default=1.0)
    parser.add_argument("--max-draw-canvas-dark-pixel-diff", type=int, default=64)
    parser.add_argument("--max-draw-segment-count-diff", type=int, default=1)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.source_hdf5, "r") as src, h5py.File(args.replay_hdf5, "r") as rep:
        actions_src = src["actions"][()]
        actions_rep = rep["actions"][()]
        require(actions_src.shape == actions_rep.shape, f"action shape mismatch {actions_src.shape} vs {actions_rep.shape}")
        max_action_diff = float(np.max(np.abs(actions_src - actions_rep))) if actions_src.size else 0.0
        require(max_action_diff == 0.0, f"actions differ: {max_action_diff}")

        phases_src = read_phases(src)
        phases_rep = read_phases(rep)
        require(phases_src == phases_rep, "phases differ")

        frames_src = src["frames"]
        frames_rep = rep["frames"]
        require(frames_src.shape == frames_rep.shape, f"frame shape mismatch {frames_src.shape} vs {frames_rep.shape}")
        idx = np.unique(np.linspace(0, frames_src.shape[0] - 1, min(args.samples, frames_src.shape[0])).round().astype(int))
        mean_diffs = []
        max_pixel_diffs = []
        for i in idx:
            diff = np.abs(frames_src[i].astype(np.int16) - frames_rep[i].astype(np.int16))
            mean_diffs.append(float(diff.mean()))
            max_pixel_diffs.append(int(diff.max()))
        max_sampled_mean = float(np.max(mean_diffs)) if mean_diffs else 0.0
        require(
            max_sampled_mean <= args.max_sampled_frame_mean_abs_diff,
            f"sampled frame mean abs diff too high: {max_sampled_mean}",
        )

        src_dark = int(src.attrs.get("draw_canvas_dark_pixels", 0))
        rep_dark = int(rep.attrs.get("draw_canvas_dark_pixels", 0))
        dark_diff = abs(src_dark - rep_dark)
        require(
            dark_diff <= args.max_draw_canvas_dark_pixel_diff,
            f"draw canvas dark pixel diff too high: {dark_diff}",
        )

        src_segments = int(src.attrs.get("draw_segment_count", 0))
        rep_segments = int(rep.attrs.get("draw_segment_count", 0))
        segment_diff = abs(src_segments - rep_segments)
        require(
            segment_diff <= args.max_draw_segment_count_diff,
            f"draw segment count mismatch {src_segments} vs {rep_segments}",
        )

        src_weld = int(src["weld_active"][()].sum()) if "weld_active" in src else -1
        rep_weld = int(rep["weld_active"][()].sum()) if "weld_active" in rep else -1
        require(src_weld == rep_weld, f"weld-active frame count mismatch {src_weld} vs {rep_weld}")

    print(
        {
            "ok": True,
            "actions": list(actions_src.shape),
            "max_action_diff": max_action_diff,
            "max_sampled_frame_mean_abs_diff": max_sampled_mean,
            "max_sampled_frame_max_pixel_diff": int(max(max_pixel_diffs) if max_pixel_diffs else 0),
            "draw_canvas_dark_pixel_diff": dark_diff,
            "draw_segment_count": src_segments,
            "replay_draw_segment_count": rep_segments,
            "draw_segment_count_diff": segment_diff,
            "weld_active_frames": src_weld,
        }
    )


def read_phases(h5: h5py.File) -> list[str]:
    if "phases" not in h5:
        return []
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["phases"][()]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"REPLAY_MATCH_FAILED: {message}")


if __name__ == "__main__":
    main()
