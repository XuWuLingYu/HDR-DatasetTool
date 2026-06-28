#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact summary for one generated episode.")
    parser.add_argument("episode_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ep = args.episode_dir
    with (ep / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    actions = np.load(ep / "actions.npy")
    states = np.load(ep / "states.npz")
    cap = cv2.VideoCapture(str(ep / "video.mp4"))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"episode: {ep}")
    print(f"sketch: {meta['sketch_type']} strokes={meta['num_strokes']} frames={meta['num_frames']}")
    print(f"video: {video_frames} frames @ {fps:.2f} fps")
    print(f"actions: shape={actions.shape} dtype={actions.dtype}")
    print(f"states: ee_xy={states['ee_xy'].shape} joints={states['joints'].shape}")
    print(f"pen_down_ratio: {states['pen_down'].mean():.3f}")


if __name__ == "__main__":
    main()
