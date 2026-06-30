from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def load_video_frames(video_path: str | Path) -> list[np.ndarray]:
    """Load all frames from an RGB video into memory."""
    resolved_path = Path(video_path)
    reader = imageio.get_reader(resolved_path)
    try:
        return [np.asarray(frame, dtype=np.uint8) for frame in reader]
    finally:
        reader.close()


def write_video_frames(video_path: str | Path, frames: list[np.ndarray], fps: int) -> None:
    """Write RGB frames to an MP4 video."""
    resolved_path = Path(video_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(resolved_path, frames, fps=fps, macro_block_size=1)
