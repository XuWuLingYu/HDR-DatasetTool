from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics_soft import evaluate_relaxed
from .metrics_strict import evaluate_strict
from .vision_mapper import extract_state_trace
from .visualization import render_evaluation_video
from evaluators.diffusion_reasoning.utils.video_io import load_video_frames


def evaluate_sample(
    video_path: str | Path,
    metadata_path: str | Path,
    *,
    gt_video_path: str | Path | None = None,
    visualization_output_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_metadata_path = Path(metadata_path)
    with resolved_metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    state_trace = extract_state_trace(video_path, metadata, gt_video_path=gt_video_path)
    strict = evaluate_strict(state_trace, metadata)
    relaxed = evaluate_relaxed(state_trace, metadata)
    result = {
        "sample_id": metadata["sample_id"],
        "task": metadata["task"],
        "difficulty_level": metadata["difficulty_level"],
        "split": metadata["split"],
        "topology_split": metadata["topology_split"],
        "bucket": metadata["bucket"],
        "family": metadata["family"],
        "video_path": str(video_path),
        "metadata_path": str(metadata_path),
        "strict": strict,
        "relaxed": relaxed,
    }
    if gt_video_path is not None and visualization_output_path is not None:
        gt_frames = load_video_frames(gt_video_path)
        pred_frames = load_video_frames(video_path)
        result["visualization_video"] = render_evaluation_video(
            gt_frames=gt_frames,
            pred_frames=pred_frames,
            state_trace=state_trace,
            strict_result=strict,
            relaxed_result=relaxed,
            metadata=metadata,
            output_path=visualization_output_path,
        )
    return result
