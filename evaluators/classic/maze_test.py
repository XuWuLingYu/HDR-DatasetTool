#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluators.classic.maze_videos import evaluate_video, find_prediction_video, load_struct, solution_path


def summarize_results(results: list[dict[str, object]], pred_dir: Path, struct_dir: Path) -> dict[str, object]:
    per_size: dict[int, dict[str, object]] = {}
    for maze_size in sorted({int(result["maze_size"]) for result in results}):
        size_results = [result for result in results if int(result["maze_size"]) == maze_size]
        strict_success_count = sum(int(result["strict_success"]) for result in size_results)
        loose_average_score = sum(float(result["loose_score"]) for result in size_results) / len(size_results) if size_results else 0.0
        per_size[maze_size] = {
            "count": len(size_results),
            "strict_success_count": strict_success_count,
            "strict_success_rate": strict_success_count / len(size_results) if size_results else 0.0,
            "strict_failure_breakdown": dict(
                Counter(result["strict_failure_reason"] for result in size_results if not int(result["strict_success"]))
            ),
            "loose_average_score": loose_average_score,
        }

    return {
        "pred_dir": str(pred_dir),
        "struct_dir": str(struct_dir),
        "strict": {
            "overall_success_rate": sum(int(result["strict_success"]) for result in results) / len(results),
            "per_size": per_size,
        },
        "loose": {
            "overall_average_score": sum(float(result["loose_score"]) for result in results) / len(results),
            "per_size": {
                maze_size: {"count": values["count"], "average_score": values["loose_average_score"]}
                for maze_size, values in per_size.items()
            },
        },
        "results": results,
    }


def print_scoreboard(summary: dict[str, object]) -> None:
    strict = summary["strict"]
    loose = summary["loose"]
    per_size = summary["strict"]["per_size"]
    payload = {
        "pred_dir": summary["pred_dir"],
        "strict": {
            "overall_success_rate": strict["overall_success_rate"],
            "per_size": {
                maze_size: {"success_rate": values["strict_success_rate"]}
                for maze_size, values in per_size.items()
            },
        },
        "loose": {
            "overall_average_score": loose["overall_average_score"],
            "per_size": {
                maze_size: {"average_score": values["average_score"]}
                for maze_size, values in loose["per_size"].items()
            },
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a prediction directory on maze_test with strict and loose metrics.")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing the predicted maze videos.")
    parser.add_argument("--struct-dir", type=str, default="maze_test_npy", help="Directory containing 50 maze_test npy files.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save the full summary JSON.")
    parser.add_argument("--output-npy", type=str, default="", help="Optional path to save the full summary NPY.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    struct_dir = Path(args.struct_dir)
    struct_paths = sorted(path for path in struct_dir.glob("maze_*.npy"))
    if len(struct_paths) != 50:
        raise RuntimeError(f"Expected 50 struct npy files in {struct_dir}, found {len(struct_paths)}")

    results: list[dict[str, object]] = []
    for struct_path in struct_paths:
        index = int(struct_path.stem.split("_")[-1])
        pred_video = find_prediction_video(pred_dir, index)
        if pred_video is None:
            struct = load_struct(struct_path)
            results.append(
                {
                    "file": f"maze_{index:04d}.mp4",
                    "maze_size": int(struct["maze_size"]),
                    "num_frames": 0,
                    "strict_success": 0,
                    "strict_failure_reason": "missing_video",
                    "success": 0,
                    "failure_reason": "missing_video",
                    "formalized_route": [],
                    "loose_score": 0.0,
                    "loose_overlap_count": 0,
                    "loose_gt_length": len(solution_path(struct)),
                    "loose_formalized_route": [],
                    "goal_reached_frame": None,
                }
            )
            continue
        results.append(evaluate_video(pred_video, struct_path))

    summary = summarize_results(results, pred_dir, struct_dir)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_npy:
        np.save(args.output_npy, summary, allow_pickle=True)

    print_scoreboard(summary)


if __name__ == "__main__":
    main()
