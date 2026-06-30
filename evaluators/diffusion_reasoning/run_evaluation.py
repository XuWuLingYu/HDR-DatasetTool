from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluators.diffusion_reasoning.eval import EVALUATORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run environment-specific evaluation.")
    parser.add_argument("--env", default="one_line_drawing", choices=sorted(EVALUATORS))
    parser.add_argument("--pred-root", required=True, help="Root directory containing predicted videos.")
    parser.add_argument("--gt-root", default="outputs/diffusion_reasoning", help="Root directory containing manifests and metadata.")
    parser.add_argument("--visualization-root", default="outputs/eval", help="Root directory for rendered evaluation videos.")
    parser.add_argument("--levels", nargs="*", type=int, default=None, help="Optional subset of levels to evaluate.")
    parser.add_argument("--split", default="eval", choices=["train", "eval"], help="Dataset split to evaluate.")
    parser.add_argument("--output", default=None, help="Optional path for the output JSON report.")
    return parser.parse_args()


def iter_manifest_entries(gt_root: Path, env_name: str, split: str, levels: list[int] | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    target_levels = levels or sorted(
        int(path.name.split("_")[-1].replace("level", ""))
        for path in gt_root.glob(f"{env_name}_*level")
        if path.is_dir()
    )
    for level in target_levels:
        manifest_path = gt_root / f"{env_name}_{level}level" / f"{split}_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for item in manifest:
            item = dict(item)
            item["level"] = level
            entries.append(item)
    return entries


def resolve_prediction_path(pred_root: Path, env_name: str, level: int, split: str, video_name: str) -> Path:
    candidates = [
        pred_root / f"{env_name}_{level}level" / split / video_name,
        pred_root / split / video_name,
        pred_root / video_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "strict_pass_rate": 0.0,
            "relaxed_score_mean": 0.0,
        }

    strict_pass_rate = sum(1 for item in results if item["strict"]["passed"]) / max(len(results), 1)
    relaxed_score_mean = sum(item["relaxed"]["score"] for item in results) / max(len(results), 1)
    summary = {
        "sample_count": len(results),
        "strict_pass_rate": strict_pass_rate,
        "relaxed_score_mean": relaxed_score_mean,
    }
    first_task = results[0]["task"]
    if first_task == "one_line_drawing":
        summary["coverage_ratio_mean"] = (
            sum(item["relaxed"]["coverage_ratio"] for item in results) / max(len(results), 1)
        )
    elif first_task == "sliding_puzzle":
        for field in (
            "progress_ratio",
            "final_state_quality",
            "rule_adherence_ratio",
            "observability_ratio",
            "illegal_transition_ratio",
            "correct_tile_ratio",
        ):
            summary[f"{field}_mean"] = sum(item["relaxed"][field] for item in results) / max(len(results), 1)
    elif first_task == "water_pour":
        for field in (
            "rule_adherence_ratio",
            "progress_ratio",
            "final_state_quality",
            "observability_ratio",
            "solved_tube_ratio",
            "monochrome_tube_ratio",
            "conservation_ok_ratio",
            "illegal_transition_ratio",
        ):
            summary[f"{field}_mean"] = sum(item["relaxed"][field] for item in results) / max(len(results), 1)
    elif first_task == "sokoban":
        for field in (
            "progress_ratio",
            "final_state_quality",
            "rule_adherence_ratio",
            "observability_ratio",
            "illegal_transition_ratio",
        ):
            summary[f"{field}_mean"] = sum(item["relaxed"][field] for item in results) / max(len(results), 1)
    return summary


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    visualization_root = Path(args.visualization_root)
    evaluate_sample = EVALUATORS[args.env]

    manifest_entries = iter_manifest_entries(gt_root, args.env, args.split, args.levels)
    results: list[dict[str, Any]] = []
    missing_predictions: list[str] = []

    for item in manifest_entries:
        level_dir = gt_root / f"{args.env}_{item['level']}level"
        metadata_path = level_dir / args.split / item["metadata"]
        gt_video_path = level_dir / args.split / item["video"]
        prediction_path = resolve_prediction_path(pred_root, args.env, item["level"], args.split, item["video"])
        if not prediction_path.exists():
            missing_predictions.append(str(prediction_path))
            continue
        visualization_dir = visualization_root / f"{args.env}_{item['level']}level"
        visualization_output_path = visualization_dir / f"{item['sample_id']}_eval.mp4"
        results.append(
            evaluate_sample(
                prediction_path,
                metadata_path,
                gt_video_path=gt_video_path,
                visualization_output_path=visualization_output_path,
            )
        )

    payload = {
        "env": args.env,
        "split": args.split,
        "levels": args.levels,
        "pred_root": str(pred_root),
        "gt_root": str(gt_root),
        "visualization_root": str(visualization_root),
        "summary": aggregate_results(results),
        "missing_predictions": missing_predictions,
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
