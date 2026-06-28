#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from eval_mixed_reasoning import (
    SKIPPED_STATUSES,
    compute_halfsample_std,
    make_summary,
)


DEFAULT_BASE_EVAL_JSON = (
    "/mnt/zezhong/users/zhuoqun/"
    "testjsonl_rollout_8gpu_len32_20260506_043655/"
    "mixed_reasoning_eval.json"
)
DEFAULT_OUTPUT_JSON = (
    "/mnt/zezhong/users/zhuoqun/"
    "testjsonl_rollout_8gpu_len32_20260506_043655/"
    "mixed_reasoning_eval_videogpt_relaxed.json"
)


RULE_DESCRIPTIONS = {
    "maze": (
        "Pass if route has at least 3 GT-overlap cells, or relaxed route score >= 0.20. "
        "This recovers partially tracked trajectories that fail official strict due to disappear."
    ),
    "hanoi": (
        "Pass if loose plan overlap >= 0.50, or if move legality is perfect and goal completion ratio >= 0.50."
    ),
    "one_line_drawing": (
        "Pass if relaxed score >= 0.85, coverage_ratio >= 0.80, and on_shape_ratio >= 0.45."
    ),
    "sliding_puzzle": (
        "Pass if relaxed score >= 0.60, final_state_quality >= 0.50, and rule_adherence_ratio >= 0.95."
    ),
    "sokoban": (
        "Pass if relaxed score >= 0.15, rule_adherence_ratio >= 0.95, and observability_ratio >= 0.13. "
        "This is intentionally weaker than official success and should be read as 'valid observable behavior'."
    ),
    "water_pour": (
        "Pass if relaxed score >= 0.65, progress_ratio >= 0.50, and conservation_ok_ratio >= 1.00."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Temporary VideoGPT-only mixed-reasoning post-processor. "
            "It keeps official loose scores unchanged, but replaces strict success "
            "with a softer task-specific success heuristic."
        )
    )
    parser.add_argument("--base-eval-json", type=str, default=DEFAULT_BASE_EVAL_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    return parser


def _maze_relaxed(row: dict[str, Any]) -> tuple[int, str]:
    detail = row.get("detail", {})
    overlap = int(detail.get("loose_overlap_count", 0))
    loose = float(row.get("loose_score") or 0.0)
    if overlap >= 3:
        return 1, f"maze_overlap_ge_3({overlap})"
    if loose >= 0.20:
        return 1, f"maze_loose_ge_0.20({loose:.4f})"
    return 0, f"maze_overlap={overlap},loose={loose:.4f}"


def _hanoi_relaxed(row: dict[str, Any]) -> tuple[int, str]:
    detail = row.get("detail", {})
    overlap = float(detail.get("loose_plan_overlap", row.get("loose_score") or 0.0))
    legal = float(detail.get("legal_move_ratio", 0.0))
    goal = float(detail.get("goal_completion_ratio", 0.0))
    if overlap >= 0.50:
        return 1, f"hanoi_loose_plan_overlap_ge_0.50({overlap:.4f})"
    if legal >= 1.00 and goal >= 0.50:
        return 1, f"hanoi_legal_and_goal_partial(legal={legal:.4f},goal={goal:.4f})"
    return 0, f"hanoi_overlap={overlap:.4f},legal={legal:.4f},goal={goal:.4f}"


def _one_line_relaxed(row: dict[str, Any]) -> tuple[int, str]:
    relaxed = row.get("detail", {}).get("relaxed", {})
    score = float(relaxed.get("score", 0.0))
    coverage = float(relaxed.get("coverage_ratio", 0.0))
    on_shape = float(relaxed.get("on_shape_ratio", 0.0))
    if score >= 0.85 and coverage >= 0.80 and on_shape >= 0.45:
        return 1, f"one_line_score={score:.4f},coverage={coverage:.4f},on_shape={on_shape:.4f}"
    return 0, f"one_line_score={score:.4f},coverage={coverage:.4f},on_shape={on_shape:.4f}"


def _sliding_relaxed(row: dict[str, Any]) -> tuple[int, str]:
    relaxed = row.get("detail", {}).get("relaxed", {})
    score = float(relaxed.get("score", 0.0))
    final_quality = float(relaxed.get("final_state_quality", 0.0))
    rule = float(relaxed.get("rule_adherence_ratio", 0.0))
    if score >= 0.60 and final_quality >= 0.50 and rule >= 0.95:
        return 1, f"sliding_score={score:.4f},final={final_quality:.4f},rule={rule:.4f}"
    return 0, f"sliding_score={score:.4f},final={final_quality:.4f},rule={rule:.4f}"


def _sokoban_relaxed(row: dict[str, Any]) -> tuple[int, str]:
    relaxed = row.get("detail", {}).get("relaxed", {})
    score = float(relaxed.get("score", 0.0))
    rule = float(relaxed.get("rule_adherence_ratio", 0.0))
    observability = float(relaxed.get("observability_ratio", 0.0))
    if score >= 0.15 and rule >= 0.95 and observability >= 0.13:
        return 1, f"sokoban_score={score:.4f},rule={rule:.4f},obs={observability:.4f}"
    return 0, f"sokoban_score={score:.4f},rule={rule:.4f},obs={observability:.4f}"


def _water_relaxed(row: dict[str, Any]) -> tuple[int, str]:
    relaxed = row.get("detail", {}).get("relaxed", {})
    score = float(relaxed.get("score", 0.0))
    progress = float(relaxed.get("progress_ratio", 0.0))
    conservation = float(relaxed.get("conservation_ok_ratio", 0.0))
    if score >= 0.65 and progress >= 0.50 and conservation >= 1.00:
        return 1, f"water_score={score:.4f},progress={progress:.4f},conservation={conservation:.4f}"
    return 0, f"water_score={score:.4f},progress={progress:.4f},conservation={conservation:.4f}"


RELAXED_RULES = {
    "maze": _maze_relaxed,
    "hanoi": _hanoi_relaxed,
    "one_line_drawing": _one_line_relaxed,
    "sliding_puzzle": _sliding_relaxed,
    "sokoban": _sokoban_relaxed,
    "water_pour": _water_relaxed,
}


def compute_relaxed_success(row: dict[str, Any]) -> tuple[int | None, str]:
    status = str(row.get("status"))
    if status in SKIPPED_STATUSES:
        return None, f"skipped:{status}"
    if status != "ok":
        return None, f"non_ok:{status}"
    task = str(row.get("task"))
    rule_fn = RELAXED_RULES.get(task)
    if rule_fn is None:
        return row.get("strict_pass"), "fallback_original_strict"
    return rule_fn(row)


def write_markdown(output_md_path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    official = payload["official_summary"]
    halfsample_std = payload["halfsample_std"]
    lines: list[str] = []
    lines.append("# VideoGPT Temporary Relaxed Eval")
    lines.append("")
    lines.append("- This file is temporary and VideoGPT-specific.")
    lines.append("- Official loose scores are preserved.")
    lines.append("- Official strict success is preserved per sample in `official_strict_pass`.")
    lines.append("- Summary `strict_mean` below refers to `videogpt_relaxed_success`, not official strict success.")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for task in ["hanoi", "maze", "one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"]:
        lines.append(f"- `{task}`: {RULE_DESCRIPTIONS[task]}")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Task | Official Strict | Relaxed Success | Loose Score | Relaxed Success STD | Loose STD |")
    lines.append("|:---|---:|---:|---:|---:|---:|")
    for task in ["hanoi", "maze", "one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"]:
        off = official["per_task"][task]
        cur = summary["per_task"][task]
        std = halfsample_std["per_task"][task]
        lines.append(
            "| {task} | {off_s:.2f} | {cur_s:.2f} | {loose:.2f} | {std_s:.2f} | {std_l:.2f} |".format(
                task=task,
                off_s=100.0 * float(off["strict_mean"] or 0.0),
                cur_s=100.0 * float(cur["strict_mean"] or 0.0),
                loose=100.0 * float(cur["loose_mean"] or 0.0),
                std_s=100.0 * float(std["strict_std_mean_over_levels"] or 0.0),
                std_l=100.0 * float(std["loose_std_mean_over_levels"] or 0.0),
            )
        )
    lines.append(
        "| overall | {off_s:.2f} | {cur_s:.2f} | {loose:.2f} | {std_s:.2f} | {std_l:.2f} |".format(
            off_s=100.0 * float(official["overall"]["strict_mean"] or 0.0),
            cur_s=100.0 * float(summary["overall"]["strict_mean"] or 0.0),
            loose=100.0 * float(summary["overall"]["loose_mean"] or 0.0),
            std_s=100.0 * float(halfsample_std["overall"]["strict_std_mean_over_tasks"] or 0.0),
            std_l=100.0 * float(halfsample_std["overall"]["loose_std_mean_over_tasks"] or 0.0),
        )
    )
    lines.append("")
    output_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    base_eval_json = Path(args.base_eval_json)
    if not base_eval_json.exists():
        raise FileNotFoundError(f"Base eval json not found: {base_eval_json}")

    base_payload = json.loads(base_eval_json.read_text(encoding="utf-8"))
    base_results = base_payload.get("results", [])
    if not isinstance(base_results, list):
        raise ValueError("Base eval json has invalid `results` field")

    transformed_results: list[dict[str, Any]] = []
    for row in base_results:
        new_row = copy.deepcopy(row)
        relaxed_success, relaxed_reason = compute_relaxed_success(row)
        new_row["official_strict_pass"] = row.get("strict_pass")
        new_row["videogpt_relaxed_success"] = relaxed_success
        new_row["videogpt_relaxed_reason"] = relaxed_reason
        if relaxed_success is not None:
            new_row["strict_pass"] = int(relaxed_success)
        transformed_results.append(new_row)

    summary = make_summary(transformed_results)
    halfsample_std = compute_halfsample_std(transformed_results)

    payload = {
        "mode": "videogpt_temp_relaxed_success",
        "temporary": True,
        "videogpt_only": True,
        "base_eval_json": str(base_eval_json),
        "rule_descriptions": RULE_DESCRIPTIONS,
        "official_summary": base_payload.get("summary"),
        "official_halfsample_std": base_payload.get("halfsample_std"),
        "pred_dir": base_payload.get("pred_dir"),
        "test_jsonl": base_payload.get("test_jsonl"),
        "paired_count": base_payload.get("paired_count"),
        "summary": summary,
        "halfsample_std": halfsample_std,
        "results": transformed_results,
    }

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md = output_json.with_suffix(".md")
    write_markdown(output_md, payload)

    brief = {
        "output_json": str(output_json),
        "output_markdown": str(output_md),
        "overall_relaxed_success": summary["overall"]["strict_mean"],
        "overall_loose_score": summary["overall"]["loose_mean"],
        "per_task_relaxed_success": {
            task: summary["per_task"][task]["strict_mean"]
            for task in ["hanoi", "maze", "one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"]
        },
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
