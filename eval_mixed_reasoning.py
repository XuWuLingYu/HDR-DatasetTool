#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


DEFAULT_TEST_JSONL = (
    "/mnt/zezhong/HiDiT/Causal-Forcing/dataset/"
    "mixed_reasoning_all_125f_224/vertical_125f/test.jsonl"
)
DEFAULT_FULL_TEST_JSONL = (
    "/mnt/zezhong/HiDiT/Causal-Forcing/dataset/"
    "mixed_reasoning_all_125f_224/vertical_125f/"
    "test_full_backup_before_level20_20260418.jsonl"
)
DEFAULT_MANIFEST_JSONL = (
    "/mnt/zezhong/HiDiT/Causal-Forcing/dataset/"
    "mixed_reasoning_all_125f_224/vertical_125f/manifest.jsonl"
)
DEFAULT_MAZE_STRUCT_DIR = "/mnt/zezhong/MazeDataset/maze_test_npy"
DEFAULT_HANOI_DATASET_DIR = "/mnt/zezhong/MazeDataset/hanoi_dataset"
DEFAULT_HANOI_SPLIT = "hanoi_test"
DEFAULT_ZERONG_REPO_ROOT = (
    "/mnt/world_foundational_model/wfm_ckp-fileset/maizerong/projects/"
    "diffusion-reasoning/repo/diffusion-reasoning-dataset"
)
SKIPPED_STATUSES = {"skipped_realbot_maze"}
LEVEL_RE = re.compile(r"_(\d+)level$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mixed_reasoning vertical_125f predictions. "
            "Prediction mp4 files are sorted lexicographically and matched "
            "one-to-one with test.jsonl rows."
        )
    )
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--test-jsonl", type=str, default=DEFAULT_TEST_JSONL)
    parser.add_argument(
        "--full-test-jsonl",
        type=str,
        default=DEFAULT_FULL_TEST_JSONL,
        help=(
            "Full test jsonl used for index mapping when number of prediction videos "
            "equals full test size."
        ),
    )
    parser.add_argument("--manifest-jsonl", type=str, default=DEFAULT_MANIFEST_JSONL)
    parser.add_argument("--maze-struct-dir", type=str, default=DEFAULT_MAZE_STRUCT_DIR)
    parser.add_argument("--hanoi-dataset-dir", type=str, default=DEFAULT_HANOI_DATASET_DIR)
    parser.add_argument("--hanoi-split", type=str, default=DEFAULT_HANOI_SPLIT)
    parser.add_argument("--zerong-repo-root", type=str, default=DEFAULT_ZERONG_REPO_ROOT)
    parser.add_argument(
        "--resize-cache-dir",
        type=str,
        default="",
        help="Optional cache dir for temporary resized videos used by zerong evaluators.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start row index in test.jsonl for one-to-one alignment with sorted predictions.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of aligned pairs to evaluate.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads for sample-level parallel evaluation. Default: 1 (single-thread).",
    )
    parser.add_argument("--output-json", type=str, default="")
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_to_task(source: str) -> str:
    if source == "hanoi":
        return "hanoi"
    if source in {"maze_recursive", "realbot"}:
        return "maze"
    if source.startswith("zerong_one_line_drawing"):
        return "one_line_drawing"
    if source.startswith("zerong_sliding_puzzle"):
        return "sliding_puzzle"
    if source.startswith("zerong_sokoban"):
        return "sokoban"
    if source.startswith("zerong_water_pour"):
        return "water_pour"
    return "unknown"


def env_from_task(task: str) -> str | None:
    if task in {"one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"}:
        return task
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def read_video_hw(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame: {path}")
    h, w = frame.shape[:2]
    return int(h), int(w)


def resize_video(input_path: Path, output_path: Path, target_h: int, target_w: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_w, target_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {output_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        cap.release()
        writer.release()


def make_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "ok_count": 0,
            "scored_count": 0,
            "skipped_count": 0,
            "strict_values": [],
            "loose_values": [],
            "status_breakdown": defaultdict(int),
        }
    )
    overall = {
        "count": 0,
        "ok_count": 0,
        "scored_count": 0,
        "skipped_count": 0,
        "strict_values": [],
        "loose_values": [],
        "status_breakdown": defaultdict(int),
    }

    for item in results:
        task = str(item["task"])
        status = str(item["status"])
        strict_pass = item.get("strict_pass")
        loose_score = item.get("loose_score")

        for bucket in (overall, per_task[task]):
            bucket["count"] += 1
            bucket["status_breakdown"][status] += 1
            if status in SKIPPED_STATUSES:
                bucket["skipped_count"] += 1
                continue
            if status == "ok":
                bucket["ok_count"] += 1
            if strict_pass is not None or loose_score is not None:
                bucket["scored_count"] += 1
            if strict_pass is not None:
                bucket["strict_values"].append(float(strict_pass))
            if loose_score is not None:
                bucket["loose_values"].append(float(loose_score))

    def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
        strict_mean = mean(bucket["strict_values"])
        loose_mean = mean(bucket["loose_values"])
        return {
            "count": int(bucket["count"]),
            "ok_count": int(bucket["ok_count"]),
            "scored_count": int(bucket["scored_count"]),
            "skipped_count": int(bucket["skipped_count"]),
            "strict_mean": strict_mean,
            "loose_mean": loose_mean,
            "status_breakdown": dict(bucket["status_breakdown"]),
        }

    return {
        "overall": finalize(overall),
        "per_task": {task: finalize(bucket) for task, bucket in sorted(per_task.items())},
    }


def compute_halfsample_std(
    results: list[dict[str, Any]],
    *,
    seed: int = 20260505,
    num_trials: int = 10,
) -> dict[str, Any]:
    scored_rows = [
        row for row in results
        if row.get("status") not in SKIPPED_STATUSES
        and row.get("strict_pass") is not None
        and row.get("loose_score") is not None
    ]

    def level_key(row: dict[str, Any]) -> str | None:
        task = str(row.get("task"))
        detail = row.get("detail")
        if not isinstance(detail, dict):
            return None
        if task == "maze":
            maze_size = detail.get("maze_size")
            return str(int(maze_size)) if maze_size is not None else None
        if task == "hanoi":
            domain = str(detail.get("domain_split", ""))
            if domain == "in_domain":
                return "InD"
            if domain == "ood":
                return "OOD"
            return None
        if task in {"one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"}:
            level = extract_level(row)
            return str(level) if level is not None else None
        return None

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in scored_rows:
        level = level_key(row)
        if level is not None:
            grouped[str(row["task"])][level].append(row)

    task_order = ["hanoi", "maze", "one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"]
    per_task: dict[str, Any] = {}
    for task in task_order:
        levels = grouped.get(task, {})
        level_items: dict[str, Any] = {}
        strict_stds: list[float] = []
        loose_stds: list[float] = []
        for level, rows in sorted(levels.items(), key=lambda kv: (kv[0] not in {"InD", "OOD"}, kv[0])):
            n = len(rows)
            k = n // 2
            if k <= 0:
                continue
            strict_values = [float(row["strict_pass"]) for row in rows]
            loose_values = [float(row["loose_score"]) for row in rows]
            strict_means: list[float] = []
            loose_means: list[float] = []
            for trial in range(num_trials):
                rng = random.Random(repr((seed, task, level, trial)))
                sampled = rng.sample(range(n), k)
                strict_means.append(sum(strict_values[i] for i in sampled) / k)
                loose_means.append(sum(loose_values[i] for i in sampled) / k)
            strict_std = statistics.pstdev(strict_means)
            loose_std = statistics.pstdev(loose_means)
            strict_stds.append(strict_std)
            loose_stds.append(loose_std)
            level_items[level] = {
                "count": n,
                "sampled_count": k,
                "strict_means": strict_means,
                "loose_means": loose_means,
                "strict_std": strict_std,
                "loose_std": loose_std,
            }
        if level_items:
            per_task[task] = {
                "levels": level_items,
                "strict_std_mean_over_levels": sum(strict_stds) / len(strict_stds),
                "loose_std_mean_over_levels": sum(loose_stds) / len(loose_stds),
            }

    strict_task_values = [item["strict_std_mean_over_levels"] for item in per_task.values()]
    loose_task_values = [item["loose_std_mean_over_levels"] for item in per_task.values()]
    return {
        "seed": seed,
        "num_trials": num_trials,
        "sampling_rule": (
            "For each task-level with N samples, sample N//2 without replacement, "
            "compute mean, repeat num_trials times, then take population std over those means."
        ),
        "per_task": per_task,
        "overall": {
            "strict_std_mean_over_tasks": mean(strict_task_values),
            "loose_std_mean_over_tasks": mean(loose_task_values),
        },
    }


def extract_level(item: dict[str, Any]) -> int | None:
    source = str(item.get("source") or "")
    match = LEVEL_RE.search(source)
    if match is not None:
        return int(match.group(1))
    detail = item.get("detail")
    if isinstance(detail, dict):
        level = detail.get("difficulty_level")
        if isinstance(level, int):
            return level
        if isinstance(level, str):
            match2 = re.search(r"(\d+)", level)
            if match2 is not None:
                return int(match2.group(1))
    return None


def strict_loose(rows: list[dict[str, Any]]) -> tuple[int, float | None, float | None]:
    strict_values: list[float] = []
    loose_values: list[float] = []
    for row in rows:
        strict = row.get("strict_pass")
        loose = row.get("loose_score")
        if strict is not None:
            strict_values.append(float(strict))
        if loose is not None:
            loose_values.append(float(loose))
    return len(rows), mean(strict_values), mean(loose_values)


def f4(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "NA"


def write_markdown_report(output_md_path: Path, payload: dict[str, Any]) -> None:
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []

    scored_rows = [
        row for row in results
        if row.get("status") not in SKIPPED_STATUSES
        and row.get("strict_pass") is not None
        and row.get("loose_score") is not None
    ]

    maze_rows = [row for row in scored_rows if row.get("task") == "maze"]
    hanoi_rows = [row for row in scored_rows if row.get("task") == "hanoi"]
    other_task_names = ["one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"]

    lines: list[str] = []
    lines.append("# Mixed Reasoning Eval Report")
    lines.append("")
    lines.append(f"- pred_dir: `{payload.get('pred_dir', '')}`")
    lines.append(f"- test_jsonl: `{payload.get('test_jsonl', '')}`")
    lines.append(f"- paired_count: `{payload.get('paired_count', 0)}`")
    lines.append(f"- scored_count: `{payload.get('summary', {}).get('overall', {}).get('scored_count', 0)}`")
    lines.append(f"- skipped_count: `{payload.get('summary', {}).get('overall', {}).get('skipped_count', 0)}`")
    halfsample_std = payload.get("halfsample_std")
    if isinstance(halfsample_std, dict):
        overall_std = halfsample_std.get("overall", {})
        lines.append(f"- halfsample_std_overall_strict: `{f4(overall_std.get('strict_std_mean_over_tasks'))}`")
        lines.append(f"- halfsample_std_overall_loose: `{f4(overall_std.get('loose_std_mean_over_tasks'))}`")
    lines.append("")

    lines.append("## Maze By N")
    lines.append("")
    lines.append("| N | Count | Strict | Loose |")
    lines.append("|---:|---:|---:|---:|")
    maze_by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in maze_rows:
        detail = row.get("detail")
        if isinstance(detail, dict) and detail.get("maze_size") is not None:
            maze_by_n[int(detail["maze_size"])].append(row)
    for n in sorted(maze_by_n):
        count, strict_avg, loose_avg = strict_loose(maze_by_n[n])
        lines.append(f"| {n} | {count} | {f4(strict_avg)} | {f4(loose_avg)} |")
    lines.append("")

    lines.append("## Hanoi By Domain")
    lines.append("")
    lines.append("| Domain | Count | Strict | Loose |")
    lines.append("|:---|---:|---:|---:|")
    hanoi_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hanoi_rows:
        detail = row.get("detail")
        domain = ""
        if isinstance(detail, dict):
            domain = str(detail.get("domain_split", ""))
        if domain in {"in_domain", "ood"}:
            hanoi_by_domain[domain].append(row)
    for domain_key, domain_name in (("in_domain", "InD"), ("ood", "OOD")):
        count, strict_avg, loose_avg = strict_loose(hanoi_by_domain.get(domain_key, []))
        lines.append(f"| {domain_name} | {count} | {f4(strict_avg)} | {f4(loose_avg)} |")
    lines.append("")

    lines.append("## Other Tasks By Level")
    lines.append("")
    for task in other_task_names:
        task_rows = [row for row in scored_rows if row.get("task") == task]
        level_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            level = extract_level(row)
            if level is not None:
                level_rows[level].append(row)
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Level | Count | Strict | Loose |")
        lines.append("|---:|---:|---:|---:|")
        for level in sorted(level_rows):
            count, strict_avg, loose_avg = strict_loose(level_rows[level])
            lines.append(f"| {level} | {count} | {f4(strict_avg)} | {f4(loose_avg)} |")
        lines.append("")

    if isinstance(halfsample_std, dict):
        per_task_std = halfsample_std.get("per_task", {})
        overall_std = halfsample_std.get("overall", {})
        lines.append("## Half-Sample STD Summary")
        lines.append("")
        lines.append("| Task | Strict STD | Loose STD |")
        lines.append("|:---|---:|---:|")
        for task in ["hanoi", "maze", "one_line_drawing", "sliding_puzzle", "sokoban", "water_pour"]:
            item = per_task_std.get(task, {})
            lines.append(
                f"| {task} | {f4(item.get('strict_std_mean_over_levels'))} | {f4(item.get('loose_std_mean_over_levels'))} |"
            )
        lines.append(
            f"| overall | {f4(overall_std.get('strict_std_mean_over_tasks'))} | {f4(overall_std.get('loose_std_mean_over_tasks'))} |"
        )
        lines.append("")

    output_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    pred_dir = Path(args.pred_dir)
    test_jsonl = Path(args.test_jsonl)
    full_test_jsonl = Path(args.full_test_jsonl)
    manifest_jsonl = Path(args.manifest_jsonl)
    maze_struct_dir = Path(args.maze_struct_dir)
    hanoi_dataset_dir = Path(args.hanoi_dataset_dir)
    zerong_repo_root = Path(args.zerong_repo_root)
    resize_cache_dir = (
        Path(args.resize_cache_dir)
        if args.resize_cache_dir
        else (pred_dir / ".mixed_eval_resize_cache")
    )

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not test_jsonl.exists():
        raise FileNotFoundError(f"test.jsonl not found: {test_jsonl}")
    if not manifest_jsonl.exists():
        raise FileNotFoundError(f"manifest.jsonl not found: {manifest_jsonl}")

    # Local evaluators (maze/hanoi)
    local_root = Path(__file__).resolve().parent
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))
    from eval_maze_videos import evaluate_video as evaluate_maze_video  # type: ignore
    from hanoi_dataset.eval_hanoi import EvalConfig as HanoiEvalConfig  # type: ignore
    from hanoi_dataset.eval_hanoi import evaluate_one as evaluate_hanoi_one  # type: ignore
    from hanoi_dataset.eval_hanoi import load_records as load_hanoi_records  # type: ignore

    # Zerong evaluators
    zerong_src_root = zerong_repo_root / "src"
    if str(zerong_src_root) not in sys.path:
        sys.path.insert(0, str(zerong_src_root))
    from eval import EVALUATORS  # type: ignore

    test_entries = load_jsonl(test_jsonl)
    manifest_entries = [entry for entry in load_jsonl(manifest_jsonl) if entry.get("split") == "test"]

    manifest_by_test_video_name: dict[str, dict[str, Any]] = {}
    for item in manifest_entries:
        test_video_name = Path(str(item["video_path"])).name
        manifest_by_test_video_name[test_video_name] = item

    hanoi_records = load_hanoi_records(hanoi_dataset_dir, args.hanoi_split)
    hanoi_record_by_file = {str(record["file"]): record for record in hanoi_records}
    hanoi_config = HanoiEvalConfig(
        dataset_dir=str(hanoi_dataset_dir),
        split=args.hanoi_split,
        pred_dir=str(pred_dir),
        output_json=None,
    )

    pred_paths = sorted(pred_dir.glob("*.mp4"))
    start_index = max(0, int(args.start_index))
    if start_index >= len(test_entries):
        raise ValueError(
            f"--start-index ({start_index}) must be < number of test entries ({len(test_entries)})"
        )

    available_test = len(test_entries) - start_index
    eval_test_indices = list(range(start_index, len(test_entries)))
    if args.limit > 0:
        eval_test_indices = eval_test_indices[: args.limit]

    alignment_mode = "prefix"
    full_test_count: int | None = None
    aligned_pred_ranks: list[int] = []
    if full_test_jsonl.exists():
        full_test_entries = load_jsonl(full_test_jsonl)
        full_test_count = len(full_test_entries)
        if len(pred_paths) == full_test_count:
            full_pred_rank_by_video_path: dict[str, int] = {}
            for pred_rank, item in enumerate(full_test_entries):
                video_path = str(item.get("video_path", ""))
                if video_path in full_pred_rank_by_video_path:
                    raise ValueError(
                        "Duplicate video_path found in full test jsonl, mapping is ambiguous: "
                        f"{video_path}"
                    )
                full_pred_rank_by_video_path[video_path] = pred_rank

            missing_video_paths: list[str] = []
            for test_index in eval_test_indices:
                video_path = str(test_entries[test_index].get("video_path", ""))
                pred_rank = full_pred_rank_by_video_path.get(video_path)
                if pred_rank is None:
                    missing_video_paths.append(video_path)
                else:
                    aligned_pred_ranks.append(pred_rank)

            if missing_video_paths:
                preview = "\n".join(missing_video_paths[:5])
                raise ValueError(
                    "Cannot map subset test entries to full test indices. "
                    f"Missing {len(missing_video_paths)} video_path(s). Examples:\n{preview}"
                )
            alignment_mode = "mapped_full_test"

    if alignment_mode == "prefix":
        aligned_total = min(len(pred_paths), len(eval_test_indices))
        eval_test_indices = eval_test_indices[:aligned_total]
        aligned_pred_ranks = list(range(aligned_total))
    else:
        aligned_total = len(eval_test_indices)

    def evaluate_pair(pair_rank: int) -> dict[str, Any]:
        pred_rank = aligned_pred_ranks[pair_rank]
        pred_path = pred_paths[pred_rank]
        test_index = eval_test_indices[pair_rank]
        test_item = test_entries[test_index]
        test_video_name = Path(str(test_item["video_path"])).name
        manifest_item = manifest_by_test_video_name.get(test_video_name)

        base_result: dict[str, Any] = {
            "index": test_index,
            "pair_rank": pair_rank,
            "pred_rank": pred_rank,
            "pred_path": str(pred_path),
            "test_video_path": str(test_item["video_path"]),
            "prompt": str(test_item["prompt"]),
            "source": None,
            "task": "unknown",
            "status": "error",
            "strict_pass": None,
            "loose_score": None,
            "detail": {},
        }

        if manifest_item is None:
            base_result["status"] = "missing_manifest"
            return base_result

        source = str(manifest_item["source"])
        task = source_to_task(source)
        src_video_path = Path(str(manifest_item["src_video_path"]))
        base_result["source"] = source
        base_result["task"] = task
        base_result["src_video_path"] = str(src_video_path)

        try:
            if task == "maze":
                if source == "realbot":
                    base_result["status"] = "skipped_realbot_maze"
                    base_result["detail"] = {
                        "reason": "realbot maze sample is excluded from scoring by policy"
                    }
                else:
                    struct_path = maze_struct_dir / src_video_path.name.replace(".mp4", ".npy")
                    if not struct_path.exists():
                        base_result["status"] = "missing_maze_struct"
                        base_result["detail"] = {"struct_path": str(struct_path)}
                    else:
                        out = evaluate_maze_video(pred_path, struct_path)
                        base_result["status"] = "ok"
                        base_result["strict_pass"] = int(out["strict_success"])
                        base_result["loose_score"] = float(out["loose_score"])
                        base_result["detail"] = out

            elif task == "hanoi":
                hanoi_record = hanoi_record_by_file.get(src_video_path.name)
                if hanoi_record is None:
                    base_result["status"] = "missing_hanoi_record"
                else:
                    out = evaluate_hanoi_one(hanoi_record, pred_path, hanoi_config)
                    loose = out.get("loose_score", out.get("loose_plan_overlap"))
                    base_result["status"] = "ok"
                    base_result["strict_pass"] = int(out["strict_success"])
                    base_result["loose_score"] = float(loose) if loose is not None else None
                    base_result["detail"] = out

            else:
                env_name = env_from_task(task)
                if env_name is None:
                    base_result["status"] = "unsupported_task"
                else:
                    metadata_path = src_video_path.with_suffix(".json")
                    if not metadata_path.exists():
                        base_result["status"] = "missing_metadata"
                        base_result["detail"] = {"metadata_path": str(metadata_path)}
                    else:
                        pred_eval_path = pred_path
                        src_h, src_w = read_video_hw(src_video_path)
                        pred_h, pred_w = read_video_hw(pred_path)
                        if (pred_h, pred_w) != (src_h, src_w):
                            pred_eval_path = resize_cache_dir / f"{test_index:06d}_{pred_path.name}"
                            if not pred_eval_path.exists():
                                resize_video(pred_path, pred_eval_path, src_h, src_w)

                        out = EVALUATORS[env_name](
                            pred_eval_path,
                            metadata_path,
                            gt_video_path=src_video_path,
                        )
                        base_result["status"] = "ok"
                        strict = out.get("strict", {})
                        relaxed = out.get("relaxed", {})
                        base_result["strict_pass"] = 1 if bool(strict.get("passed", False)) else 0
                        base_result["loose_score"] = float(relaxed.get("score", 0.0))
                        if pred_eval_path != pred_path:
                            base_result["detail_pred_eval_path"] = str(pred_eval_path)
                        base_result["detail"] = out

        except Exception as exc:  # noqa: BLE001
            base_result["status"] = "exception"
            base_result["detail"] = {"error": repr(exc)}

        return base_result

    results: list[dict[str, Any]] = []
    num_workers = max(1, int(args.num_workers))
    if num_workers <= 1 or aligned_total <= 1:
        for pair_rank in tqdm(range(aligned_total), desc="Evaluating mixed", leave=True):
            results.append(evaluate_pair(pair_rank))
    else:
        max_workers = min(num_workers, aligned_total)
        indexed_results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_by_rank = {
                executor.submit(evaluate_pair, pair_rank): pair_rank
                for pair_rank in range(aligned_total)
            }
            progress_desc = f"Evaluating mixed ({max_workers} threads)"
            for future in tqdm(as_completed(future_by_rank), total=aligned_total, desc=progress_desc, leave=True):
                pair_rank = future_by_rank[future]
                indexed_results[pair_rank] = future.result()

        results = [indexed_results[pair_rank] for pair_rank in range(aligned_total)]

    summary = make_summary(results)
    halfsample_std = compute_halfsample_std(results)
    payload = {
        "pred_dir": str(pred_dir),
        "test_jsonl": str(test_jsonl),
        "full_test_jsonl": str(full_test_jsonl),
        "manifest_jsonl": str(manifest_jsonl),
        "alignment_mode": alignment_mode,
        "paired_count": aligned_total,
        "start_index": start_index,
        "pred_count": len(pred_paths),
        "test_count": len(test_entries),
        "full_test_count": full_test_count,
        "unpaired_predictions": max(0, len(pred_paths) - len(set(aligned_pred_ranks))),
        "unpaired_test_entries": max(0, available_test - len(eval_test_indices)),
        "summary": summary,
        "halfsample_std": halfsample_std,
        "results": results,
    }

    output_path = Path(args.output_json) if args.output_json else pred_dir / "mixed_reasoning_eval.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md_path = output_path.with_suffix(".md")
    write_markdown_report(output_md_path, payload)

    brief = {
        "output_json": str(output_path),
        "output_markdown": str(output_md_path),
        "paired_count": aligned_total,
        "start_index": start_index,
        "alignment_mode": alignment_mode,
        "pred_count": len(pred_paths),
        "test_count": len(test_entries),
        "full_test_count": full_test_count,
        "overall": summary["overall"],
        "halfsample_std": halfsample_std["overall"],
        "per_task": {
            task: {
                "count": item["count"],
                "strict_mean": item["strict_mean"],
                "loose_mean": item["loose_mean"],
                "status_breakdown": item["status_breakdown"],
            }
            for task, item in summary["per_task"].items()
        },
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
