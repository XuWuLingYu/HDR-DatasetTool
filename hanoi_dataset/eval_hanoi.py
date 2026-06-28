#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


INDEX_RE = re.compile(r"^(\d+)")

FIXED_DISK_COLORS = np.asarray(
    [
        [46, 166, 78],
        [65, 105, 225],
        [236, 105, 180],
        [139, 94, 60],
        [148, 87, 235],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EvalConfig:
    dataset_dir: str = "/mnt/zezhong/MazeDataset/hanoi_dataset"
    split: str = "hanoi_test"
    pred_dir: str = (
        "/mnt/zezhong/HiDiT/Causal-Forcing/output/"
        "ar_diffusion_vertical_wan22_ti2v_5b_hanoi_125f_budget_7_11_17_26_40_50_eval"
    )
    output_json: str | None = None
    bucket_size: int = 10
    rod_margin_ratio: float = 0.23
    rod_margin_min: float = 14.0
    smooth_window: int = 2
    stable_stop_len: int = 8
    transient_segment_max_len_2disk: int = 12
    transient_segment_gap_max_2disk: int = 8
    min_component_area: int = 8
    min_x_span_ratio: float = 0.55
    min_lift_margin_disk_heights: float = 0.8
    brown_color_threshold: float = 35.0
    default_color_threshold: float = 45.0
    pink_color_threshold: float = 55.0
    purple_color_threshold: float = 50.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Hanoi predictions by decoding airborne disk moves, "
            "checking legality from the known initial state, and scoring plan overlap."
        )
    )
    parser.add_argument("--dataset-dir", type=str, default="/mnt/zezhong/MazeDataset/hanoi_dataset")
    parser.add_argument("--split", type=str, default="hanoi_test")
    parser.add_argument(
        "--pred-dir",
        type=str,
        default=(
            "/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_tf_framewise_wan22_ti2v_5b_hanoi_ar32_split21_eval"
        ),
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--bucket-size", type=int, default=10)
    return parser


def load_records(dataset_dir: Path, split: str) -> list[dict[str, object]]:
    metadata_path = dataset_dir / split / "metadata.jsonl"
    with metadata_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_prediction_paths(pred_dir: Path) -> dict[int, Path]:
    predictions: dict[int, Path] = {}
    for path in sorted(pred_dir.glob("*.mp4")):
        match = INDEX_RE.match(path.name)
        if match is None:
            continue
        predictions[int(match.group(1))] = path
    return predictions


def layout(num_disks: int, frame_size: int) -> dict[str, object]:
    disk_height = max(10, int(round(frame_size * 0.055)))
    base_y = frame_size - int(round(frame_size * 0.12))
    rod_top_y = max(22, base_y - disk_height * (num_disks + 1))
    margin = frame_size * 0.18
    rod_xs = np.asarray([margin, frame_size * 0.5, frame_size - margin], dtype=np.float32)
    spacing = float(rod_xs[1] - rod_xs[0])
    max_width = int(round(spacing * 0.72))
    min_width = max(18, int(round(max_width * 0.45)))
    widths = np.linspace(min_width, max_width, num_disks).round().astype(np.int16)
    lift_y = max(12.0, float(rod_top_y - disk_height))
    return {
        "frame_size": frame_size,
        "disk_height": disk_height,
        "base_y": float(base_y),
        "rod_top_y": float(rod_top_y),
        "rod_xs": rod_xs,
        "gap": spacing,
        "disk_widths": widths,
        "lift_y": lift_y,
    }


def color_threshold_for_disk(config: EvalConfig, disk_index: int) -> float:
    if disk_index == 2:
        return config.pink_color_threshold
    if disk_index == 3:
        return config.brown_color_threshold
    if disk_index == 4:
        return config.purple_color_threshold
    return config.default_color_threshold


def connected_disk_candidate(
    frame: np.ndarray,
    disk_index: int,
    num_disks: int,
    layout_info: dict[str, object],
    config: EvalConfig,
    prev_center: np.ndarray | None,
) -> dict[str, float] | None:
    color = FIXED_DISK_COLORS[disk_index]
    threshold = color_threshold_for_disk(config, disk_index)
    frame_arr = np.asarray(frame, dtype=np.float32)
    dist = np.linalg.norm(frame_arr - color[None, None, :], axis=2)
    mask = (dist < threshold).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    expected_width = float(np.asarray(layout_info["disk_widths"])[disk_index])
    expected_height = float(layout_info["disk_height"])
    area_target = expected_width * expected_height * 0.65

    best: dict[str, float] | None = None
    best_cost = float("inf")
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        cx, cy = centroids[label]
        if area < config.min_component_area:
            continue
        if height < max(3, 0.35 * expected_height) or height > 2.5 * expected_height:
            continue
        if width < 0.4 * expected_width or width > 2.2 * expected_width:
            continue
        if width < height * 1.1:
            continue
        if cy < float(layout_info["lift_y"]) - 0.6 * expected_height:
            continue

        cost = abs(float(area) - area_target) / max(area_target, 1.0)
        cost += abs(float(width) - expected_width) / max(expected_width, 1.0)
        cost += abs(float(height) - expected_height) / max(expected_height, 1.0)
        if prev_center is not None:
            cost += 0.02 * float(np.linalg.norm(np.asarray([cx, cy], dtype=np.float32) - prev_center))
        if cost < best_cost:
            best_cost = cost
            best = {
                "cx": float(cx),
                "cy": float(cy),
                "width": float(width),
                "height": float(height),
                "area": float(area),
            }
    return best


def decode_tracks(video_path: Path, num_disks: int, config: EvalConfig) -> tuple[list[list[dict[str, float] | None]], dict[str, int]]:
    reader = imageio.get_reader(str(video_path))
    try:
        num_frames = reader.count_frames()
        frames = [reader.get_data(index) for index in range(num_frames)]
    finally:
        reader.close()

    frame_height, frame_width = frames[0].shape[:2]
    layout_info = layout(num_disks, frame_width)
    prev_centers: list[np.ndarray | None] = [None] * num_disks
    tracks: list[list[dict[str, float] | None]] = [[] for _ in range(num_disks)]

    for frame in frames:
        for disk_index in range(num_disks):
            det = connected_disk_candidate(frame, disk_index, num_disks, layout_info, config, prev_centers[disk_index])
            if det is not None:
                prev_centers[disk_index] = np.asarray([det["cx"], det["cy"]], dtype=np.float32)
            tracks[disk_index].append(det)

    return tracks, {
        "num_frames": num_frames,
        "frame_height": frame_height,
        "frame_width": frame_width,
    }


def smooth_rods(rod_assignments: list[int | None], window: int) -> list[int | None]:
    smoothed: list[int | None] = []
    for index in range(len(rod_assignments)):
        values = [
            rod_assignments[offset]
            for offset in range(max(0, index - window), min(len(rod_assignments), index + window + 1))
            if rod_assignments[offset] is not None
        ]
        if not values:
            smoothed.append(rod_assignments[index])
            continue
        counts = Counter(values)
        best_rod, best_count = counts.most_common(1)[0]
        smoothed.append(best_rod if best_count >= 2 else rod_assignments[index])
    return smoothed


def rod_segments(rod_assignments: list[int | None]) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    index = 0
    while index < len(rod_assignments):
        rod = rod_assignments[index]
        if rod is None:
            index += 1
            continue
        end = index + 1
        while end < len(rod_assignments) and rod_assignments[end] == rod:
            end += 1
        segments.append((index, end - 1, rod))
        index = end
    return segments


def collapse_transient_segments(
    segments: list[tuple[int, int, int]],
    num_disks: int,
    config: EvalConfig,
) -> list[tuple[int, int, int]]:
    if num_disks != 2 or len(segments) < 3:
        return segments

    collapsed = list(segments)
    index = 1
    while index < len(collapsed) - 1:
        prev_segment = collapsed[index - 1]
        cur_segment = collapsed[index]
        next_segment = collapsed[index + 1]

        cur_len = cur_segment[1] - cur_segment[0] + 1
        prev_gap = cur_segment[0] - prev_segment[1] - 1
        next_gap = next_segment[0] - cur_segment[1] - 1
        is_short_pass_through = (
            cur_len <= config.transient_segment_max_len_2disk
            and prev_gap <= config.transient_segment_gap_max_2disk
            and next_gap <= config.transient_segment_gap_max_2disk
            and prev_segment[2] != cur_segment[2]
            and cur_segment[2] != next_segment[2]
            and prev_segment[2] != next_segment[2]
        )
        if is_short_pass_through:
            del collapsed[index]
            continue
        index += 1
    return collapsed


def high_rest_y(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return float(np.percentile(arr, 85))


def infer_airborne_moves(
    tracks: list[list[dict[str, float] | None]],
    num_disks: int,
    frame_width: int,
    config: EvalConfig,
) -> list[dict[str, object]]:
    layout_info = layout(num_disks, frame_width)
    rod_xs = np.asarray(layout_info["rod_xs"], dtype=np.float32)
    rod_margin = max(config.rod_margin_min, float(layout_info["gap"]) * config.rod_margin_ratio)
    min_x_span = float(layout_info["gap"]) * config.min_x_span_ratio
    min_lift_margin = float(layout_info["disk_height"]) * config.min_lift_margin_disk_heights

    inferred_moves: list[dict[str, object]] = []
    for disk_index in range(num_disks):
        xs: list[float | None] = []
        ys: list[float | None] = []
        rod_assignments: list[int | None] = []
        for det in tracks[disk_index]:
            if det is None:
                xs.append(None)
                ys.append(None)
                rod_assignments.append(None)
                continue
            cx = float(det["cx"])
            cy = float(det["cy"])
            dx = np.abs(rod_xs - cx)
            best_rod = int(np.argmin(dx))
            rod = best_rod if float(dx[best_rod]) <= rod_margin else None
            xs.append(cx)
            ys.append(cy)
            rod_assignments.append(rod)

        segments = collapse_transient_segments(
            rod_segments(smooth_rods(rod_assignments, config.smooth_window)),
            num_disks,
            config,
        )
        kept_segments: list[tuple[int, int, int]] = []
        for segment_index, segment in enumerate(segments):
            seg_len = segment[1] - segment[0] + 1
            if seg_len >= config.stable_stop_len or segment_index == 0 or segment_index == len(segments) - 1:
                kept_segments.append(segment)

        for source_segment, target_segment in zip(kept_segments, kept_segments[1:]):
            if source_segment[2] == target_segment[2]:
                continue

            window_start = max(0, source_segment[1] - 1)
            window_end = min(len(xs), target_segment[0] + 2)
            window_xs = [xs[index] for index in range(window_start, window_end) if xs[index] is not None]
            window_ys = [ys[index] for index in range(window_start, window_end) if ys[index] is not None]
            if not window_xs or not window_ys:
                continue

            source_rest = high_rest_y(
                [ys[index] for index in range(source_segment[0], source_segment[1] + 1) if ys[index] is not None]
            )
            target_rest = high_rest_y(
                [ys[index] for index in range(target_segment[0], target_segment[1] + 1) if ys[index] is not None]
            )
            if source_rest is None or target_rest is None:
                continue

            if max(window_xs) - min(window_xs) < min_x_span:
                continue
            if min(window_ys) > max(source_rest, target_rest) - min_lift_margin:
                continue

            inferred_moves.append(
                {
                    "disk": disk_index,
                    "source": int(source_segment[2]),
                    "target": int(target_segment[2]),
                    "time": float(source_segment[1] + target_segment[0]) / 2.0,
                    "source_segment": [int(source_segment[0]), int(source_segment[1])],
                    "target_segment": [int(target_segment[0]), int(target_segment[1])],
                    "source_rest_y": float(source_rest),
                    "target_rest_y": float(target_rest),
                    "min_air_y": float(min(window_ys)),
                }
            )

    inferred_moves.sort(key=lambda item: float(item["time"]))
    deduped: list[dict[str, object]] = []
    for move in inferred_moves:
        if (
            deduped
            and move["disk"] == deduped[-1]["disk"]
            and move["source"] == deduped[-1]["source"]
            and move["target"] == deduped[-1]["target"]
            and float(move["time"]) - float(deduped[-1]["time"]) <= 8.0
        ):
            continue
        deduped.append(move)
    return deduped


def assignment_to_state(assignment: list[int]) -> list[list[int]]:
    num_disks = len(assignment)
    state = [[], [], []]
    for disk in range(num_disks - 1, -1, -1):
        state[assignment[disk]].append(disk)
    return state


def is_top_disk(assignment: list[int], disk: int, rod: int) -> bool:
    if assignment[disk] != rod:
        return False
    return all(assignment[smaller] != rod for smaller in range(disk))


def target_rod_accepts(assignment: list[int], disk: int, rod: int) -> bool:
    return all(assignment[smaller] != rod for smaller in range(disk))


def goal_completion_ratio(assignment: list[int], goal_rod: int) -> float:
    if not assignment:
        return 1.0
    return sum(int(rod == goal_rod) for rod in assignment) / len(assignment)


def apply_predicted_moves(
    initial_assignment: list[int],
    predicted_moves: list[dict[str, object]],
    goal_rod: int,
) -> dict[str, object]:
    assignment = list(initial_assignment)
    legal_flags: list[bool] = []
    failure_reasons: list[str | None] = []

    for move in predicted_moves:
        disk = int(move["disk"])
        source = int(move["source"])
        target = int(move["target"])
        if assignment[disk] != source:
            legal_flags.append(False)
            failure_reasons.append("source_mismatch")
            continue
        if not is_top_disk(assignment, disk, source):
            legal_flags.append(False)
            failure_reasons.append("not_top_disk")
            continue
        if not target_rod_accepts(assignment, disk, target):
            legal_flags.append(False)
            failure_reasons.append("illegal_target")
            continue
        assignment[disk] = target
        legal_flags.append(True)
        failure_reasons.append(None)

    final_state = assignment_to_state(assignment)
    goal_reached = all(rod == goal_rod for rod in assignment)
    return {
        "final_assignment": assignment,
        "final_state": final_state,
        "legal_flags": legal_flags,
        "failure_reasons": failure_reasons,
        "num_legal_moves": sum(int(flag) for flag in legal_flags),
        "goal_reached": goal_reached,
        "goal_completion_ratio": goal_completion_ratio(assignment, goal_rod),
    }


def lcs_length(seq_a: list[tuple[int, int, int]], seq_b: list[tuple[int, int, int]]) -> int:
    if not seq_a or not seq_b:
        return 0
    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i, move_a in enumerate(seq_a, start=1):
        for j, move_b in enumerate(seq_b, start=1):
            if move_a == move_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def plan_overlap_score(pred_moves: list[tuple[int, int, int]], gt_moves: list[tuple[int, int, int]]) -> float:
    if not pred_moves and not gt_moves:
        return 1.0
    if not pred_moves or not gt_moves:
        return 0.0
    lcs = lcs_length(pred_moves, gt_moves)
    return (2.0 * lcs) / (len(pred_moves) + len(gt_moves))


def evaluate_one(
    record: dict[str, object],
    prediction_path: Path | None,
    config: EvalConfig,
) -> dict[str, object]:
    result = {
        "index": int(record["index"]),
        "domain_split": str(record["domain_split"]),
        "num_disks": int(record["num_disks"]),
        "prediction_found": prediction_path is not None,
        "prediction_file": str(prediction_path) if prediction_path is not None else None,
        "decoded_prediction": False,
        "status": "missing_prediction",
        "strict_success": 0,
        "exact_plan_match": 0,
        "loose_plan_overlap": 0.0,
        "legal_move_ratio": 0.0,
        "goal_completion_ratio": 0.0,
        "goal_reached": False,
        "predicted_move_count": 0,
        "gt_move_count": int(record["num_moves"]),
        "predicted_moves": [],
        "gt_moves": list(record["move_sequence"]),
        "move_legality": [],
        "legal_failure_reasons": [],
        "final_state_from_moves": list(record["initial_state"]),
    }
    if prediction_path is None:
        return result

    try:
        tracks, video_info = decode_tracks(prediction_path, int(record["num_disks"]), config)
        inferred_moves = infer_airborne_moves(tracks, int(record["num_disks"]), int(video_info["frame_width"]), config)
    except Exception as exc:
        result["status"] = "decode_error"
        result["error"] = str(exc)
        return result

    predicted_moves = [(int(move["disk"]), int(move["source"]), int(move["target"])) for move in inferred_moves]
    gt_moves = [tuple(int(value) for value in move) for move in list(record["move_sequence"])]
    simulated = apply_predicted_moves(list(record["initial_assignment"]), inferred_moves, int(record["goal_rod"]))
    legal_move_count = int(simulated["num_legal_moves"])
    legal_move_ratio = legal_move_count / len(predicted_moves) if predicted_moves else 0.0
    strict_success = int(bool(predicted_moves) and all(simulated["legal_flags"]) and bool(simulated["goal_reached"]))
    exact_plan_match = int(predicted_moves == gt_moves)
    loose_plan_overlap = plan_overlap_score(predicted_moves, gt_moves)

    result.update(
        {
            "decoded_prediction": True,
            "status": "ok",
            "strict_success": strict_success,
            "exact_plan_match": exact_plan_match,
            "loose_plan_overlap": loose_plan_overlap,
            "legal_move_ratio": legal_move_ratio,
            "goal_completion_ratio": float(simulated["goal_completion_ratio"]),
            "goal_reached": bool(simulated["goal_reached"]),
            "predicted_move_count": len(predicted_moves),
            "predicted_moves": [list(move) for move in predicted_moves],
            "move_legality": [int(flag) for flag in simulated["legal_flags"]],
            "legal_failure_reasons": simulated["failure_reasons"],
            "final_state_from_moves": simulated["final_state"],
            "video_num_frames": int(video_info["num_frames"]),
            "video_frame_height": int(video_info["frame_height"]),
            "video_frame_width": int(video_info["frame_width"]),
            "inferred_move_details": inferred_moves,
        }
    )
    return result


def summarize_subset(results: list[dict[str, object]]) -> dict[str, object]:
    available = [item for item in results if item["decoded_prediction"]]
    strict_success_count = sum(int(item["strict_success"]) for item in available)
    exact_plan_match_count = sum(int(item["exact_plan_match"]) for item in available)
    return {
        "num_samples": len(results),
        "available_predictions": len(available),
        "strict_success_count": strict_success_count,
        "strict_success_rate": (strict_success_count / len(available)) if available else None,
        "exact_plan_match_count": exact_plan_match_count,
        "exact_plan_match_rate": (exact_plan_match_count / len(available)) if available else None,
        "average_loose_plan_overlap": (
            sum(float(item["loose_plan_overlap"]) for item in available) / len(available) if available else None
        ),
        "average_legal_move_ratio": (
            sum(float(item["legal_move_ratio"]) for item in available) / len(available) if available else None
        ),
        "average_goal_completion_ratio": (
            sum(float(item["goal_completion_ratio"]) for item in available) / len(available) if available else None
        ),
        "goal_reached_count": sum(int(item["goal_reached"]) for item in available),
        "goal_reached_rate": (
            sum(int(item["goal_reached"]) for item in available) / len(available) if available else None
        ),
        "failure_breakdown": dict(
            Counter(item["status"] for item in results if not item["decoded_prediction"])
            + Counter(
                reason
                for item in available
                if not int(item["strict_success"])
                for reason in item["legal_failure_reasons"]
                if reason is not None
            )
        ),
    }


def interval_summaries(results: list[dict[str, object]], bucket_size: int) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for start in range(0, len(results), bucket_size):
        bucket = results[start : start + bucket_size]
        if not bucket:
            continue
        first = bucket[0]
        bucket_summary = summarize_subset(bucket)
        bucket_summary.update(
            {
                "sample_range": f"{start + 1}-{start + len(bucket)}",
                "index_range": f"{start:04d}-{start + len(bucket) - 1:04d}",
                "domain_split": str(first["domain_split"]),
                "num_disks": int(first["num_disks"]),
            }
        )
        summaries.append(bucket_summary)
    return summaries


def build_summary(results: list[dict[str, object]], bucket_size: int) -> dict[str, object]:
    domain_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in results:
        domain_groups[str(item["domain_split"])].append(item)

    return {
        "num_samples": len(results),
        "available_predictions": sum(int(item["decoded_prediction"]) for item in results),
        "overall": summarize_subset(results),
        "by_domain": {domain: summarize_subset(items) for domain, items in sorted(domain_groups.items())},
        "by_bucket": interval_summaries(results, bucket_size),
    }


def default_output_path(pred_dir: Path) -> Path:
    return pred_dir / "hanoi_eval.json"


def print_summary(summary: dict[str, object]) -> None:
    overall = summary["overall"]
    print(
        f"Overall: {overall['available_predictions']}/{overall['num_samples']} available, "
        f"strict={overall['strict_success_count']}/{overall['available_predictions']} "
        f"({overall['strict_success_rate']:.4f})" if overall["available_predictions"] else
        f"Overall: 0/{overall['num_samples']} available"
    )
    if overall["available_predictions"]:
        print(
            f"Loose plan overlap={overall['average_loose_plan_overlap']:.4f}, "
            f"legal ratio={overall['average_legal_move_ratio']:.4f}, "
            f"goal ratio={overall['average_goal_completion_ratio']:.4f}"
        )
    print("By domain:")
    for domain, values in summary["by_domain"].items():
        if values["available_predictions"]:
            print(
                f"  {domain}: {values['available_predictions']}/{values['num_samples']} available, "
                f"strict={values['strict_success_count']}/{values['available_predictions']} "
                f"({values['strict_success_rate']:.4f}), "
                f"loose={values['average_loose_plan_overlap']:.4f}"
            )
        else:
            print(f"  {domain}: 0/{values['num_samples']} available")
    print("By 10-sample bucket:")
    for bucket in summary["by_bucket"]:
        if bucket["available_predictions"]:
            print(
                f"  {bucket['index_range']} ({bucket['domain_split']}, {bucket['num_disks']} disks): "
                f"{bucket['strict_success_count']}/{bucket['available_predictions']} strict "
                f"({bucket['strict_success_rate']:.4f}), "
                f"loose={bucket['average_loose_plan_overlap']:.4f}"
            )
        else:
            print(
                f"  {bucket['index_range']} ({bucket['domain_split']}, {bucket['num_disks']} disks): "
                "0 available"
            )


def main() -> None:
    args = build_parser().parse_args()
    config = EvalConfig(
        dataset_dir=args.dataset_dir,
        split=args.split,
        pred_dir=args.pred_dir,
        output_json=args.output_json,
        bucket_size=args.bucket_size,
    )

    dataset_dir = Path(config.dataset_dir)
    pred_dir = Path(config.pred_dir)
    records = load_records(dataset_dir, config.split)
    predictions = collect_prediction_paths(pred_dir)

    results: list[dict[str, object]] = []
    for record in tqdm(records, desc="Evaluating Hanoi videos", leave=True):
        prediction_path = predictions.get(int(record["index"]))
        results.append(evaluate_one(record, prediction_path, config))

    output_path = Path(config.output_json) if config.output_json else default_output_path(pred_dir)
    summary = build_summary(results, config.bucket_size)
    payload = {
        "config": {
            "dataset_dir": config.dataset_dir,
            "split": config.split,
            "pred_dir": config.pred_dir,
            "bucket_size": config.bucket_size,
        },
        "summary": summary,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print_summary(summary)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
