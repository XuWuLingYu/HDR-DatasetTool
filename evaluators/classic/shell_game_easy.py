#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


INDEX_PATTERNS = [
    re.compile(r"^(\d+)"),
    re.compile(r"^sample_(\d+)"),
]


@dataclass(frozen=True)
class EvalConfig:
    dataset_dir: str = "outputs/shell_game_easy"
    split: str = "test"
    pred_dir: str = "outputs/predictions/shell_game_easy"
    output_json: str | None = None
    tail_frames: int = 8
    bucket_size: int = 10
    window_ratio: float = 0.10
    dark_threshold: int = 70
    min_dark_pixels: int = 120
    lift_margin_ratio: float = 0.04
    red_r_min: int = 150
    red_delta_min: int = 60
    red_g_max: int = 140
    red_b_max: int = 140
    min_red_pixels: int = 20
    red_margin_pixels: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate shell_game_easy predictions against the test split.")
    parser.add_argument("--dataset-dir", type=str, default="outputs/shell_game_easy")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="outputs/predictions/shell_game_easy",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--tail-frames", type=int, default=8)
    parser.add_argument("--bucket-size", type=int, default=10)
    return parser


def load_test_records(dataset_dir: Path, split: str) -> list[dict[str, object]]:
    metadata_path = dataset_dir / split / "metadata.jsonl"
    with metadata_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_slot_xs(dataset_dir: Path, split: str, npy_file: str) -> np.ndarray:
    npy_path = dataset_dir / split / "npy" / npy_file
    payload = np.load(npy_path, allow_pickle=True).item()
    return np.asarray(payload["slot_xs"], dtype=np.float32)


def collect_prediction_paths(pred_dir: Path) -> dict[int, Path]:
    predictions: dict[int, Path] = {}
    for path in sorted(pred_dir.glob("*.mp4")):
        match = None
        for pattern in INDEX_PATTERNS:
            match = pattern.match(path.name)
            if match is not None:
                break
        if match is None:
            continue
        predictions[int(match.group(1))] = path
    return predictions


def slot_windows(slot_xs_scaled: np.ndarray, frame_width: int, window_ratio: float) -> list[tuple[int, int]]:
    if len(slot_xs_scaled) > 1:
        min_gap = float(np.min(np.diff(np.sort(slot_xs_scaled))))
    else:
        min_gap = frame_width * 0.25
    half_width = min(int(round(frame_width * window_ratio)), int(round(0.45 * min_gap)))
    half_width = max(6, half_width)
    windows: list[tuple[int, int]] = []
    for sx in slot_xs_scaled:
        left = max(0, int(round(float(sx) - half_width)))
        right = min(frame_width, int(round(float(sx) + half_width)))
        windows.append((left, right))
    return windows


def analyze_frame(frame: np.ndarray, windows: list[tuple[int, int]], config: EvalConfig) -> tuple[list[int], list[int], list[int]]:
    frame = np.asarray(frame)
    red_channel = frame[..., 0].astype(np.int16)
    green_channel = frame[..., 1].astype(np.int16)
    blue_channel = frame[..., 2].astype(np.int16)

    dark_mask = np.max(frame, axis=2) < config.dark_threshold
    red_mask = (
        (red_channel > config.red_r_min)
        & (red_channel - green_channel > config.red_delta_min)
        & (red_channel - blue_channel > config.red_delta_min)
        & (green_channel < config.red_g_max)
        & (blue_channel < config.red_b_max)
    )

    top_ys: list[int] = []
    dark_counts: list[int] = []
    red_counts: list[int] = []
    for left, right in windows:
        dark_crop = dark_mask[:, left:right]
        red_crop = red_mask[:, left:right]
        dark_count = int(dark_crop.sum())
        red_count = int(red_crop.sum())
        if dark_count > 0:
            top_y = int(np.argwhere(dark_crop)[:, 0].min())
        else:
            top_y = frame.shape[0] + 1
        top_ys.append(top_y)
        dark_counts.append(dark_count)
        red_counts.append(red_count)
    return top_ys, dark_counts, red_counts


def majority_vote(votes: list[int]) -> int | None:
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def infer_lifted_slot(frames: list[np.ndarray], windows: list[tuple[int, int]], config: EvalConfig) -> tuple[int | None, list[int]]:
    votes: list[int] = []
    for frame in frames:
        top_ys, dark_counts, _ = analyze_frame(frame, windows, config)
        valid_slots = [idx for idx, count in enumerate(dark_counts) if count >= config.min_dark_pixels]
        if len(valid_slots) < 2:
            continue
        ranked = sorted(((top_ys[idx], idx) for idx in valid_slots), key=lambda item: item[0])
        best_top, best_idx = ranked[0]
        second_top = ranked[1][0]
        if second_top - best_top >= max(8, int(round(config.lift_margin_ratio * frame.shape[0]))):
            votes.append(best_idx)
    return majority_vote(votes), votes


def infer_red_slot(frames: list[np.ndarray], windows: list[tuple[int, int]], config: EvalConfig) -> tuple[int | None, list[int]]:
    votes: list[int] = []
    for frame in frames:
        _, _, red_counts = analyze_frame(frame, windows, config)
        ranked = sorted(((count, idx) for idx, count in enumerate(red_counts)), reverse=True)
        best_count, best_idx = ranked[0]
        second_count = ranked[1][0] if len(ranked) > 1 else 0
        if best_count >= config.min_red_pixels and best_count - second_count >= config.red_margin_pixels:
            votes.append(best_idx)
    return majority_vote(votes), votes


def decode_tail_frames(video_path: Path, tail_frames: int) -> tuple[list[np.ndarray], dict[str, int]]:
    reader = imageio.get_reader(video_path)
    try:
        num_frames = reader.count_frames()
        start = max(0, num_frames - tail_frames)
        frames = [reader.get_data(index) for index in range(start, num_frames)]
        frame_height, frame_width = frames[-1].shape[:2]
        return frames, {
            "num_frames": num_frames,
            "frame_height": frame_height,
            "frame_width": frame_width,
        }
    finally:
        reader.close()


def evaluate_one(
    record: dict[str, object],
    prediction_path: Path | None,
    dataset_dir: Path,
    split: str,
    config: EvalConfig,
) -> dict[str, object]:
    result = {
        "index": int(record["index"]),
        "num_swaps": int(record["num_swaps"]),
        "gt_slot": int(record["final_ball_slot"]),
        "prediction_found": prediction_path is not None,
        "decoded_prediction": False,
        "prediction_file": str(prediction_path) if prediction_path is not None else None,
        "predicted_slot": None,
        "lifted_slot": None,
        "red_slot": None,
        "is_correct": False,
        "status": "missing_prediction",
    }
    if prediction_path is None:
        return result

    try:
        frames, video_info = decode_tail_frames(prediction_path, config.tail_frames)
    except Exception as exc:
        result["status"] = "decode_error"
        result["error"] = str(exc)
        return result

    slot_xs = load_slot_xs(dataset_dir, split, str(record["npy_file"]))
    slot_xs_scaled = slot_xs * (video_info["frame_width"] / float(record["frame_width"]))
    windows = slot_windows(slot_xs_scaled, video_info["frame_width"], config.window_ratio)

    lifted_slot, lifted_votes = infer_lifted_slot(frames, windows, config)
    red_slot, red_votes = infer_red_slot(frames, windows, config)

    predicted_slot: int | None
    status: str
    if lifted_slot is not None and red_slot is not None:
        if lifted_slot == red_slot:
            predicted_slot = lifted_slot
            status = "ok"
        else:
            predicted_slot = None
            status = "slot_conflict"
    elif lifted_slot is not None:
        predicted_slot = lifted_slot
        status = "lift_only"
    elif red_slot is not None:
        predicted_slot = red_slot
        status = "red_only"
    else:
        predicted_slot = None
        status = "no_signal"

    result.update(
        {
            "decoded_prediction": True,
            "predicted_slot": predicted_slot,
            "lifted_slot": lifted_slot,
            "red_slot": red_slot,
            "is_correct": predicted_slot == int(record["final_ball_slot"]),
            "status": status,
            "video_num_frames": int(video_info["num_frames"]),
            "video_frame_height": int(video_info["frame_height"]),
            "video_frame_width": int(video_info["frame_width"]),
            "lifted_votes": lifted_votes,
            "red_votes": red_votes,
        }
    )
    return result


def interval_summaries(results: list[dict[str, object]], bucket_size: int) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for offset in range(0, len(results), bucket_size):
        bucket = results[offset : offset + bucket_size]
        decoded = [item for item in bucket if item["decoded_prediction"]]
        correct = [item for item in decoded if item["is_correct"]]
        summaries.append(
            {
                "sample_range": f"{offset + 1}-{offset + len(bucket)}",
                "index_range": f"{offset:04d}-{offset + len(bucket) - 1:04d}",
                "decoded_predictions": len(decoded),
                "correct_predictions": len(correct),
                "accuracy": (len(correct) / len(decoded)) if decoded else None,
            }
        )
    return summaries


def build_summary(results: list[dict[str, object]], bucket_size: int) -> dict[str, object]:
    found = sum(1 for item in results if item["prediction_found"])
    decoded = [item for item in results if item["decoded_prediction"]]
    correct = [item for item in decoded if item["is_correct"]]
    return {
        "total_test_samples": len(results),
        "found_predictions": found,
        "decoded_predictions": len(decoded),
        "correct_predictions": len(correct),
        "total_accuracy": (len(correct) / len(decoded)) if decoded else None,
        "interval_accuracy": interval_summaries(results, bucket_size),
    }


def print_report(summary: dict[str, object], results: list[dict[str, object]]) -> None:
    print(
        f"Decoded {summary['decoded_predictions']} / {summary['total_test_samples']} predictions, "
        f"correct {summary['correct_predictions']}, total_accuracy={summary['total_accuracy']}"
    )
    print("Interval accuracy:")
    for item in summary["interval_accuracy"]:
        print(
            f"  {item['sample_range']}: "
            f"{item['correct_predictions']} / {item['decoded_predictions']} = {item['accuracy']}"
        )
    print("Per-sample results:")
    for item in results:
        print(
            f"  idx={item['index']:04d} "
            f"gt={item['gt_slot']} pred={item['predicted_slot']} "
            f"correct={item['is_correct']} status={item['status']}"
        )


def main() -> None:
    args = build_parser().parse_args()
    config = EvalConfig(
        dataset_dir=args.dataset_dir,
        split=args.split,
        pred_dir=args.pred_dir,
        output_json=args.output_json,
        tail_frames=args.tail_frames,
        bucket_size=args.bucket_size,
    )

    dataset_dir = Path(config.dataset_dir)
    pred_dir = Path(config.pred_dir)
    output_json = Path(config.output_json) if config.output_json else pred_dir / "shell_game_easy_eval.json"

    records = load_test_records(dataset_dir, config.split)
    prediction_paths = collect_prediction_paths(pred_dir)
    results = [
        evaluate_one(record, prediction_paths.get(int(record["index"])), dataset_dir, config.split, config)
        for record in records
    ]
    summary = build_summary(results, config.bucket_size)
    payload = {
        "config": asdict(config),
        "summary": summary,
        "results": results,
    }

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary, results)
    print(f"Saved report to {output_json}")


if __name__ == "__main__":
    main()
