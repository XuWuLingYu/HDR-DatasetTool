#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import av


EXPECTED_PROMPT = "用机器人复制左边的图案"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a LIBERO drawing-copy text-to-video dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--expected-prompt", default=EXPECTED_PROMPT)
    parser.add_argument("--require-square", action="store_true", default=True)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-contiguous-ids", action="store_true")
    parser.add_argument("--require-use-weld", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_dir
    metadata_path = root / "metadata.jsonl"
    manifest_path = root / "manifest.jsonl"
    videos_dir = root / "videos"

    errors = []
    if not metadata_path.exists():
        errors.append(f"missing {metadata_path}")
    if not videos_dir.is_dir():
        errors.append(f"missing {videos_dir}")
    if errors:
        fail(errors)

    records = read_jsonl(metadata_path, errors, required_keys={"prompt", "video_path"}, exact_keys={"prompt", "video_path"})
    if args.expected_count is not None and len(records) != args.expected_count:
        errors.append(f"metadata count {len(records)} != expected {args.expected_count}")

    seen_paths = set()
    video_jobs = []
    for row, record in enumerate(records, start=1):
        prompt = record.get("prompt")
        if prompt != args.expected_prompt:
            errors.append(f"row {row}: prompt mismatch: {prompt!r}")
            if len(errors) >= args.max_errors:
                break
        video_path = Path(record.get("video_path", ""))
        if not video_path.is_absolute():
            errors.append(f"row {row}: video_path is not absolute: {video_path}")
            if len(errors) >= args.max_errors:
                break
        if str(video_path) in seen_paths:
            errors.append(f"row {row}: duplicate video_path: {video_path}")
            if len(errors) >= args.max_errors:
                break
        seen_paths.add(str(video_path))
        if not video_path.exists():
            errors.append(f"row {row}: missing video: {video_path}")
            if len(errors) >= args.max_errors:
                break
            continue
        video_jobs.append((row, video_path))

    video_infos = []
    video_info_by_path = {}
    if len(errors) < args.max_errors:
        video_infos = inspect_videos(video_jobs, workers=max(1, int(args.workers)))
        video_info_by_path = {info["path"]: info for info in video_infos if info.get("ok")}
        for info in sorted(video_infos, key=lambda item: item["row"]):
            row = info["row"]
            video_path = info["path"]
            if not info["ok"]:
                errors.append(f"row {row}: unreadable video {video_path}: {info['error']}")
            elif info["frames"] < args.min_frames:
                errors.append(f"row {row}: too few frames {info['frames']} in {video_path}")
            elif args.require_square and info["width"] != info["height"]:
                errors.append(f"row {row}: video is not square {info['width']}x{info['height']} in {video_path}")
            if len(errors) >= args.max_errors:
                break

    manifest_records = []
    if manifest_path.exists():
        manifest_records = read_jsonl(manifest_path, errors, required_keys={"id", "seed", "prompt", "video_path"}, exact_keys=None)
        validate_manifest(
            records,
            manifest_records,
            errors,
            args.require_contiguous_ids,
            args.expected_prompt,
            args.require_use_weld,
            videos_dir.resolve(),
            video_info_by_path,
        )

    video_files = {str(path.resolve()) for path in videos_dir.glob("*.mp4")}
    extra_videos = sorted(video_files - seen_paths)
    if extra_videos:
        errors.append(f"{len(extra_videos)} extra mp4 files not listed in metadata")

    if errors:
        fail(errors[: args.max_errors])

    frame_counts = [info["frames"] for info in video_infos]
    widths = sorted({info["width"] for info in video_infos})
    heights = sorted({info["height"] for info in video_infos})
    result = {
        "ok": True,
        "dataset_dir": str(root.resolve()),
        "metadata": str(metadata_path.resolve()),
        "manifest": str(manifest_path.resolve()) if manifest_path.exists() else None,
        "videos_dir": str(videos_dir.resolve()),
        "count": len(records),
        "frame_min": min(frame_counts) if frame_counts else 0,
        "frame_max": max(frame_counts) if frame_counts else 0,
        "widths": widths,
        "heights": heights,
        "workers": max(1, int(args.workers)),
        "require_contiguous_ids": bool(args.require_contiguous_ids),
        "require_use_weld": bool(args.require_use_weld),
        "prompt": args.expected_prompt,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def read_jsonl(path: Path, errors: list[str], required_keys: set[str], exact_keys: set[str] | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for row, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path} row {row}: invalid json: {exc}")
                continue
            keys = set(record.keys())
            missing = required_keys - keys
            if missing:
                errors.append(f"{path} row {row}: missing keys {sorted(missing)}")
            if exact_keys is not None and keys != exact_keys:
                errors.append(f"{path} row {row}: keys {sorted(keys)} != {sorted(exact_keys)}")
            records.append(record)
    return records


def inspect_videos(jobs: list[tuple[int, Path]], workers: int) -> list[dict]:
    if workers <= 1 or len(jobs) <= 1:
        return [inspect_video(row, path) for row, path in jobs]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(inspect_video, row, path): (row, path) for row, path in jobs}
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())
    return results


def inspect_video(row: int, path: Path) -> dict:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            frames = sum(1 for _ in container.decode(stream))
        return {"ok": True, "row": row, "path": str(path), "width": width, "height": height, "frames": frames}
    except Exception as exc:
        return {"ok": False, "row": row, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def validate_manifest(
    metadata_records: list[dict],
    manifest_records: list[dict],
    errors: list[str],
    require_contiguous: bool,
    expected_prompt: str,
    require_use_weld: bool,
    videos_dir: Path,
    video_info_by_path: dict[str, dict],
) -> None:
    if len(manifest_records) != len(metadata_records):
        errors.append(f"manifest count {len(manifest_records)} != metadata count {len(metadata_records)}")
    ids = []
    seen_ids = set()
    manifest_paths = set()
    for row, record in enumerate(manifest_records, start=1):
        try:
            idx = int(record["id"])
        except Exception:
            errors.append(f"manifest row {row}: invalid id {record.get('id')!r}")
            continue
        ids.append(idx)
        if idx in seen_ids:
            errors.append(f"manifest row {row}: duplicate id {idx}")
        seen_ids.add(idx)
        expected_name = f"drawing_copy_{idx:06d}.mp4"
        video_path = Path(str(record.get("video_path", "")))
        manifest_paths.add(str(video_path))
        try:
            resolved_video = video_path.resolve()
            resolved_video.relative_to(videos_dir)
        except Exception:
            errors.append(f"manifest row {row}: video_path is not under videos dir: {video_path}")
        if video_path.name != expected_name:
            errors.append(f"manifest row {row}: video filename {video_path.name} != {expected_name}")
        info = video_info_by_path.get(str(video_path))
        if info is not None:
            frames = record.get("frames")
            if not isinstance(frames, int) or frames != info["frames"]:
                errors.append(f"manifest row {row}: frames {frames!r} != decoded frames {info['frames']}")
        tilt = record.get("max_brush_tilt_degrees")
        if tilt is not None and (not isinstance(tilt, (int, float)) or float(tilt) > 35.0):
            errors.append(f"manifest row {row}: invalid max_brush_tilt_degrees {tilt!r}")
        if record.get("prompt") != expected_prompt:
            errors.append(f"manifest row {row}: prompt mismatch: {record.get('prompt')!r}")
        if require_use_weld:
            if record.get("use_weld") is not True:
                errors.append(f"manifest row {row}: use_weld is not true")
            if record.get("scripted_ink_mode") is not False:
                errors.append(f"manifest row {row}: scripted_ink_mode is not false")
            ink_pixels = record.get("estimated_added_ink_pixels")
            if not isinstance(ink_pixels, int) or ink_pixels < 20:
                errors.append(f"manifest row {row}: invalid estimated_added_ink_pixels {ink_pixels!r}")
            draw_segments = record.get("draw_segment_count")
            if not isinstance(draw_segments, int) or draw_segments < 20:
                errors.append(f"manifest row {row}: invalid draw_segment_count {draw_segments!r}")
            elif draw_segments >= 1024:
                errors.append(f"manifest row {row}: draw_segment_count reached capacity and may be truncated: {draw_segments!r}")
            canvas_pixels = record.get("draw_canvas_dark_pixels")
            if not isinstance(canvas_pixels, int) or canvas_pixels < 500:
                errors.append(f"manifest row {row}: invalid draw_canvas_dark_pixels {canvas_pixels!r}")
            physics = record.get("physics_metrics")
            if not isinstance(physics, dict):
                errors.append(f"manifest row {row}: missing physics_metrics")
            else:
                required_physics = {
                    "pre_attach_max_brush_move_m": (0.0, 0.003),
                    "post_attach_max_brush_move_m": (0.01, None),
                    "attach_site_distance_m": (0.0, 0.05),
                    "draw_tip_z_rel_mean_m": (-0.016, 0.002),
                    "draw_tip_z_rel_p99_m": (None, 0.015),
                    "draw_tip_contact_fraction": (0.95, None),
                    "draw_tip_xy_error_max_norm_m": (0.0, 0.06),
                }
                for key, (lo, hi) in required_physics.items():
                    value = physics.get(key)
                    if not isinstance(value, (int, float)):
                        errors.append(f"manifest row {row}: missing physics metric {key}")
                        continue
                    if lo is not None and float(value) < lo:
                        errors.append(f"manifest row {row}: physics metric {key}={value} < {lo}")
                    if hi is not None and float(value) > hi:
                        errors.append(f"manifest row {row}: physics metric {key}={value} > {hi}")
        if row <= len(metadata_records):
            meta_path = metadata_records[row - 1].get("video_path")
            if meta_path != record.get("video_path"):
                errors.append(f"row {row}: metadata video_path does not match manifest video_path")
    metadata_paths = {record.get("video_path") for record in metadata_records}
    missing_from_manifest = sorted(metadata_paths - manifest_paths)
    if missing_from_manifest:
        errors.append(f"{len(missing_from_manifest)} metadata videos missing from manifest")
    extra_in_manifest = sorted(manifest_paths - metadata_paths)
    if extra_in_manifest:
        errors.append(f"{len(extra_in_manifest)} manifest videos missing from metadata")
    if ids != sorted(ids):
        errors.append("manifest ids are not sorted")
    if require_contiguous and ids:
        expected_ids = list(range(len(ids)))
        if ids != expected_ids:
            errors.append(f"manifest ids are not contiguous 0..{len(ids) - 1}")


def fail(errors: list[str]) -> None:
    print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
