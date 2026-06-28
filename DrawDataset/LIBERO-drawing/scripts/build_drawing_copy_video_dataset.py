#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path(sys.executable)
PROMPT = "用机器人复制左边的图案"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a text-to-video LIBERO drawing-copy dataset.")
    parser.add_argument("--out", type=Path, default=Path("outputs/drawing_copy_video_dataset"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--control-fps", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--speed-scale", type=float, default=0.08)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--use-weld", action="store_true", help="Use physical weld mode instead of scripted brush following.")
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--keep-episodes", action="store_true", help="Keep per-sample HDF5, frames, previews, and summaries.")
    parser.add_argument("--final-validate", action="store_true")
    parser.add_argument("--no-final-validate", dest="final_validate", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(final_validate=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_retries is None:
        args.max_retries = 8 if args.use_weld else 2
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")

    out_dir = args.out.resolve()
    videos_dir = out_dir / "videos"
    episodes_dir = out_dir / "episodes"
    metadata_path = out_dir / "metadata.jsonl"
    manifest_path = out_dir / "manifest.jsonl"
    failures_path = out_dir / "failures.jsonl"

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_manifest(manifest_path, metadata_path, require_physical_audit=args.use_weld)
    jobs = []
    for idx in range(args.num_samples):
        if idx % args.num_shards != args.shard_index:
            continue
        video_name = f"drawing_copy_{idx:06d}.mp4"
        if idx in existing and (videos_dir / video_name).exists():
            continue
        jobs.append((idx, args.seed_start + idx))
    shard_expected_count = count_shard_items(args.num_samples, args.num_shards, args.shard_index)

    results = []
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_one, idx, seed, args, out_dir): (idx, seed) for idx, seed in jobs}
        for future in concurrent.futures.as_completed(future_map):
            idx, seed = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ok": False,
                    "id": idx,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if result.get("ok"):
                results.append(result)
                append_jsonl(manifest_path, manifest_record(result))
                rewrite_outputs_sorted(metadata_path, manifest_path)
            else:
                failures.append(result)
                append_jsonl(failures_path, result)
            done = len(results) + len(failures)
            print(json.dumps({"done": done, "ok": len(results), "failed": len(failures), "last": result}, ensure_ascii=False), flush=True)

    summary = {
        "ok": len(failures) == 0,
        "out": str(out_dir),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "videos": str(videos_dir),
        "episodes": str(episodes_dir) if args.keep_episodes else None,
        "requested": args.num_samples,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_expected_count": shard_expected_count,
        "new_jobs": len(jobs),
        "new_successes": len(results),
        "new_failures": len(failures),
        "prompt": PROMPT,
        "use_weld": args.use_weld,
    }
    rewrite_outputs_sorted(metadata_path, manifest_path)
    validation = None
    if args.final_validate:
        validation = run_final_validator(
            out_dir,
            shard_expected_count,
            workers=args.workers,
            require_contiguous_ids=args.num_shards == 1,
            require_use_weld=args.use_weld,
        )
        summary["validation"] = validation
        if not validation.get("ok", False):
            summary["ok"] = False
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures or (validation is not None and not validation.get("ok", False)):
        raise SystemExit(1)


def run_one(idx: int, seed: int, args: argparse.Namespace, out_dir: Path) -> dict:
    videos_dir = out_dir / "videos"
    episodes_dir = out_dir / "episodes"
    final_video = videos_dir / f"drawing_copy_{idx:06d}.mp4"
    final_episode = episodes_dir / f"drawing_copy_{idx:06d}"

    for attempt in range(args.max_retries + 1):
        seed_attempt = seed_attempt_for_attempt(attempt)
        z_attempt = z_attempt_for_attempt(attempt)
        current_seed = sample_seed(seed, seed_attempt)
        draw_z_lift = draw_z_lift_for_z_attempt(z_attempt) if args.use_weld else None
        work_parent = episodes_dir if args.keep_episodes else Path(tempfile.mkdtemp(prefix=f"drawing_copy_{idx:06d}_", dir=str(out_dir)))
        episode_dir = final_episode if args.keep_episodes else work_parent / "episode"
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        cmd = [
            str(PYTHON_BIN),
            str(ROOT / "scripts" / "collect_drawing_copy_dataset.py"),
            "--out",
            str(episode_dir),
            "--seed",
            str(current_seed),
            "--camera-size",
            str(args.camera_size),
            "--fps",
            str(args.fps),
            "--control-fps",
            str(args.control_fps),
            "--stride",
            str(args.stride),
            "--record-every",
            str(args.record_every),
            "--speed-scale",
            str(args.speed_scale),
            "--horizon",
            str(args.horizon),
        ]
        if args.use_weld:
            cmd.append("--use-weld")
            cmd.extend(["--draw-z-lift", str(draw_z_lift)])
        collect = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if collect.returncode == 0:
            verify_cmd = [
                str(PYTHON_BIN),
                str(ROOT / "scripts" / "verify_drawing_copy_episode.py"),
                str(episode_dir),
            ]
            verify = subprocess.run(verify_cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            verify = None

        if collect.returncode == 0 and verify is not None and verify.returncode == 0:
            atomic_copy(episode_dir / "episode.mp4", final_video)
            summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
            verification = parse_json_from_output(verify.stdout)
            if not args.keep_episodes:
                shutil.rmtree(work_parent, ignore_errors=True)
            return {
                "ok": True,
                "id": idx,
                "seed": current_seed,
                "attempt": attempt,
                "seed_attempt": seed_attempt,
                "z_attempt": z_attempt,
                "video_path": str(final_video.resolve()),
                "episode_dir": str(final_episode.resolve()) if args.keep_episodes else None,
                "frames": summary.get("frames"),
                "max_brush_tilt_degrees": summary.get("max_brush_tilt_degrees"),
                "use_weld": summary.get("use_weld"),
                "scripted_ink_mode": summary.get("scripted_ink_mode"),
                "draw_z_lift": summary.get("draw_z_lift"),
                "estimated_added_ink_pixels": verification.get("estimated_added_ink_pixels"),
                "draw_segment_count": verification.get("draw_segment_count"),
                "draw_canvas_dark_pixels": verification.get("draw_canvas_dark_pixels"),
                "draw_ee_z_range": verification.get("draw_ee_z_range"),
                "physics_metrics": verification.get("physics_metrics"),
                "phase_counts": verification.get("phase_counts"),
            }

        if not args.keep_episodes:
            shutil.rmtree(work_parent, ignore_errors=True)

    return {
        "ok": False,
        "id": idx,
        "seed": seed,
        "error": "collector or verifier failed after retries",
        "last_episode_dir": str(episode_dir.resolve()) if "episode_dir" in locals() else None,
        "collector_tail": tail_text(collect.stdout if "collect" in locals() else ""),
        "verifier_tail": tail_text(verify.stdout if verify is not None else ""),
    }


def sample_seed(base_seed: int, attempt: int) -> int:
    if attempt == 0:
        return int(base_seed)
    return int(base_seed + attempt * 9973 + attempt * attempt * 7919)


def draw_z_lift_schedule() -> list[float]:
    return [0.0, -0.003, 0.003]


def seed_attempt_for_attempt(attempt: int) -> int:
    return int(attempt // len(draw_z_lift_schedule()))


def z_attempt_for_attempt(attempt: int) -> int:
    return int(attempt % len(draw_z_lift_schedule()))


def draw_z_lift_for_z_attempt(z_attempt: int) -> float:
    schedule = draw_z_lift_schedule()
    return float(schedule[z_attempt % len(schedule)])


def count_shard_items(num_samples: int, num_shards: int, shard_index: int) -> int:
    if shard_index >= num_samples:
        return 0
    return ((num_samples - 1 - shard_index) // num_shards) + 1


def metadata_record(result: dict) -> dict:
    return {
        "prompt": PROMPT,
        "video_path": result["video_path"],
    }


def manifest_record(result: dict) -> dict:
    return {
        "id": result["id"],
        "seed": result["seed"],
        "prompt": PROMPT,
        "video_path": result["video_path"],
        "frames": result.get("frames"),
        "max_brush_tilt_degrees": result.get("max_brush_tilt_degrees"),
        "use_weld": result.get("use_weld"),
        "scripted_ink_mode": result.get("scripted_ink_mode"),
        "draw_z_lift": result.get("draw_z_lift"),
        "estimated_added_ink_pixels": result.get("estimated_added_ink_pixels"),
        "draw_segment_count": result.get("draw_segment_count"),
        "draw_canvas_dark_pixels": result.get("draw_canvas_dark_pixels"),
        "draw_ee_z_range": result.get("draw_ee_z_range"),
        "physics_metrics": result.get("physics_metrics"),
        "phase_counts": result.get("phase_counts"),
    }


def load_existing_manifest(path: Path, metadata_path: Path, require_physical_audit: bool = False) -> set[int]:
    existing = set()
    sources = [path] if require_physical_audit else [path, metadata_path]
    for source in sources:
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if require_physical_audit and not has_physical_audit(record):
                    continue
                if "id" in record:
                    existing.add(int(record["id"]))
                elif "video_path" in record:
                    parsed = parse_video_id(Path(record["video_path"]).name)
                    if parsed is not None:
                        existing.add(parsed)
    return existing


def has_physical_audit(record: dict) -> bool:
    if record.get("use_weld") is not True or record.get("scripted_ink_mode") is not False:
        return False
    if not isinstance(record.get("estimated_added_ink_pixels"), int):
        return False
    if not isinstance(record.get("draw_segment_count"), int):
        return False
    if not isinstance(record.get("draw_canvas_dark_pixels"), int):
        return False
    physics = record.get("physics_metrics")
    if not isinstance(physics, dict):
        return False
    required = [
        "pre_attach_max_brush_move_m",
        "post_attach_max_brush_move_m",
        "attach_site_distance_m",
        "gripper_brush_contact_frames_near_attach",
        "draw_tip_z_rel_mean_m",
        "draw_tip_xy_error_max_norm_m",
    ]
    return all(key in physics for key in required)


def parse_video_id(name: str) -> int | None:
    stem = Path(name).stem
    prefix = "drawing_copy_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_outputs_sorted(metadata_path: Path, manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    by_id = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            by_id[int(record["id"])] = record
    records = [by_id[idx] for idx in sorted(by_id)]
    write_jsonl_atomic(manifest_path, records)
    write_jsonl_atomic(metadata_path, [metadata_record(record) for record in records])


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def run_final_validator(out_dir: Path, expected_count: int, workers: int, require_contiguous_ids: bool, require_use_weld: bool) -> dict:
    cmd = [
        str(PYTHON_BIN),
        str(ROOT / "scripts" / "validate_drawing_copy_video_dataset.py"),
        str(out_dir),
        "--expected-count",
        str(expected_count),
        "--workers",
        str(max(1, int(workers))),
    ]
    if require_contiguous_ids:
        cmd.append("--require-contiguous-ids")
    if require_use_weld:
        cmd.append("--require-use-weld")
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"ok": False, "error": "validator returned non-json output", "output": tail_text(proc.stdout)}
    result["returncode"] = proc.returncode
    return result


def tail_text(text: str, n: int = 2400) -> str:
    return text[-n:] if len(text) > n else text


def parse_json_from_output(text: str) -> dict:
    decoder = json.JSONDecoder()
    fallback = None
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            fallback = obj
            if obj.get("ok") is True and "episode_dir" in obj and "physics_metrics" in obj:
                return obj
    if fallback is None:
        raise ValueError(f"could not parse verifier JSON output: {tail_text(text)}")
    return fallback


if __name__ == "__main__":
    main()
