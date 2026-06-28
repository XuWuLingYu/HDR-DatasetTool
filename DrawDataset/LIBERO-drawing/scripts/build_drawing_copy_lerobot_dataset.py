#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


LEROBOT_V21_SOURCE = Path(
    "/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/RICL/.uv_cache/archive-v0/vRRWyXWO4hY-NIBr"
)
if LEROBOT_V21_SOURCE.exists():
    sys.path.insert(0, str(LEROBOT_V21_SOURCE))
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path(sys.executable)
PROMPT = "Copy the pattern on the left board with the robot."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LeRobot-format LIBERO drawing-copy dataset.")
    parser.add_argument("--out", type=Path, default=Path("outputs/drawing_copy_lerobot_1k"))
    parser.add_argument("--repo-id", default="local/drawing_copy_lerobot")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--control-fps", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--speed-scale", type=float, default=0.08)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=2200)
    parser.add_argument("--max-recorded-frames", type=int, default=0)
    parser.add_argument("--max-draw-frames", type=int, default=1200)
    parser.add_argument("--max-copy-uv-mean", type=float, default=0.025)
    parser.add_argument("--max-copy-uv-p95", type=float, default=0.06)
    parser.add_argument("--min-target-dark-pixels", type=int, default=140)
    parser.add_argument("--min-target-reference-dark-ratio", type=float, default=0.35)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum candidate ids to try before giving up. 0 means num_samples * 20.",
    )
    parser.add_argument("--video-codec", default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--video-crf", type=int, default=18, help="Lower is higher quality. LeRobot default is 30.")
    parser.add_argument("--video-gop", type=int, default=2)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output dataset.")
    parser.add_argument("--keep-episodes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out.resolve()
    configure_lerobot_video_codec(args.video_codec, args.video_crf, args.video_gop)
    episodes_dir = out_dir / "source_episodes"
    replay_dir = out_dir / "replay"
    staging_dir = out_dir / "_staging"
    manifest_path = out_dir / "manifest.jsonl"
    failures_path = out_dir / "failures.jsonl"

    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume cannot be used together")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = out_dir / "lerobot"
    feature_spec = {
        "observation.images.agentview": {"dtype": "video", "shape": (3, args.camera_size, args.camera_size), "names": ["channel", "height", "width"]},
        "observation.images.wrist": {"dtype": "video", "shape": (3, args.camera_size, args.camera_size), "names": ["channel", "height", "width"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["ee_delta_x", "ee_delta_y", "ee_delta_z", "ee_delta_roll", "ee_delta_pitch", "ee_delta_yaw", "gripper_abs"]},
    }
    if args.resume and (dataset_root / "meta" / "info.json").is_file():
        ds = open_lerobot_dataset_for_recording_resume(
            args.repo_id,
            root=dataset_root,
            video_backend="pyav",
        )
        if int(ds.fps) != int(args.fps):
            raise SystemExit(f"cannot resume: existing fps={ds.fps}, requested fps={args.fps}")
        expected_shape = [3, args.camera_size, args.camera_size]
        for key in ("observation.images.agentview", "observation.images.wrist"):
            actual_shape = list(ds.features[key]["shape"])
            if actual_shape != expected_shape:
                raise SystemExit(f"cannot resume: existing {key} shape={actual_shape}, requested={expected_shape}")
        ds.episode_buffer = ds.create_episode_buffer()
    else:
        if args.resume:
            print(f"resume requested but no existing dataset found at {dataset_root}; creating a new one", flush=True)
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        ds = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=dataset_root,
            fps=args.fps,
            robot_type="libero_franka_drawing_copy",
            use_videos=True,
            video_backend="pyav",
            image_writer_processes=0,
            image_writer_threads=4,
            features=feature_spec,
        )

    existing_written = read_jsonl(manifest_path) if args.resume else []
    existing_failures = read_jsonl(failures_path) if args.resume else []
    if existing_written and len(existing_written) != int(ds.meta.total_episodes):
        raise SystemExit(
            f"cannot resume: manifest rows={len(existing_written)} but LeRobot episodes={ds.meta.total_episodes}"
        )

    written = []
    failures: list[dict] = []
    max_candidates = args.max_candidates if args.max_candidates > 0 else max(args.num_samples * 20, args.num_samples + args.workers)
    used_candidate_ids = []
    for record in existing_written + existing_failures:
        try:
            used_candidate_ids.append(int(record["id"]))
        except Exception:
            pass
    next_candidate_idx = max(used_candidate_ids, default=-1) + 1 if args.resume else 0
    if current_episode_count(ds) < args.num_samples:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {}

            def submit_next() -> bool:
                nonlocal next_candidate_idx
                if next_candidate_idx >= max_candidates:
                    return False
                idx = next_candidate_idx
                seed = args.seed_start + idx
                future_map[executor.submit(collect_one, idx, seed, args, staging_dir)] = (idx, seed)
                next_candidate_idx += 1
                return True

            for _ in range(min(max(1, args.workers), max_candidates)):
                submit_next()

            while future_map and current_episode_count(ds) < args.num_samples:
                done, _ = concurrent.futures.wait(future_map, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    idx, seed = future_map.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"ok": False, "id": idx, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                    if result.get("ok") and current_episode_count(ds) < args.num_samples:
                        episode_index = current_episode_count(ds)
                        frames = write_episode_to_lerobot(ds, result)
                        written_record = {
                            **result,
                            "episode_index": episode_index,
                            "frames": int(frames),
                        }
                        if args.keep_episodes:
                            dst = episodes_dir / f"episode_{result['id']:06d}.hdf5"
                            shutil.copy2(result["episode_hdf5"], dst)
                            written_record["kept_episode_hdf5"] = str(dst.resolve())
                        replay_npz = replay_dir / f"episode_{episode_index:06d}.npz"
                        write_replay_sidecar(Path(result["episode_hdf5"]), replay_npz)
                        written_record["replay_npz"] = str(replay_npz.resolve())
                        cleanup_result(result)
                        written.append(written_record)
                        append_jsonl(manifest_path, written_record)
                    elif result.get("ok"):
                        cleanup_result(result)
                    else:
                        failures.append(result)
                        append_jsonl(failures_path, result)
                    print(
                        json.dumps(
                            {
                                "written_episodes": current_episode_count(ds),
                                "session_written": len(written),
                                "failed": len(existing_failures) + len(failures),
                                "session_failed": len(failures),
                                "last": result,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if current_episode_count(ds) < args.num_samples:
                        submit_next()
                if current_episode_count(ds) >= args.num_samples:
                    for pending_future in future_map:
                        pending_future.cancel()
                    future_map.clear()

    shutil.rmtree(staging_dir, ignore_errors=True)

    summary = {
        "ok": current_episode_count(ds) >= args.num_samples,
        "out": str(out_dir),
        "lerobot_root": str(dataset_root),
        "repo_id": args.repo_id,
        "codebase_version": CODEBASE_VERSION,
        "num_episodes": current_episode_count(ds),
        "session_episodes": len(written),
        "initial_episodes": len(existing_written),
        "num_failures": len(existing_failures) + len(failures),
        "session_failures": len(failures),
        "num_candidates_tried": next_candidate_idx,
        "max_candidates": max_candidates,
        "fps": args.fps,
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "video_gop": args.video_gop,
        "prompt": PROMPT,
        "features": ["observation.images.agentview", "observation.images.wrist", "action"],
        "action_schema": "6D controller ee delta + 1D absolute gripper",
        "action_frame_alignment": "post_step_frame_matches_action_t_result",
        "replay_sidecars": str(replay_dir),
        "manifest": str(manifest_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if current_episode_count(ds) < args.num_samples:
        raise SystemExit(
            f"only collected {current_episode_count(ds)} / {args.num_samples} valid episodes after "
            f"{next_candidate_idx} candidates; see {failures_path}"
        )


def current_episode_count(ds: LeRobotDataset) -> int:
    return int(ds.meta.total_episodes)


def open_lerobot_dataset_for_recording_resume(
    repo_id: str,
    root: Path,
    video_backend: str = "pyav",
) -> LeRobotDataset:
    """Open local LeRobot metadata for appending without loading prior parquet rows.

    The v2.1 loader can fail with newer `datasets` because it expects transformed
    timestamp columns to already be tensors. For recording, we only need metadata,
    an empty in-memory HF dataset, and the normal save_episode path.
    """
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.repo_id = repo_id
    obj.root = Path(root)
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.episodes = None
    obj.tolerance_s = 1e-4
    obj.revision = CODEBASE_VERSION
    obj.video_backend = video_backend
    obj.delta_indices = None
    obj.image_writer = None
    obj.episode_buffer = None
    obj.meta = LeRobotDatasetMetadata(repo_id, root=obj.root)
    obj.episode_data_index = None
    obj.hf_dataset = obj.create_hf_dataset()
    return obj


def write_episode_to_lerobot(ds: LeRobotDataset, result: dict) -> int:
    episode_h5 = Path(result["episode_hdf5"])
    with h5py.File(episode_h5, "r") as f:
        agent_frames = f["frames"][()]
        wrist_frames = f["wrist_frames"][()]
        actions = f["actions"][()].astype(np.float32)
    if not (len(agent_frames) == len(wrist_frames) == len(actions)):
        raise RuntimeError(f"length mismatch for episode {result['id']}")
    for frame_idx in range(len(actions)):
        ds.add_frame(
            {
                "observation.images.agentview": agent_frames[frame_idx],
                "observation.images.wrist": wrist_frames[frame_idx],
                "action": actions[frame_idx],
                "task": PROMPT,
            }
        )
    ds.save_episode()
    return int(len(actions))


def write_replay_sidecar(episode_h5: Path, out_path: Path) -> None:
    with h5py.File(episode_h5, "r") as f:
        actions = f["actions"][()].astype(np.float32)
        phases = f["phases"][()].astype(str) if "phases" in f else np.asarray([""] * len(actions), dtype=str)
        initial_sim_state = f["initial_sim_state"][()].astype(np.float64) if "initial_sim_state" in f else np.asarray([], dtype=np.float64)
        seed = int(f.attrs.get("seed", -1))
        fps = int(f.attrs.get("fps", 16))
        control_fps = int(f.attrs.get("control_fps", fps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        actions=actions,
        phases=phases,
        initial_sim_state=initial_sim_state,
        seed=np.asarray(seed, dtype=np.int64),
        fps=np.asarray(fps, dtype=np.int64),
        control_fps=np.asarray(control_fps, dtype=np.int64),
    )


def configure_lerobot_video_codec(video_codec: str, video_crf: int, video_gop: int) -> None:
    original = lerobot_dataset_module.encode_video_frames

    def encode_video_frames_with_codec(imgs_dir, video_path, fps, *args, **kwargs):
        kwargs["vcodec"] = video_codec
        kwargs["crf"] = int(video_crf)
        kwargs["g"] = int(video_gop)
        return original(imgs_dir, video_path, fps, *args, **kwargs)

    lerobot_dataset_module.encode_video_frames = encode_video_frames_with_codec


def cleanup_result(result: dict) -> None:
    work_parent = result.get("work_parent")
    if work_parent:
        shutil.rmtree(work_parent, ignore_errors=True)


def collect_one(idx: int, seed: int, args: argparse.Namespace, staging_dir: Path) -> dict:
    for attempt in range(args.max_retries + 1):
        current_seed = seed if attempt == 0 else seed + attempt * 9973 + attempt * attempt * 7919
        work_parent = Path(tempfile.mkdtemp(prefix=f"episode_{idx:06d}_", dir=str(staging_dir)))
        episode_dir = work_parent / "episode"
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
            "--max-recorded-frames",
            str(args.max_recorded_frames if args.max_recorded_frames > 0 else args.max_frames),
            "--use-weld",
            "--skip-debug-media",
        ]
        started = time.time()
        collect = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        collect_seconds = time.time() - started
        if collect.returncode == 0:
            verify_cmd = [
                str(PYTHON_BIN),
                str(ROOT / "scripts" / "verify_drawing_copy_episode.py"),
                str(episode_dir),
                "--max-draw-tip-z-rel",
                "0.03",
                "--max-draw-tip-xy-error",
                "0.08",
                "--max-copy-uv-mean",
                str(args.max_copy_uv_mean),
                "--max-copy-uv-p95",
                str(args.max_copy_uv_p95),
                "--max-frames",
                str(args.max_frames),
                "--max-draw-frames",
                str(args.max_draw_frames),
                "--min-target-dark-pixels",
                str(args.min_target_dark_pixels),
                "--min-target-reference-dark-ratio",
                str(args.min_target_reference_dark_ratio),
                "--allow-hdf5-only",
            ]
            verify_started = time.time()
            verify = subprocess.run(verify_cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            verify_seconds = time.time() - verify_started
        else:
            verify = None
            verify_seconds = 0.0

        if collect.returncode == 0 and verify is not None and verify.returncode == 0:
            h5_path = episode_dir / "episode.hdf5"
            summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
            verify_summary = parse_verify_json(verify.stdout)
            return {
                "ok": True,
                "id": idx,
                "seed": current_seed,
                "attempt": attempt,
                "episode_hdf5": str(h5_path.resolve()),
                "work_parent": str(work_parent.resolve()),
                "frames": int(summary["frames"]),
                "duration_s": float(summary["frames"]) / float(args.fps),
                "collect_wall_seconds": collect_seconds,
                "verify_wall_seconds": verify_seconds,
                "reference_sketch_type": summary.get("reference_sketch_type"),
                "num_reference_strokes": summary.get("num_reference_strokes"),
                "reference_points": summary.get("reference_points"),
                "drawing_points": summary.get("drawing_points"),
                "max_brush_tilt_degrees": summary.get("max_brush_tilt_degrees"),
                "phase_counts": summary.get("phase_counts"),
                "action_frame_alignment": summary.get("action_frame_alignment"),
                "low_level_step_recording": summary.get("low_level_step_recording"),
                "verify": verify_summary,
            }

        shutil.rmtree(work_parent, ignore_errors=True)

    return {
        "ok": False,
        "id": idx,
        "seed": seed,
        "error": "collector or verifier failed after retries",
        "collector_tail": tail_text(collect.stdout if "collect" in locals() else ""),
        "verifier_tail": tail_text(verify.stdout if verify is not None else ""),
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def tail_text(text: str, lines: int = 60) -> str:
    return "\n".join(text.splitlines()[-lines:])


def parse_verify_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return {}
    physics = parsed.get("physics_metrics") or {}
    copy = physics.get("copy_uv_metrics") or {}
    return {
        "estimated_added_ink_pixels": parsed.get("estimated_added_ink_pixels"),
        "target_board_dark_pixels": parsed.get("target_board_dark_pixels"),
        "reference_board_dark_pixels": parsed.get("reference_board_dark_pixels"),
        "target_reference_dark_ratio": parsed.get("target_reference_dark_ratio"),
        "bucket_visible_fraction": (parsed.get("bucket_visibility") or {}).get("visible_fraction"),
        "bucket_min_blue_pixels": (parsed.get("bucket_visibility") or {}).get("min_blue_pixels"),
        "draw_canvas_dark_pixels": parsed.get("draw_canvas_dark_pixels"),
        "copy_actual_to_reference_mean": copy.get("actual_to_reference_mean"),
        "copy_reference_to_actual_mean": copy.get("reference_to_actual_mean"),
        "copy_actual_to_reference_p95": copy.get("actual_to_reference_p95"),
        "copy_reference_to_actual_p95": copy.get("reference_to_actual_p95"),
    }


if __name__ == "__main__":
    main()
