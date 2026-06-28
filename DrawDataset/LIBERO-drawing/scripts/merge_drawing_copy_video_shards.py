#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path(sys.executable)
PROMPT = "用机器人复制左边的图案"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LIBERO drawing-copy video dataset shards.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--require-use-weld", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out.resolve()
    videos_dir = out_dir / "videos"
    metadata_path = out_dir / "metadata.jsonl"
    manifest_path = out_dir / "manifest.jsonl"

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    by_id = {}
    for shard in args.shards:
        manifest = shard / "manifest.jsonl"
        if not manifest.exists():
            raise SystemExit(f"missing shard manifest: {manifest}")
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                idx = int(record["id"])
                if idx in by_id:
                    raise SystemExit(f"duplicate id {idx} from {manifest}")
                if args.require_use_weld and not has_physical_audit(record):
                    raise SystemExit(f"shard manifest id {idx} is missing strict physical audit fields")
                src = Path(record["video_path"])
                if not src.exists():
                    raise SystemExit(f"missing shard video for id {idx}: {src}")
                dst = videos_dir / f"drawing_copy_{idx:06d}.mp4"
                atomic_materialize(src, dst, mode=args.copy_mode)
                merged = dict(record)
                merged["prompt"] = PROMPT
                merged["video_path"] = str(dst.resolve())
                by_id[idx] = merged

    records = [by_id[idx] for idx in sorted(by_id)]
    write_jsonl(manifest_path, records)
    write_jsonl(metadata_path, [{"prompt": PROMPT, "video_path": record["video_path"]} for record in records])

    expected = args.expected_count if args.expected_count is not None else len(records)
    validation = run_validator(out_dir, expected, args.workers, require_use_weld=args.require_use_weld)
    summary = {
        "ok": validation.get("ok", False),
        "out": str(out_dir),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "videos": str(videos_dir),
        "shards": [str(path.resolve()) for path in args.shards],
        "count": len(records),
        "expected_count": expected,
        "require_use_weld": args.require_use_weld,
        "validation": validation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    if mode == "hardlink":
        tmp.hardlink_to(src)
    else:
        shutil.copy2(src, tmp)
    tmp.replace(dst)


def run_validator(out_dir: Path, expected_count: int, workers: int, require_use_weld: bool = False) -> dict:
    cmd = [
        str(PYTHON_BIN),
        str(ROOT / "scripts" / "validate_drawing_copy_video_dataset.py"),
        str(out_dir),
        "--expected-count",
        str(expected_count),
        "--workers",
        str(max(1, int(workers))),
        "--require-contiguous-ids",
    ]
    if require_use_weld:
        cmd.append("--require-use-weld")
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"ok": False, "output": proc.stdout[-2400:]}
    result["returncode"] = proc.returncode
    return result


def has_physical_audit(record: dict) -> bool:
    if record.get("use_weld") is not True or record.get("scripted_ink_mode") is not False:
        return False
    if not isinstance(record.get("estimated_added_ink_pixels"), int):
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


if __name__ == "__main__":
    main()
