#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_draw_dataset.config import load_config
from sim_draw_dataset.dataset import generate_episode_from_strokes
from sim_draw_dataset.external import iter_quickdraw_ndjson


def _slug(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def _load_categories(args: argparse.Namespace) -> list[str]:
    cats: list[str] = []
    if args.categories_file is not None:
        cats.extend(line.strip() for line in args.categories_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if args.categories:
        cats.extend(args.categories)
    if not cats:
        raise ValueError("Provide --categories or --categories-file")
    return cats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert QuickDraw NDJSON stroke files into robot-copy videos.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--quickdraw-dir", type=Path, required=True, help="Directory containing *.ndjson QuickDraw files.")
    parser.add_argument("--categories", nargs="+", default=None, help="Category names, without .ndjson suffix.")
    parser.add_argument("--categories-file", type=Path, default=None, help="One category per line. Safer for names with spaces.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, default=100000)
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--all-samples", action="store_true", help="Generate every recognized sample in every selected category.")
    parser.add_argument("--single-stroke-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recognized-only", action="store_true", default=True)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not (0 <= args.worker_id < args.num_workers):
        raise ValueError("--worker-id must satisfy 0 <= worker-id < num-workers")
    categories = _load_categories(args)
    cfg = load_config(args.config)
    cfg.setdefault("output", {})
    cfg["output"]["video_only"] = True
    cfg["output"]["write_manifest"] = False

    if args.out.exists() and args.overwrite and args.worker_id == 0:
        import shutil

        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shard_size = int(cfg.get("output", {}).get("shard_size", 1000))
    per_category = None if args.all_samples else (args.per_category or max(1, int(np.ceil(args.total / len(categories)))))
    manifest_name = "metadata.jsonl" if args.num_workers == 1 else f"metadata.worker{args.worker_id:02d}.jsonl"
    manifest = (args.out / manifest_name).open("w", encoding="utf-8")

    global_idx = 0
    written = 0
    try:
        for category in categories:
            path = args.quickdraw_dir / f"{category}.ndjson"
            if not path.exists():
                raise FileNotFoundError(path)

            used = 0
            for strokes, source_meta in iter_quickdraw_ndjson(
                path,
                recognized_only=args.recognized_only,
                single_stroke_only=args.single_stroke_only,
            ):
                if per_category is not None and used >= per_category:
                    break
                if not args.all_samples and global_idx >= args.total:
                    break
                current_idx = global_idx
                global_idx += 1
                current_used = used
                used += 1
                if current_idx % args.num_workers != args.worker_id:
                    continue
                shard_dir = args.out / f"shard_{current_idx // shard_size:04d}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                video_path = shard_dir / f"{_slug(category)}{current_used:06d}.mp4"
                meta = generate_episode_from_strokes(
                    cfg,
                    video_path=video_path,
                    episode_id=current_idx,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    strokes=strokes,
                    sketch_type=str(category),
                    source_meta=source_meta,
                )
                manifest.write(json.dumps({"prompt": meta["prompt"], "video": str(video_path)}, sort_keys=True) + "\n")
                if not args.quiet:
                    print(f"wrote {video_path} frames={meta['num_frames']} category={category}")
                elif written > 0 and written % 1000 == 0:
                    print(f"worker {args.worker_id}: wrote {written} videos")
                written += 1
            if not args.all_samples and global_idx >= args.total:
                break
    finally:
        manifest.close()

    print(f"done worker={args.worker_id}/{args.num_workers} videos={written} out={args.out}")


if __name__ == "__main__":
    main()
