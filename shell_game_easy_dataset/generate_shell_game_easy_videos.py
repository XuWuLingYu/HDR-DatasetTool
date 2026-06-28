#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shell_game_dataset.generate_shell_game_videos import ShellGameConfig, generate_dataset


@dataclass(frozen=True)
class ShellGameEasyConfig:
    num_videos: int = 3000
    fps: int = 20
    frame_size: int = 208
    output_base_dir: str = "shell_game_easy_dataset/train"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260415
    workers: int = 8
    overwrite: bool = False
    codec: str = "libx264"
    quality: int = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate shell-game easy videos with exactly one swap.")
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-base-dir", type=str, default="shell_game_easy_dataset/train")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    easy_config = ShellGameEasyConfig(
        num_videos=args.num_videos,
        fps=args.fps,
        frame_size=args.frame_size,
        output_base_dir=args.output_base_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        workers=args.workers,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
    )
    config = ShellGameConfig(
        num_swaps=1,
        num_videos=easy_config.num_videos,
        swap_values=(),
        swap_counts=(),
        shuffle_swap_values=False,
        fps=easy_config.fps,
        frame_size=easy_config.frame_size,
        output_base_dir=easy_config.output_base_dir,
        metadata_name=easy_config.metadata_name,
        seed=easy_config.seed,
        workers=easy_config.workers,
        overwrite=easy_config.overwrite,
        codec=easy_config.codec,
        quality=easy_config.quality,
    )
    print(json.dumps(asdict(easy_config), ensure_ascii=False, indent=2))
    generate_dataset(config)


if __name__ == "__main__":
    main()
