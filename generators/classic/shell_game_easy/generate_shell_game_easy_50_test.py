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

from generators.classic.shell_game_easy.generate_shell_game_easy_videos import ShellGameEasyConfig
from generators.classic.shell_game.generate_shell_game_videos import ShellGameConfig, generate_dataset


@dataclass(frozen=True)
class ShellGameEasyTestConfig:
    num_videos: int = 50
    fps: int = 20
    frame_size: int = 208
    output_base_dir: str = "outputs/shell_game_easy/test"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260415
    overwrite: bool = True
    codec: str = "libx264"
    quality: int = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the 50-sample shell-game easy test set.")
    parser.add_argument("--num-videos", type=int, default=50)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-base-dir", type=str, default="outputs/shell_game_easy/test")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    test_config = ShellGameEasyTestConfig(
        num_videos=args.num_videos,
        fps=args.fps,
        frame_size=args.frame_size,
        output_base_dir=args.output_base_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
    )
    print(json.dumps(asdict(test_config), ensure_ascii=False, indent=2))
    config = ShellGameConfig(
        num_swaps=1,
        num_videos=test_config.num_videos,
        swap_values=(),
        swap_counts=(),
        shuffle_swap_values=False,
        fps=test_config.fps,
        frame_size=test_config.frame_size,
        output_base_dir=test_config.output_base_dir,
        metadata_name=test_config.metadata_name,
        seed=test_config.seed,
        workers=1,
        overwrite=test_config.overwrite,
        codec=test_config.codec,
        quality=test_config.quality,
    )
    generate_dataset(config)


if __name__ == "__main__":
    main()
