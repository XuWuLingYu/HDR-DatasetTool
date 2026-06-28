#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

try:
    from shell_game_dataset.generate_shell_game_videos import ShellGameConfig, generate_dataset, parse_int_list
except ModuleNotFoundError:
    from generate_shell_game_videos import ShellGameConfig, generate_dataset, parse_int_list


@dataclass(frozen=True)
class ShellGameTestConfig:
    swap_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    count_per_swap: int = 10
    fps: int = 20
    frame_size: int = 208
    output_base_dir: str = "shell_game_dataset/test"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260413
    overwrite: bool = True
    codec: str = "libx264"
    quality: int = 7

    @property
    def num_videos(self) -> int:
        return len(self.swap_values) * self.count_per_swap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the 50-sample shell-game test set.")
    parser.add_argument("--swap-values", type=str, default="1,2,3,4,5")
    parser.add_argument("--count-per-swap", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-base-dir", type=str, default="shell_game_dataset/test")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    test_config = ShellGameTestConfig(
        swap_values=parse_int_list(args.swap_values),
        count_per_swap=args.count_per_swap,
        fps=args.fps,
        frame_size=args.frame_size,
        output_base_dir=args.output_base_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
    )
    config = ShellGameConfig(
        num_swaps=test_config.swap_values[0],
        num_videos=test_config.num_videos,
        swap_values=test_config.swap_values,
        swap_counts=tuple(test_config.count_per_swap for _ in test_config.swap_values),
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
    print(json.dumps(asdict(test_config), ensure_ascii=False, indent=2))
    generate_dataset(config)


if __name__ == "__main__":
    main()
