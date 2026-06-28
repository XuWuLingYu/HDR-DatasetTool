#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

try:
    from bounce_ball_dataset.generate_bounce_ball_videos import BounceBallConfig, generate_dataset, parse_int_list
except ModuleNotFoundError:
    from generate_bounce_ball_videos import BounceBallConfig, generate_dataset, parse_int_list


@dataclass(frozen=True)
class BounceBallTestConfig:
    bounce_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    count_per_bounce: int = 10
    fps: int = 20
    pixels_per_frame: float = 4.0
    frame_size: int = 208
    output_dir: str = "bounce_ball_dataset/bounce_ball_test"
    struct_dir: str = "bounce_ball_dataset/bounce_ball_test_npy"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260412
    overwrite: bool = True
    codec: str = "libx264"
    quality: int = 7

    @property
    def num_videos(self) -> int:
        return len(self.bounce_values) * self.count_per_bounce


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the 50-sample bounce-ball test set.")
    parser.add_argument("--bounce-values", type=str, default="1,2,3,4,5")
    parser.add_argument("--count-per-bounce", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--pixels-per-frame", type=float, default=4.0)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-dir", type=str, default="bounce_ball_dataset/bounce_ball_test")
    parser.add_argument("--struct-dir", type=str, default="bounce_ball_dataset/bounce_ball_test_npy")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    test_config = BounceBallTestConfig(
        bounce_values=parse_int_list(args.bounce_values),
        count_per_bounce=args.count_per_bounce,
        fps=args.fps,
        pixels_per_frame=args.pixels_per_frame,
        frame_size=args.frame_size,
        output_dir=args.output_dir,
        struct_dir=args.struct_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
    )
    config = BounceBallConfig(
        bounce_count=test_config.bounce_values[0],
        num_videos=test_config.num_videos,
        bounce_values=test_config.bounce_values,
        bounce_counts=tuple(test_config.count_per_bounce for _ in test_config.bounce_values),
        shuffle_bounce_values=False,
        fps=test_config.fps,
        pixels_per_frame=test_config.pixels_per_frame,
        frame_size=test_config.frame_size,
        output_dir=test_config.output_dir,
        struct_dir=test_config.struct_dir,
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
