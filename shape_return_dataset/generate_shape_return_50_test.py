#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

try:
    from shape_return_dataset.generate_shape_return_videos import ShapeReturnConfig, generate_dataset, parse_shape_list
except ModuleNotFoundError:
    from generate_shape_return_videos import ShapeReturnConfig, generate_dataset, parse_shape_list


@dataclass(frozen=True)
class ShapeReturnTestConfig:
    shape_values: tuple[str, ...] = ("ball", "square", "triangle", "parallelogram", "star")
    count_per_shape: int = 10
    fps: int = 20
    move_seconds: float = 1.0
    offscreen_wait_seconds: float = 1.0
    frame_size: int = 208
    output_dir: str = "shape_return_dataset/shape_return_test"
    struct_dir: str = "shape_return_dataset/shape_return_test_npy"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260412
    overwrite: bool = True
    codec: str = "libx264"
    quality: int = 7

    @property
    def num_videos(self) -> int:
        return len(self.shape_values) * self.count_per_shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the 50-sample shape-return test set.")
    parser.add_argument("--shape-values", type=str, default="ball,square,triangle,parallelogram,star")
    parser.add_argument("--count-per-shape", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--move-seconds", type=float, default=1.0)
    parser.add_argument("--offscreen-wait-seconds", type=float, default=1.0)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-dir", type=str, default="shape_return_dataset/shape_return_test")
    parser.add_argument("--struct-dir", type=str, default="shape_return_dataset/shape_return_test_npy")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    test_config = ShapeReturnTestConfig(
        shape_values=parse_shape_list(args.shape_values),
        count_per_shape=args.count_per_shape,
        fps=args.fps,
        move_seconds=args.move_seconds,
        offscreen_wait_seconds=args.offscreen_wait_seconds,
        frame_size=args.frame_size,
        output_dir=args.output_dir,
        struct_dir=args.struct_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
    )
    config = ShapeReturnConfig(
        shape_name=test_config.shape_values[0],
        num_videos=test_config.num_videos,
        shape_values=test_config.shape_values,
        shape_counts=tuple(test_config.count_per_shape for _ in test_config.shape_values),
        shuffle_shape_values=False,
        fps=test_config.fps,
        move_seconds=test_config.move_seconds,
        offscreen_wait_seconds=test_config.offscreen_wait_seconds,
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
