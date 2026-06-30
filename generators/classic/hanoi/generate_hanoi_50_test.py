#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

try:
    from generators.classic.hanoi.generate_hanoi_videos import HanoiConfig, VideoTask, parse_int_list, process_task
except ModuleNotFoundError:
    from generate_hanoi_videos import HanoiConfig, VideoTask, parse_int_list, process_task


@dataclass(frozen=True)
class HanoiTestConfig:
    sizes: tuple[int, ...] = (2, 3, 4, 5)
    count_per_size: int = 10
    fps: int = 10
    frames_per_move: int = 15
    frame_size: int = 208
    output_dir: str = "outputs/hanoi/test"
    struct_dir: str = "outputs/hanoi/test_npy"
    metadata_name: str = "metadata.jsonl"
    seed: int = 20260412
    overwrite: bool = True
    codec: str = "libx264"
    quality: int = 7
    in_domain_initial_rods: tuple[int, ...] = (0, 1)
    ood_initial_rods: tuple[int, ...] = (0, 1, 2)
    goal_rod: int = 2
    require_ood_right_rod: bool = True

    @property
    def in_domain_count(self) -> int:
        return len(self.sizes) * self.count_per_size

    @property
    def ood_count(self) -> int:
        return len(self.sizes) * self.count_per_size

    @property
    def num_videos(self) -> int:
        return self.in_domain_count + self.ood_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the mixed Tower of Hanoi test set with in-domain and OOD subsets."
    )
    parser.add_argument("--sizes", type=str, default="2,3,4,5")
    parser.add_argument("--count-per-size", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frames-per-move", type=int, default=15)
    parser.add_argument("--frame-size", type=int, default=208)
    parser.add_argument("--output-dir", type=str, default="outputs/hanoi/test")
    parser.add_argument("--struct-dir", type=str, default="outputs/hanoi/test_npy")
    parser.add_argument("--metadata-name", type=str, default="metadata.jsonl")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--in-domain-initial-rods", type=str, default="0,1")
    parser.add_argument("--ood-initial-rods", type=str, default="0,1,2")
    parser.add_argument("--goal-rod", type=int, default=2)
    parser.add_argument("--allow-ood-without-right-rod", action="store_true")
    return parser


def cleanup_existing_outputs(output_dir: Path, struct_dir: Path, metadata_name: str) -> None:
    for path in output_dir.glob("hanoi_*.mp4"):
        path.unlink(missing_ok=True)
    (output_dir / metadata_name).unlink(missing_ok=True)
    for path in struct_dir.glob("hanoi_*.npy"):
        path.unlink(missing_ok=True)


def write_metadata(metadata_path: Path, records: list[dict[str, object]]) -> None:
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_mixed_test_specs(test_config: HanoiTestConfig) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    index = 0
    subset_defs = (
        (
            "in_domain",
            test_config.in_domain_initial_rods,
            (),
        ),
        (
            "ood",
            test_config.ood_initial_rods,
            (test_config.goal_rod,) if test_config.require_ood_right_rod else (),
        ),
    )
    for domain_split, initial_rods, required_initial_rods_present in subset_defs:
        domain_offset = index
        for num_disks in test_config.sizes:
            for size_offset in range(test_config.count_per_size):
                specs.append(
                    {
                        "index": index,
                        "seed": test_config.seed + index,
                        "num_disks": num_disks,
                        "domain_split": domain_split,
                        "allowed_initial_rods": initial_rods,
                        "required_initial_rods_present": required_initial_rods_present,
                        "domain_index": index - domain_offset,
                        "size_index": size_offset,
                    }
                )
                index += 1
    return specs


def task_config_from_spec(test_config: HanoiTestConfig, spec: dict[str, object]) -> HanoiConfig:
    return HanoiConfig(
        num_disks=int(spec["num_disks"]),
        num_videos=1,
        disk_values=(int(spec["num_disks"]),),
        disk_counts=(1,),
        shuffle_disk_values=False,
        fps=test_config.fps,
        frames_per_move=test_config.frames_per_move,
        frame_size=test_config.frame_size,
        output_dir=test_config.output_dir,
        struct_dir=test_config.struct_dir,
        metadata_name=test_config.metadata_name,
        seed=int(spec["seed"]),
        workers=1,
        overwrite=test_config.overwrite,
        codec=test_config.codec,
        quality=test_config.quality,
        initial_rods=tuple(int(value) for value in spec["allowed_initial_rods"]),
        goal_rod=test_config.goal_rod,
        allow_solved_initial=False,
        required_initial_rods_present=tuple(int(value) for value in spec["required_initial_rods_present"]),
    )


def main() -> None:
    args = build_parser().parse_args()
    test_config = HanoiTestConfig(
        sizes=parse_int_list(args.sizes),
        count_per_size=args.count_per_size,
        fps=args.fps,
        frames_per_move=args.frames_per_move,
        frame_size=args.frame_size,
        output_dir=args.output_dir,
        struct_dir=args.struct_dir,
        metadata_name=args.metadata_name,
        seed=args.seed,
        overwrite=args.overwrite,
        codec=args.codec,
        quality=args.quality,
        in_domain_initial_rods=parse_int_list(args.in_domain_initial_rods),
        ood_initial_rods=parse_int_list(args.ood_initial_rods),
        goal_rod=args.goal_rod,
        require_ood_right_rod=not args.allow_ood_without_right_rod,
    )
    print(json.dumps(asdict(test_config), ensure_ascii=False, indent=2))

    output_dir = Path(test_config.output_dir)
    struct_dir = Path(test_config.struct_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    if test_config.overwrite:
        cleanup_existing_outputs(output_dir, struct_dir, test_config.metadata_name)

    records: list[dict[str, object]] = []
    for spec in tqdm(build_mixed_test_specs(test_config), desc="Generating mixed Hanoi test set", leave=True):
        metadata = process_task(
            VideoTask(
                index=int(spec["index"]),
                num_disks=int(spec["num_disks"]),
                seed=int(spec["seed"]),
            ),
            task_config_from_spec(test_config, spec),
        )
        metadata["domain_split"] = str(spec["domain_split"])
        metadata["domain_index"] = int(spec["domain_index"])
        metadata["size_index"] = int(spec["size_index"])
        records.append(metadata)

    records.sort(key=lambda item: int(item["index"]))
    write_metadata(output_dir / test_config.metadata_name, records)


if __name__ == "__main__":
    main()
