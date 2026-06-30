from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.diffusion_reasoning.envs import OneLineDrawingEnv, SlidingPuzzleEnv, SokobanEnv, WaterPourEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diffusion reasoning datasets.")
    parser.add_argument(
        "--config",
        default="configs/diffusion_reasoning/env_configs.yaml",
        help="Path to the environment config YAML.",
    )
    parser.add_argument(
        "--env",
        default="one_line_drawing",
        choices=["one_line_drawing", "water_pour", "sliding_puzzle", "sokoban"],
        help="Environment to generate.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of difficulty levels to generate.",
    )
    parser.add_argument(
        "--count-override",
        type=int,
        default=None,
        help="Optional override for both train and eval counts per level, useful for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes to use across level/split generation.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_count_override(config: dict, env_name: str, count_override: int | None) -> dict:
    if count_override is None:
        return config

    environments = config.setdefault("environments", {})
    env_config = environments[env_name]
    for level_config in env_config.get("levels", {}).values():
        level_config.setdefault("counts", {})
        level_config["counts"]["train"] = count_override
        level_config["counts"]["eval"] = count_override
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_count_override(config, args.env, args.count_override)
    env_config = config["environments"][args.env]
    env_config["global"] = config.get("global", {})
    env_map = {
        "one_line_drawing": OneLineDrawingEnv,
        "sliding_puzzle": SlidingPuzzleEnv,
        "sokoban": SokobanEnv,
        "water_pour": WaterPourEnv,
    }
    env = env_map[args.env](env_config)
    env.generate_dataset(output_root=args.output_root, levels=args.levels, workers=args.workers)


if __name__ == "__main__":
    main()
