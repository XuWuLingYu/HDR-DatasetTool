#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sim_draw_dataset.config import load_config
from sim_draw_dataset.dataset import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate robot sketch-copy simulation episodes.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["dataset"]["episodes"] = args.episodes
    if args.seed is not None:
        cfg["dataset"]["seed"] = args.seed
    generate_dataset(cfg, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

