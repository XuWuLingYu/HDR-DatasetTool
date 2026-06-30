from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

class BaseEnv(ABC):
    """Minimal shared interface for config-driven dataset environments."""

    task_name: str

    def __init__(self, env_config: dict[str, Any]) -> None:
        self.config = env_config
        self.seed = int(env_config.get("seed", 0))

    @classmethod
    def load_config(cls, config_path: str | Path, env_name: str) -> dict[str, Any]:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        environments = config.get("environments", {})
        if env_name not in environments:
            raise KeyError(f"Environment '{env_name}' not found in {config_path}")

        env_config = dict(environments[env_name])
        env_config.setdefault("global", config.get("global", {}))
        return env_config

    def ensure_split_dir(self, difficulty_dir: Path, split: str) -> Path:
        split_dir = difficulty_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def write_metadata(self, metadata_path: Path, payload: dict[str, Any]) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @abstractmethod
    def generate_dataset(self, output_root: str | Path, levels: list[int] | None = None) -> None:
        """Generate dataset artifacts for this environment."""
