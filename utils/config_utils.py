"""Configuration loading helpers for repository-driven workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to the repository root or an optional base."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base_dir = Path(base) if base is not None else get_repo_root()
    return (base_dir / candidate).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    config_path = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def save_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist a YAML file."""
    output_path = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return output_path
