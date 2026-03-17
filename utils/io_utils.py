"""Filesystem and lightweight serialization helpers for DEM repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def describe_path(path: str | Path) -> dict[str, str]:
    """Return a small description for a path."""
    target = Path(path)
    return {
        "path": str(target),
        "exists": str(target.exists()),
        "is_dir": str(target.is_dir()),
    }


def list_files(directory: str | Path, patterns: Iterable[str]) -> list[Path]:
    """Return sorted files that match any of the supplied glob patterns."""
    root = Path(directory)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    return sorted({path.resolve() for path in matches})


def save_npz(path: str | Path, **arrays: Any) -> Path:
    """Save arrays to a compressed NPZ file."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    np.savez_compressed(output_path, **arrays)
    return output_path


def load_npz(path: str | Path) -> dict[str, Any]:
    """Load an NPZ file into a regular dictionary."""
    file_path = Path(path)
    with np.load(file_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)
