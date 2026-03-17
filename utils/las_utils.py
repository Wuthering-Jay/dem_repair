"""LAS reading and class-filtering helpers for DEM repair preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np


def read_las_points(las_path: str | Path) -> dict[str, Any]:
    """Read a LAS file and extract core point attributes."""
    path = Path(las_path)
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float32)
    intensity = np.asarray(getattr(las, "intensity", np.zeros_like(z)), dtype=np.float32)
    classification = np.asarray(getattr(las, "classification", np.zeros_like(z)), dtype=np.int16)
    return_num = np.asarray(getattr(las, "return_number", np.ones_like(classification)), dtype=np.int16)
    num_returns = np.asarray(getattr(las, "number_of_returns", np.ones_like(classification)), dtype=np.int16)
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None
    return {
        "path": str(path),
        "x": x,
        "y": y,
        "z": z,
        "intensity": intensity,
        "classification": classification,
        "return_num": return_num,
        "num_returns": num_returns,
        "crs": crs,
        "point_count": int(x.size),
    }


def read_las_header_info(las_path: str | Path) -> dict[str, Any]:
    """Read LAS header metadata without loading all points into memory."""
    path = Path(las_path)
    with laspy.open(path) as reader:
        header = reader.header
        try:
            crs = header.parse_crs()
        except Exception:
            crs = None
        mins = np.asarray(header.mins, dtype=np.float64)
        maxs = np.asarray(header.maxs, dtype=np.float64)
        return {
            "path": str(path),
            "point_count": int(header.point_count),
            "mins": mins,
            "maxs": maxs,
            "crs": crs,
        }


def iter_las_chunks(las_path: str | Path, chunk_size: int) -> Any:
    """Yield LAS point data in chunks for streaming raster generation."""
    path = Path(las_path)
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(max(1, int(chunk_size))):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float32)
            intensity = np.asarray(getattr(chunk, "intensity", np.zeros_like(z)), dtype=np.float32)
            classification = np.asarray(getattr(chunk, "classification", np.zeros_like(z)), dtype=np.int16)
            return_num = np.asarray(getattr(chunk, "return_number", np.ones_like(classification)), dtype=np.int16)
            num_returns = np.asarray(getattr(chunk, "number_of_returns", np.ones_like(classification)), dtype=np.int16)
            yield {
                "x": x,
                "y": y,
                "z": z,
                "intensity": intensity,
                "classification": classification,
                "return_num": return_num,
                "num_returns": num_returns,
                "point_count": int(x.size),
            }


def read_las_metadata(las_path: str | Path) -> dict[str, Any]:
    """Read lightweight metadata for a LAS file."""
    points = read_las_points(las_path)
    classification = points["classification"]
    unique_classes, counts = np.unique(classification, return_counts=True)
    return {
        "path": points["path"],
        "point_count": points["point_count"],
        "classes": {int(class_id): int(count) for class_id, count in zip(unique_classes, counts, strict=False)},
        "crs": str(points["crs"]) if points["crs"] is not None else None,
    }


def filter_supported_classes(class_ids: list[int] | np.ndarray, supported: set[int]) -> list[int]:
    """Filter class ids using a supported-class set."""
    return [int(class_id) for class_id in class_ids if int(class_id) in supported]


def build_class_masks(classification: np.ndarray, class_map: dict[str, int]) -> dict[str, np.ndarray]:
    """Build boolean masks for configured classes."""
    return {
        name: classification == int(class_id)
        for name, class_id in class_map.items()
    }


def filter_points(points: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    """Return a filtered copy of a point dictionary."""
    filtered: dict[str, Any] = {"path": points["path"], "crs": points.get("crs")}
    for key, value in points.items():
        if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
            filtered[key] = value[mask]
        elif key not in filtered:
            filtered[key] = value
    filtered["point_count"] = int(mask.sum())
    return filtered


def get_valid_surface_mask(classification: np.ndarray, supported_classes: set[int]) -> np.ndarray:
    """Return the default valid-point mask for raster generation."""
    return np.isin(classification, list(supported_classes))
