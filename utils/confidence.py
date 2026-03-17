"""Confidence-map construction for DEM repair inputs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _normalize_zero_one(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    minimum = float(np.nanmin(array[finite]))
    maximum = float(np.nanmax(array[finite]))
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    normalized = (array - minimum) / (maximum - minimum)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32)


def build_confidence_map(
    mask: np.ndarray,
    ground_penetration_ratio: np.ndarray | None = None,
    partial_dem: np.ndarray | None = None,
    dsm: np.ndarray | None = None,
    coarse_dem: np.ndarray | None = None,
    distance_scale: float = 24.0,
) -> np.ndarray:
    """Construct a simple confidence map for DEM repair."""
    hole_mask = mask.astype(bool)
    if partial_dem is not None:
        valid_mask = np.isfinite(partial_dem) & ~hole_mask
    else:
        valid_mask = ~hole_mask

    distance = distance_transform_edt(~valid_mask)
    distance_conf = np.exp(-(distance / max(distance_scale, 1e-6))).astype(np.float32)

    if ground_penetration_ratio is None:
        penetration_conf = np.ones_like(distance_conf, dtype=np.float32)
    else:
        penetration_conf = np.clip(np.nan_to_num(ground_penetration_ratio, nan=0.0), 0.0, 1.0).astype(np.float32)

    if dsm is not None and coarse_dem is not None:
        height_delta = np.abs(np.asarray(dsm, dtype=np.float32) - np.asarray(coarse_dem, dtype=np.float32))
        height_conf = 1.0 - _normalize_zero_one(height_delta)
    else:
        height_conf = np.ones_like(distance_conf, dtype=np.float32)

    combined = (0.5 * distance_conf) + (0.35 * penetration_conf) + (0.15 * height_conf)
    confidence = np.where(hole_mask, combined, 1.0).astype(np.float32)
    return np.clip(confidence, 0.0, 1.0)
