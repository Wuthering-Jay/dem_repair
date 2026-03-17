"""Terrain feature extraction for DEM repair conditioning and diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from .interpolation import interpolate_missing_values
from .mask_utils import extract_boundary_band


def _prepare_surface(dem: np.ndarray) -> np.ndarray:
    surface = np.asarray(dem, dtype=np.float32)
    return interpolate_missing_values(surface, method="nearest") if (~np.isfinite(surface)).any() else surface


def compute_slope(dem: Any, resolution: float) -> np.ndarray:
    """Compute slope magnitude from a DEM-like surface."""
    surface = _prepare_surface(np.asarray(dem, dtype=np.float32))
    grad_y, grad_x = np.gradient(surface, float(resolution))
    slope = np.sqrt(grad_x**2 + grad_y**2)
    return slope.astype(np.float32)


def compute_curvature(dem: Any, resolution: float) -> np.ndarray:
    """Compute a simple Laplacian-style curvature map."""
    surface = _prepare_surface(np.asarray(dem, dtype=np.float32))
    grad_y, grad_x = np.gradient(surface, float(resolution))
    grad_yy = np.gradient(grad_y, float(resolution), axis=0)
    grad_xx = np.gradient(grad_x, float(resolution), axis=1)
    curvature = grad_xx + grad_yy
    return curvature.astype(np.float32)


def compute_openness(dem: Any, radius: int) -> np.ndarray:
    """Compute a stable openness proxy using local min-max envelopes."""
    surface = _prepare_surface(np.asarray(dem, dtype=np.float32))
    size = max(3, (int(radius) * 2) + 1)
    local_max = maximum_filter(surface, size=size, mode="nearest")
    local_min = minimum_filter(surface, size=size, mode="nearest")
    openness = surface - (0.5 * (local_max + local_min))
    return openness.astype(np.float32)


def compute_multi_scale_openness(dem: Any, radii: list[int]) -> dict[str, np.ndarray]:
    """Compute multiple openness maps with stable naming."""
    return {
        f"openness_s{index + 1}": compute_openness(dem, radius)
        for index, radius in enumerate(radii)
    }


def compute_boundary_band(mask: np.ndarray, inner: int = 2, outer: int = 2) -> np.ndarray:
    """Compute a band around the void boundary."""
    return extract_boundary_band(mask, inner=inner, outer=outer).astype(np.float32)


def compute_terrain_feature_stack(
    dem: np.ndarray,
    resolution: float,
    openness_scales: list[int],
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the terrain feature maps used by the sample builder."""
    features = {
        "slope": compute_slope(dem, resolution),
        "curvature": compute_curvature(dem, resolution),
    }
    features.update(compute_multi_scale_openness(dem, openness_scales))
    if mask is not None:
        features["boundary_band"] = compute_boundary_band(mask)
    return {name: value.astype(np.float32) for name, value in features.items()}
