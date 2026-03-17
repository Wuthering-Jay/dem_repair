"""Mask simulation and boundary extraction helpers for DEM repair."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def build_hole_mask(dem: np.ndarray, nodata_value: Any = None) -> np.ndarray:
    """Create a hole mask from nodata or non-finite pixels."""
    if dem is None:
        return np.zeros((0, 0), dtype=bool)
    mask = ~np.isfinite(dem)
    if nodata_value is not None and not np.isnan(nodata_value):
        mask |= dem == nodata_value
    return mask


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert a binary mask."""
    return ~mask.astype(bool)


def apply_mask_to_dem(dem: np.ndarray, mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """Apply a void mask to a DEM."""
    output = dem.astype(np.float32, copy=True)
    output[mask.astype(bool)] = fill_value
    return output


def _sample_target_pixels(valid_mask: np.ndarray, coverage_range: tuple[float, float], rng: np.random.Generator) -> int:
    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        return 0
    low = max(0.0, float(coverage_range[0]))
    high = min(0.9, float(coverage_range[1]))
    coverage = float(rng.uniform(low, high))
    return max(1, int(valid_pixels * coverage))


def _draw_disk(shape: tuple[int, int], center_row: int, center_col: int, radius: int) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius**2


def _irregular_random_mask(
    valid_mask: np.ndarray,
    target_pixels: int,
    rng: np.random.Generator,
    max_iters: int = 200,
) -> np.ndarray:
    mask = np.zeros_like(valid_mask, dtype=bool)
    valid_positions = np.argwhere(valid_mask)
    if valid_positions.size == 0:
        return mask
    for _ in range(max_iters):
        row, col = valid_positions[
            rng.integers(0, len(valid_positions))]
        radius = int(rng.integers(6, 24))
        mask |= _draw_disk(
            valid_mask.shape, int(row), int(col), radius)
        if rng.random() < 0.75:
            mask = binary_dilation(
                mask, iterations=int(rng.integers(1, 4)))
        mask &= valid_mask
        if int(mask.sum()) >= target_pixels:
            break
    return mask & valid_mask


def _block_mask(valid_mask: np.ndarray, target_pixels: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros_like(valid_mask, dtype=bool)
    height, width = valid_mask.shape
    attempts = 0
    while int(mask.sum()) < target_pixels and attempts < 64:
        block_h = int(rng.integers(max(8, height // 12), max(12, height // 4)))
        block_w = int(rng.integers(max(8, width // 12), max(12, width // 4)))
        row = int(rng.integers(0, max(1, height - block_h)))
        col = int(rng.integers(0, max(1, width - block_w)))
        mask[row : row + block_h, col : col + block_w] = True
        mask &= valid_mask
        attempts += 1
    return mask & valid_mask


def _vegetation_priority_mask(
    valid_mask: np.ndarray,
    vegetation_density: np.ndarray,
    target_pixels: int,
    rng: np.random.Generator,
    max_iters: int = 200,
) -> np.ndarray:
    mask = np.zeros_like(valid_mask, dtype=bool)
    weights = np.asarray(
        vegetation_density, dtype=np.float32).copy()
    weights[~np.isfinite(weights)] = 0.0
    weights[~valid_mask] = 0.0
    flat_weights = weights.reshape(-1)
    if flat_weights.sum() <= 0:
        return _irregular_random_mask(
            valid_mask, target_pixels, rng)

    probabilities = flat_weights / flat_weights.sum()
    flat_indices = np.arange(flat_weights.size)
    for _ in range(max_iters):
        index = int(rng.choice(
            flat_indices, p=probabilities))
        row, col = np.unravel_index(
            index, valid_mask.shape)
        radius = int(rng.integers(8, 28))
        blob = _draw_disk(
            valid_mask.shape, row, col, radius)
        blob = binary_dilation(
            blob, iterations=int(rng.integers(1, 3)))
        mask |= blob
        mask &= valid_mask
        if int(mask.sum()) >= target_pixels:
            break
    return mask & valid_mask


def simulate_void_mask(
    valid_mask: np.ndarray,
    vegetation_density: np.ndarray | None = None,
    strategy: str = "mixed",
    coverage_range: tuple[float, float] = (0.05, 0.22),
    seed: int | None = None,
) -> np.ndarray:
    """Simulate artificial DEM voids on valid terrain pixels."""
    rng = np.random.default_rng(seed)
    valid = valid_mask.astype(bool)
    target_pixels = _sample_target_pixels(valid, coverage_range, rng)
    if target_pixels <= 0:
        return np.zeros_like(valid, dtype=bool)

    selected_strategy = strategy
    if strategy == "mixed":
        candidates = ["irregular", "block", "vegetation_priority"] if vegetation_density is not None else ["irregular", "block"]
        selected_strategy = str(rng.choice(candidates))

    if selected_strategy == "irregular":
        mask = _irregular_random_mask(valid, target_pixels, rng)
    elif selected_strategy == "block":
        mask = _block_mask(valid, target_pixels, rng)
    elif selected_strategy == "vegetation_priority":
        density = vegetation_density if vegetation_density is not None else np.zeros_like(valid, dtype=np.float32)
        mask = _vegetation_priority_mask(valid, density, target_pixels, rng)
    else:
        raise ValueError(f"Unsupported void simulation strategy: {selected_strategy}")

    return mask & valid


def extract_boundary_band(mask: np.ndarray, inner: int = 2, outer: int = 2) -> np.ndarray:
    """Extract a narrow band around mask boundaries."""
    hole_mask = mask.astype(bool)
    inner_band = hole_mask & ~binary_erosion(hole_mask, iterations=max(1, inner))
    outer_band = binary_dilation(hole_mask, iterations=max(1, outer)) & ~hole_mask
    return inner_band | outer_band
