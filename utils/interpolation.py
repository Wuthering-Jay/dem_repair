"""Interpolation helpers for coarse DEM completion and reference DEM repair."""

from __future__ import annotations

import numpy as np
from rasterio.fill import fillnodata
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt, gaussian_filter, label, median_filter

from .raster_utils import nearest_fill


def interpolate_missing_values(
    array: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "nearest_gaussian",
    smooth_sigma: float = 2.0,
    max_search_distance: float = 64.0,
    smoothing_iterations: int = 0,
) -> np.ndarray:
    """Interpolate missing values in a DEM-like array."""
    source = np.asarray(array, dtype=np.float32)
    if mask is None:
        invalid = ~np.isfinite(source)
    else:
        invalid = mask.astype(bool) | (~np.isfinite(source))
    valid = ~invalid

    if valid.all():
        return source.copy()

    if method in {"idw", "idw_gaussian"}:
        filled = _idw_fill(
            source,
            valid_mask=valid,
            max_search_distance=max_search_distance,
            smoothing_iterations=smoothing_iterations,
        )
        if method == "idw":
            return np.where(valid, source, filled).astype(np.float32)

        smoothed = gaussian_filter(filled, sigma=max(0.0, float(smooth_sigma)))
        return np.where(valid, source, smoothed).astype(np.float32)

    filled = nearest_fill(np.where(valid, source, np.nan))
    if method == "nearest":
        return np.where(valid, source, filled).astype(np.float32)

    if method == "nearest_gaussian":
        smoothed = gaussian_filter(filled, sigma=max(0.0, float(smooth_sigma)))
        return np.where(valid, source, smoothed).astype(np.float32)

    raise ValueError(f"Unsupported interpolation method: {method}")


def interpolate_coarse_dem(
    partial_dem: np.ndarray,
    mask: np.ndarray,
    method: str = "nearest_gaussian",
    smooth_sigma: float = 2.0,
    max_search_distance: float = 64.0,
    smoothing_iterations: int = 0,
) -> np.ndarray:
    """Build a coarse DEM prior from a partial DEM."""
    return interpolate_missing_values(
        partial_dem,
        mask=mask,
        method=method,
        smooth_sigma=smooth_sigma,
        max_search_distance=max_search_distance,
        smoothing_iterations=smoothing_iterations,
    )


def _idw_fill(
    source: np.ndarray,
    valid_mask: np.ndarray,
    max_search_distance: float,
    smoothing_iterations: int,
) -> np.ndarray:
    """Fill invalid cells using rasterio's inverse-distance-weighted nodata fill."""
    if valid_mask.sum() <= 0:
        return np.zeros_like(source, dtype=np.float32)

    image = np.where(valid_mask, source, 0.0).astype(np.float32)
    fill_mask = valid_mask.astype(np.uint8)
    filled = fillnodata(
        image,
        mask=fill_mask,
        max_search_distance=float(max_search_distance),
        smoothing_iterations=max(0, int(smoothing_iterations)),
    )
    return np.asarray(filled, dtype=np.float32)


def _linear_confidence(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map values to a [0, 1] confidence range using linear scaling."""
    low_value = float(low)
    high_value = float(high)
    if high_value <= low_value:
        high_value = low_value + 1.0
    confidence = (np.asarray(values, dtype=np.float32) - low_value) / (high_value - low_value)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _stabilize_with_local_median(
    surface: np.ndarray,
    support_mask: np.ndarray,
    blend_weight: np.ndarray,
    median_size: int,
) -> np.ndarray:
    """Suppress cell-wise spikes in low-confidence regions using a local median trend."""
    size = max(1, int(median_size))
    if size <= 1:
        return surface.astype(np.float32, copy=True)

    support = np.asarray(support_mask, dtype=bool)
    weight = np.clip(np.asarray(blend_weight, dtype=np.float32), 0.0, 1.0)
    if not support.any() or float(weight.max()) <= 0.0:
        return surface.astype(np.float32, copy=True)

    source = np.asarray(surface, dtype=np.float32)
    support_surface = np.where(support & np.isfinite(source), source, np.nan)
    support_filled = nearest_fill(support_surface)
    median_surface = median_filter(support_filled, size=size, mode="nearest").astype(np.float32)

    stabilized = source.astype(np.float32, copy=True)
    target_mask = support & np.isfinite(source) & (weight > 0.0)
    stabilized[target_mask] = (
        ((1.0 - weight[target_mask]) * source[target_mask])
        + (weight[target_mask] * median_surface[target_mask])
    ).astype(np.float32)
    stabilized[~support] = np.nan
    return stabilized


def _promote_small_support_gaps(
    support_mask: np.ndarray,
    observed_mask: np.ndarray,
    max_gap_area: int,
    min_neighbor_count: int,
) -> np.ndarray:
    """Promote tiny isolated gaps adjacent to support into the fillable support domain."""
    support = np.asarray(support_mask, dtype=bool)
    observed = np.asarray(observed_mask, dtype=bool)
    if max_gap_area <= 0:
        return support

    unsupported = ~support
    if not unsupported.any():
        return support

    kernel = np.ones((3, 3), dtype=np.int32)
    finite_neighbors = convolve(observed.astype(np.int32), kernel, mode="constant", cval=0)
    support_neighbors = convolve(support.astype(np.int32), kernel, mode="constant", cval=0)
    candidate = unsupported & (finite_neighbors >= int(min_neighbor_count)) & (support_neighbors >= int(min_neighbor_count))
    if not candidate.any():
        return support

    labels, num_labels = label(candidate)
    if num_labels == 0:
        return support

    component_sizes = np.bincount(labels.ravel())
    promoted = (labels > 0) & (component_sizes[labels] <= int(max_gap_area))
    promoted_support = support.copy()
    promoted_support[promoted] = True
    return promoted_support


def _build_tiny_hole_mask(
    surface: np.ndarray,
    candidate_domain: np.ndarray,
    max_hole_area: int,
    min_neighbor_count: int,
) -> np.ndarray:
    """Find tiny NaN holes surrounded by enough finite neighbors."""
    if max_hole_area <= 0:
        return np.zeros_like(surface, dtype=bool)

    finite = np.isfinite(surface)
    nan_mask = ~finite
    if not nan_mask.any():
        return np.zeros_like(surface, dtype=bool)

    kernel = np.ones((3, 3), dtype=np.int32)
    finite_neighbors = convolve(finite.astype(np.int32), kernel, mode="constant", cval=0)
    labels, num_labels = label(nan_mask)
    if num_labels == 0:
        return np.zeros_like(surface, dtype=bool)

    component_sizes = np.bincount(labels.ravel())
    component_neighbor_max = np.zeros(num_labels + 1, dtype=np.int32)
    np.maximum.at(component_neighbor_max, labels.ravel(), finite_neighbors.ravel())
    candidate = (
        (labels > 0)
        & (component_sizes[labels] <= int(max_hole_area))
        & (component_neighbor_max[labels] >= int(min_neighbor_count))
        & np.asarray(candidate_domain, dtype=bool)
    )
    return candidate.astype(bool)


def build_reference_dem(
    dem: np.ndarray,
    ground_density: np.ndarray,
    ground_penetration_ratio: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    method: str = "idw",
    smooth_sigma: float = 1.5,
    max_search_distance: float = 48.0,
    smoothing_iterations: int = 1,
    support_dilation: int = 0,
    support_gap_area: int = 4,
    support_gap_neighbor_count: int = 4,
    post_median_size: int = 3,
    post_median_blend: float = 0.75,
    final_hole_area: int = 8,
    final_hole_neighbor_count: int = 5,
    final_hole_search_distance: float = 12.0,
    final_hole_smoothing_iterations: int = 1,
    low_ground_density: float = 2.0,
    trusted_ground_density: float = 5.0,
    low_penetration_ratio: float = 0.03,
    trusted_penetration_ratio: float = 0.2,
    confidence_threshold: float = 0.55,
) -> dict[str, np.ndarray]:
    """Build a training-ready reference DEM from a raw DEM and density cues.

    The function preserves high-confidence observed cells, fully repairs missing
    cells, and softly blends low-confidence cells toward an interpolated surface.
    """
    base_dem = np.asarray(dem, dtype=np.float32)
    if observed_mask is None:
        observed = np.isfinite(base_dem)
    else:
        observed = np.asarray(observed_mask, dtype=bool) & np.isfinite(base_dem)

    if support_mask is None:
        support = observed.copy()
    else:
        support = np.asarray(support_mask, dtype=bool)
    if support_dilation > 0:
        support = binary_dilation(support, iterations=int(support_dilation))
    support |= observed
    support = _promote_small_support_gaps(
        support_mask=support,
        observed_mask=observed,
        max_gap_area=support_gap_area,
        min_neighbor_count=support_gap_neighbor_count,
    )

    density = np.nan_to_num(np.asarray(ground_density, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    density_conf = _linear_confidence(density, low_ground_density, trusted_ground_density)

    if ground_penetration_ratio is None:
        penetration_conf = np.ones_like(density_conf, dtype=np.float32)
    else:
        penetration = np.nan_to_num(
            np.asarray(ground_penetration_ratio, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        penetration_conf = _linear_confidence(penetration, low_penetration_ratio, trusted_penetration_ratio)

    base_confidence = ((0.7 * density_conf) + (0.3 * penetration_conf)).astype(np.float32)
    base_confidence[~support] = 0.0
    distance_to_observed = distance_transform_edt(~observed)
    fillable_missing_cells = support & (~observed)
    if float(max_search_distance) > 0.0:
        fillable_missing_cells &= distance_to_observed <= float(max_search_distance)

    low_confidence_cells = observed & support & (base_confidence < float(confidence_threshold))
    repair_mask = fillable_missing_cells | low_confidence_cells

    repaired_surface = interpolate_missing_values(
        base_dem,
        mask=repair_mask,
        method=method,
        smooth_sigma=smooth_sigma,
        max_search_distance=max_search_distance,
        smoothing_iterations=smoothing_iterations,
    )

    reference_dem = np.full_like(base_dem, np.nan, dtype=np.float32)
    reference_dem[support & observed] = base_dem[support & observed]
    reference_dem[fillable_missing_cells] = repaired_surface[fillable_missing_cells]
    if low_confidence_cells.any():
        blend_weight = np.clip(base_confidence[low_confidence_cells], 0.0, 1.0)
        reference_dem[low_confidence_cells] = (
            (blend_weight * base_dem[low_confidence_cells])
            + ((1.0 - blend_weight) * repaired_surface[low_confidence_cells])
        ).astype(np.float32)

    repair_weight = np.zeros_like(base_dem, dtype=np.float32)
    repair_weight[fillable_missing_cells] = 1.0
    repair_weight[low_confidence_cells] = 1.0 - base_confidence[low_confidence_cells]
    stabilization_weight = np.clip(repair_weight * float(post_median_blend), 0.0, 1.0).astype(np.float32)
    reference_dem = _stabilize_with_local_median(
        surface=reference_dem,
        support_mask=support,
        blend_weight=stabilization_weight,
        median_size=post_median_size,
    )
    tiny_hole_domain = binary_dilation(support, iterations=1)
    tiny_hole_mask = _build_tiny_hole_mask(
        surface=reference_dem,
        candidate_domain=tiny_hole_domain,
        max_hole_area=final_hole_area,
        min_neighbor_count=final_hole_neighbor_count,
    )
    if tiny_hole_mask.any():
        tiny_hole_filled = interpolate_missing_values(
            reference_dem,
            mask=tiny_hole_mask,
            method="idw",
            smooth_sigma=0.0,
            max_search_distance=final_hole_search_distance,
            smoothing_iterations=final_hole_smoothing_iterations,
        )
        reference_dem[tiny_hole_mask] = tiny_hole_filled[tiny_hole_mask]

    return {
        "dem_reference": reference_dem.astype(np.float32),
        "dem_valid_mask": observed.astype(np.float32),
        "dem_support_mask": support.astype(np.float32),
        "dem_lowconf_mask": low_confidence_cells.astype(np.float32),
        "dem_tinyhole_mask": tiny_hole_mask.astype(np.float32),
        "dem_repair_weight": np.clip(repair_weight, 0.0, 1.0).astype(np.float32),
        "dem_confidence": np.clip(base_confidence, 0.0, 1.0).astype(np.float32),
    }
