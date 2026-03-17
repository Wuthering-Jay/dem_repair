"""Rasterization and GeoTIFF helpers for LAS-derived DEM/DSM products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, label


DEFAULT_NODATA = -9999.0


@dataclass
class GridSpec:
    """Definition of a regular raster grid."""

    transform: Affine
    width: int
    height: int
    resolution: float
    center_min_x: float
    center_max_y: float
    crs: Any | None = None

    @property
    def num_cells(self) -> int:
        """Return the flattened raster cell count."""
        return int(self.width * self.height)


def build_grid_spec_from_points(
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    crs: Any | None = None,
) -> GridSpec:
    """Build a raster grid aligned to the supplied point cloud."""
    center_min_x = float(np.floor(np.nanmin(x) / resolution) * resolution)
    center_max_y = float(np.ceil(np.nanmax(y) / resolution) * resolution)
    width = int(np.floor((np.nanmax(x) - center_min_x) / resolution)) + 1
    height = int(np.floor((center_max_y - np.nanmin(y)) / resolution)) + 1
    transform = from_origin(center_min_x - (resolution / 2.0), center_max_y + (resolution / 2.0), resolution, resolution)
    return GridSpec(
        transform=transform,
        width=max(1, width),
        height=max(1, height),
        resolution=float(resolution),
        center_min_x=center_min_x,
        center_max_y=center_max_y,
        crs=crs,
    )


def build_grid_spec_from_bounds(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    resolution: float,
    crs: Any | None = None,
) -> GridSpec:
    """Build a raster grid from explicit bounds."""
    center_min_x = float(np.floor(min_x / resolution) * resolution)
    center_max_y = float(np.ceil(max_y / resolution) * resolution)
    width = int(np.floor((max_x - center_min_x) / resolution)) + 1
    height = int(np.floor((center_max_y - min_y) / resolution)) + 1
    transform = from_origin(center_min_x - (resolution / 2.0), center_max_y + (resolution / 2.0), resolution, resolution)
    return GridSpec(
        transform=transform,
        width=max(1, width),
        height=max(1, height),
        resolution=float(resolution),
        center_min_x=center_min_x,
        center_max_y=center_max_y,
        crs=crs,
    )


def points_to_indices(x: np.ndarray, y: np.ndarray, grid: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map point coordinates to row and column indices."""
    west = grid.transform.c
    north = grid.transform.f
    cols = np.floor((x - west) / grid.resolution).astype(np.int64)
    rows = np.floor((north - y) / grid.resolution).astype(np.int64)
    valid = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    return rows[valid], cols[valid], valid


def rasterize_points(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray | None,
    grid: GridSpec,
    aggregator: str = "mean",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Rasterize point values to a regular grid."""
    rows, cols, valid = points_to_indices(x, y, grid)
    if rows.size == 0:
        return np.full((grid.height, grid.width), fill_value, dtype=np.float32)

    flat_index = rows * grid.width + cols
    num_cells = grid.height * grid.width

    if aggregator == "count":
        counts = np.bincount(flat_index, minlength=num_cells).astype(np.float32)
        return counts.reshape(grid.height, grid.width)

    if values is None:
        raise ValueError(f"`values` is required for aggregator={aggregator}.")

    value_array = np.asarray(values, dtype=np.float64)[valid]

    if aggregator == "sum":
        sums = np.bincount(flat_index, weights=value_array, minlength=num_cells).astype(np.float32)
        return sums.reshape(grid.height, grid.width)

    if aggregator == "mean":
        sums = np.bincount(flat_index, weights=value_array, minlength=num_cells).astype(np.float64)
        counts = np.bincount(flat_index, minlength=num_cells).astype(np.float64)
        output = np.full(num_cells, fill_value, dtype=np.float32)
        valid_cells = counts > 0
        output[valid_cells] = (sums[valid_cells] / counts[valid_cells]).astype(np.float32)
        return output.reshape(grid.height, grid.width)

    if aggregator == "max":
        output = np.full(num_cells, -np.inf, dtype=np.float32)
        np.maximum.at(output, flat_index, value_array.astype(np.float32))
        output[~np.isfinite(output)] = fill_value
        return output.reshape(grid.height, grid.width)

    raise ValueError(f"Unsupported aggregator: {aggregator}")


def tin_interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: GridSpec,
    max_edge_length: float = 0.0,
) -> np.ndarray:
    """Interpolate irregular points to a regular grid using Delaunay TIN.

    Builds a Delaunay triangulation from the input points, then evaluates
    linear (barycentric) interpolation at every grid cell centre.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates and values of the source points.
    grid : GridSpec
        Target raster grid specification.
    max_edge_length : float
        If > 0, triangles whose longest edge exceeds this value are masked
        out (set to NaN) to avoid unreliable extrapolation across large gaps.

    Returns
    -------
    np.ndarray
        2-D float32 array of shape ``(grid.height, grid.width)``.
        Cells outside the convex hull or masked by *max_edge_length* are NaN.
    """
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    pts_xy = np.column_stack([
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    ])
    vals = np.asarray(z, dtype=np.float64)

    if pts_xy.shape[0] < 3:
        return np.full((grid.height, grid.width), np.nan, dtype=np.float32)

    tri = Delaunay(pts_xy)
    interp = LinearNDInterpolator(tri, vals)

    # Build grid cell centre coordinates
    west = grid.transform.c + grid.resolution / 2.0
    north = grid.transform.f - grid.resolution / 2.0
    col_coords = west + np.arange(grid.width, dtype=np.float64) * grid.resolution
    row_coords = north - np.arange(grid.height, dtype=np.float64) * grid.resolution
    grid_x, grid_y = np.meshgrid(col_coords, row_coords)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    result = interp(query).astype(np.float32).reshape(grid.height, grid.width)

    if max_edge_length > 0.0:
        # Mask out cells whose enclosing triangle has an edge longer than the
        # threshold — these are typically low-confidence extrapolations across
        # large voids.
        simplex_indices = tri.find_simplex(query)
        simplex_grid = simplex_indices.reshape(grid.height, grid.width)
        # Pre-compute per-simplex max edge length
        vertices = tri.points[tri.simplices]  # (n_simplices, 3, 2)
        edge_lengths = np.empty((vertices.shape[0], 3), dtype=np.float64)
        for ei, (i0, i1) in enumerate([(0, 1), (1, 2), (2, 0)]):
            diff = vertices[:, i0, :] - vertices[:, i1, :]
            edge_lengths[:, ei] = np.sqrt((diff ** 2).sum(axis=1))
        simplex_max_edge = edge_lengths.max(axis=1)

        valid_simplex = simplex_grid >= 0
        long_triangle = np.zeros_like(valid_simplex, dtype=bool)
        long_triangle[valid_simplex] = simplex_max_edge[simplex_grid[valid_simplex]] > max_edge_length
        result[long_triangle] = np.nan

    return result


def fill_small_holes(array: np.ndarray, max_area: int = 32) -> np.ndarray:
    """Fill only small invalid regions using nearest-neighbor propagation."""
    if max_area <= 0:
        return array.astype(np.float32, copy=True)

    output = array.astype(np.float32, copy=True)
    invalid = ~np.isfinite(output)
    if not invalid.any():
        return output

    labels, num_labels = label(invalid)
    if num_labels == 0:
        return output

    nearest_filled = nearest_fill(output)
    component_sizes = np.bincount(labels.ravel())
    small_component_mask = (labels > 0) & (component_sizes[labels] <= int(max_area))
    output[small_component_mask] = nearest_filled[small_component_mask]
    return output


def nearest_fill(array: np.ndarray) -> np.ndarray:
    """Fill invalid pixels using nearest valid neighbors."""
    output = array.astype(np.float32, copy=True)
    invalid = ~np.isfinite(output)
    if not invalid.any():
        return output
    valid = ~invalid
    if not valid.any():
        return np.zeros_like(output, dtype=np.float32)
    _, indices = distance_transform_edt(invalid, return_indices=True)
    filled = output[tuple(indices)]
    return filled.astype(np.float32)


def finalize_mean_raster(
    sum_values: np.ndarray,
    count_values: np.ndarray,
    grid: GridSpec,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Convert flattened sum/count accumulators into a raster mean."""
    sums = np.asarray(sum_values, dtype=np.float32).reshape(-1)
    counts = np.asarray(count_values, dtype=np.int64).reshape(-1)
    output = np.full_like(sums, fill_value, dtype=np.float32)
    valid = counts > 0
    output[valid] = sums[valid] / counts[valid].astype(np.float32)
    return output.reshape(grid.height, grid.width)


def finalize_max_raster(
    max_values: np.ndarray,
    grid: GridSpec,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Convert a flattened max accumulator into a 2D raster."""
    output = np.asarray(max_values, dtype=np.float32).reshape(grid.height, grid.width)
    output = output.copy()
    output[~np.isfinite(output)] = fill_value
    return output


def write_raster(
    path: str | Path,
    array: np.ndarray,
    grid: GridSpec,
    crs: Any | None = None,
    nodata: float = DEFAULT_NODATA,
    compress: str | None = "deflate",
) -> Path:
    """Write a single-band float raster."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raster = np.asarray(array, dtype=np.float32)
    write_array = np.where(np.isfinite(raster), raster, nodata).astype(np.float32)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "float32",
        "transform": grid.transform,
        "crs": crs if crs is not None else grid.crs,
        "nodata": nodata,
        "BIGTIFF": "IF_SAFER",
    }
    if compress is not None and str(compress).lower() != "none":
        profile["compress"] = str(compress).lower()
        if str(compress).lower() in {"deflate", "lzw", "zstd"}:
            profile["predictor"] = 3
    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(write_array, 1)
    return output_path


def write_named_rasters(
    rasters: Mapping[str, np.ndarray],
    output_dir: str | Path,
    tile_stem: str,
    grid: GridSpec,
    crs: Any | None = None,
    compress: str | None = "deflate",
) -> list[Path]:
    """Write a collection of named rasters with a shared grid."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, raster in rasters.items():
        outputs.append(write_raster(destination / f"{tile_stem}_{name}.tif", raster, grid, crs=crs, compress=compress))
    return outputs


def read_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band raster and convert nodata to NaN."""
    raster_path = Path(path)
    with rasterio.open(raster_path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            array[array == nodata] = np.nan
        profile = src.profile.copy()
        profile["crs"] = src.crs
        profile["transform"] = src.transform
    return array, profile


def read_raster_metadata(raster_path: str | Path) -> dict[str, Any]:
    """Read raster metadata without loading all values into memory."""
    path = Path(raster_path)
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "count": src.count,
            "crs": str(src.crs) if src.crs is not None else None,
            "nodata": src.nodata,
            "transform": tuple(src.transform),
        }


def get_nodata_mask(array: np.ndarray, nodata_value: float | None) -> np.ndarray:
    """Return a nodata mask for an array."""
    if nodata_value is None or np.isnan(nodata_value):
        return ~np.isfinite(array)
    return (~np.isfinite(array)) | (array == nodata_value)
