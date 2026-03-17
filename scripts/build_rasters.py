"""Generate reference DEM/DSM rasters and LAS-derived statistics from raw LAS tiles."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.interpolation import build_reference_dem, interpolate_missing_values
from utils.config_utils import load_yaml, resolve_path
from utils.io_utils import list_files
from utils.las_utils import iter_las_chunks, read_las_header_info
from utils.raster_utils import (
    build_grid_spec_from_bounds,
    fill_small_holes,
    finalize_max_raster,
    finalize_mean_raster,
    nearest_fill,
    points_to_indices,
    tin_interpolate_to_grid,
    write_raster,
)


AUXILIARY_RASTER_NAMES = [
    "all_density",
    "ground_density",
    "vegetation_density",
    "building_density",
    "mean_intensity",
    "mean_return_num",
    "last_return_ratio",
    "ground_penetration_ratio",
]

DIAGNOSTIC_RASTER_NAMES = [
    "dem_valid_mask",
    "dem_support_mask",
    "dem_lowconf_mask",
    "dem_tinyhole_mask",
    "dem_repair_weight",
    "dem_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DEM/DSM and auxiliary rasters from LAS.")
    parser.add_argument("--config", default="configs/data/dataset.yaml", help="Dataset config path.")
    parser.add_argument("--classes", default="configs/data/classes.yaml", help="Class mapping config path.")
    parser.add_argument("--split", default=None, help="Optional split to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rasters.")
    return parser.parse_args()


def _aggregate_chunk(
    flat_index: np.ndarray,
    max_values: np.ndarray,
    payloads: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Aggregate per-point payloads onto unique raster cells for one chunk."""
    if flat_index.size == 0:
        empty_index = np.empty(0, dtype=np.int64)
        empty_payloads = {
            "count": np.empty(0, dtype=np.int32),
            "max_z": np.empty(0, dtype=np.float32),
            **{name: np.empty(0, dtype=np.float32) for name in payloads},
        }
        return empty_index, empty_payloads

    order = np.argsort(flat_index, kind="mergesort")
    flat_sorted = flat_index[order]
    unique_cells, start_idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)

    aggregated: dict[str, np.ndarray] = {
        "count": counts.astype(np.int32, copy=False),
    }
    for name, values in payloads.items():
        values_sorted = np.asarray(values, dtype=np.float32)[order]
        aggregated[name] = np.add.reduceat(values_sorted, start_idx).astype(np.float32, copy=False)

    max_values_sorted = np.asarray(max_values, dtype=np.float32)[order]
    aggregated["max_z"] = np.maximum.reduceat(max_values_sorted, start_idx).astype(np.float32, copy=False)
    return unique_cells.astype(np.int64, copy=False), aggregated


def compute_tile_rasters_streaming(
    las_path: Path,
    header_info: dict[str, Any],
    class_map: dict[str, int],
    resolution: float,
    small_gap_area: int,
    reference_cfg: dict[str, Any],
    chunk_size: int,
    chunk_progress_interval: int,
    progress_prefix: str,
    dem_generation_method: str = "ground_mean",
    tin_max_edge_length: float = 0.0,
) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
    """Stream a LAS tile chunk by chunk and accumulate DEM/DSM/stat rasters.

    Parameters
    ----------
    dem_generation_method : str
        ``"ground_mean"`` (default) uses per-cell mean of ground Z values.
        ``"tin"`` builds a Delaunay TIN from all ground points and linearly
        interpolates onto the raster grid.
    tin_max_edge_length : float
        When using TIN, triangles with any edge longer than this value (in
        map units) are masked out.  Set to 0 to keep all triangles.
    """
    min_x, min_y = float(header_info["mins"][0]), float(header_info["mins"][1])
    max_x, max_y = float(header_info["maxs"][0]), float(header_info["maxs"][1])
    grid = build_grid_spec_from_bounds(min_x, max_x, min_y, max_y, resolution, crs=header_info.get("crs"))
    num_cells = grid.num_cells
    total_points = int(header_info["point_count"])
    use_tin = dem_generation_method == "tin"

    ground_id = int(class_map["ground"])
    vegetation_id = int(class_map["vegetation"])
    building_id = int(class_map["building"])
    supported_ids = np.array([ground_id, vegetation_id, building_id], dtype=np.int16)

    dem_sum = np.zeros(num_cells, dtype=np.float32)
    dem_count = np.zeros(num_cells, dtype=np.int32)
    dsm_max = np.full(num_cells, -np.inf, dtype=np.float32)
    all_count = np.zeros(num_cells, dtype=np.int32)
    ground_count = np.zeros(num_cells, dtype=np.int32)
    vegetation_count = np.zeros(num_cells, dtype=np.int32)
    building_count = np.zeros(num_cells, dtype=np.int32)
    intensity_sum = np.zeros(num_cells, dtype=np.float32)
    return_sum = np.zeros(num_cells, dtype=np.float32)
    last_return_sum = np.zeros(num_cells, dtype=np.float32)

    processed_points = 0
    valid_surface_points = 0
    ground_points = 0

    # Accumulate raw ground point coordinates for TIN-based DEM generation.
    # Only used when dem_generation_method == "tin".
    ground_x_chunks: list[np.ndarray] = []
    ground_y_chunks: list[np.ndarray] = []
    ground_z_chunks: list[np.ndarray] = []

    for chunk_index, chunk in enumerate(iter_las_chunks(las_path, chunk_size), start=1):
        processed_points += int(chunk["point_count"])
        classification = chunk["classification"]
        supported_mask = np.isin(classification, supported_ids)
        if not supported_mask.any():
            if chunk_index == 1 or chunk_index % max(1, chunk_progress_interval) == 0 or processed_points >= total_points:
                pct = (processed_points / max(total_points, 1)) * 100.0
                print(f"{progress_prefix} chunk={chunk_index} processed={processed_points:,}/{total_points:,} ({pct:.1f}%)")
            continue

        surface_x = chunk["x"][supported_mask]
        surface_y = chunk["y"][supported_mask]
        rows, cols, spatial_valid = points_to_indices(surface_x, surface_y, grid)
        if rows.size == 0:
            continue

        surface_z = chunk["z"][supported_mask][spatial_valid]
        surface_intensity = chunk["intensity"][supported_mask][spatial_valid]
        surface_return_int = chunk["return_num"][supported_mask][spatial_valid]
        surface_return = surface_return_int.astype(np.float32)
        surface_num_returns = chunk["num_returns"][supported_mask][spatial_valid]
        surface_class = classification[supported_mask][spatial_valid]
        flat_index = (rows * grid.width) + cols

        valid_surface_points += int(surface_z.size)
        ground_mask = surface_class == ground_id
        vegetation_mask = surface_class == vegetation_id
        building_mask = surface_class == building_id
        ground_points += int(ground_mask.sum())

        # Collect raw ground point coordinates for TIN DEM generation.
        if use_tin and ground_mask.any():
            gx = chunk["x"][supported_mask][spatial_valid][ground_mask]
            gy = chunk["y"][supported_mask][spatial_valid][ground_mask]
            gz = surface_z[ground_mask]
            ground_x_chunks.append(gx.astype(np.float64))
            ground_y_chunks.append(gy.astype(np.float64))
            ground_z_chunks.append(gz.astype(np.float32))

        payloads = {
            "intensity_sum": surface_intensity,
            "return_sum": surface_return,
            "last_return_sum": (surface_return_int == surface_num_returns).astype(np.float32),
            "ground_count": ground_mask.astype(np.float32),
            "vegetation_count": vegetation_mask.astype(np.float32),
            "building_count": building_mask.astype(np.float32),
            "ground_z_sum": np.where(ground_mask, surface_z, 0.0).astype(np.float32),
        }
        unique_cells, aggregated = _aggregate_chunk(flat_index, surface_z, payloads)
        if unique_cells.size == 0:
            continue

        dsm_max[unique_cells] = np.maximum(dsm_max[unique_cells], aggregated["max_z"])
        all_count[unique_cells] += aggregated["count"]
        ground_count[unique_cells] += aggregated["ground_count"].astype(np.int32, copy=False)
        vegetation_count[unique_cells] += aggregated["vegetation_count"].astype(np.int32, copy=False)
        building_count[unique_cells] += aggregated["building_count"].astype(np.int32, copy=False)
        dem_count[unique_cells] += aggregated["ground_count"].astype(np.int32, copy=False)
        dem_sum[unique_cells] += aggregated["ground_z_sum"]
        intensity_sum[unique_cells] += aggregated["intensity_sum"]
        return_sum[unique_cells] += aggregated["return_sum"]
        last_return_sum[unique_cells] += aggregated["last_return_sum"]

        if chunk_index == 1 or chunk_index % max(1, chunk_progress_interval) == 0 or processed_points >= total_points:
            pct = (processed_points / max(total_points, 1)) * 100.0
            print(
                f"{progress_prefix} chunk={chunk_index} processed={processed_points:,}/{total_points:,} "
                f"({pct:.1f}%) valid_surface={valid_surface_points:,} ground={ground_points:,}"
            )

    if ground_points <= 0:
        raise ValueError("No ground points found for DEM generation.")
    if valid_surface_points <= 0:
        raise ValueError("No valid surface points remain after default filtering.")

    if use_tin:
        print(f"{progress_prefix} postprocess: build DEM via Delaunay TIN "
              f"(ground_points={ground_points:,}, max_edge={tin_max_edge_length})")
        all_gx = np.concatenate(ground_x_chunks) if ground_x_chunks else np.empty(0, dtype=np.float64)
        all_gy = np.concatenate(ground_y_chunks) if ground_y_chunks else np.empty(0, dtype=np.float64)
        all_gz = np.concatenate(ground_z_chunks) if ground_z_chunks else np.empty(0, dtype=np.float32)
        # Free chunk lists early to reduce peak memory
        del ground_x_chunks, ground_y_chunks, ground_z_chunks
        dem_raw = tin_interpolate_to_grid(all_gx, all_gy, all_gz, grid,
                                         max_edge_length=tin_max_edge_length)
        del all_gx, all_gy, all_gz
    else:
        print(f"{progress_prefix} postprocess: finalize DEM mean raster")
        dem_raw = finalize_mean_raster(dem_sum, dem_count, grid)

    # "observed" = cells where actual ground LiDAR points exist.
    raw_observed_mask = (
        dem_count.reshape(grid.height, grid.width) > 0
    )

    if small_gap_area > 0:
        print(f"{progress_prefix} postprocess: fill DEM small holes "
              f"(max_area={small_gap_area})")
        dem_prefilled = fill_small_holes(dem_raw, max_area=small_gap_area)
    else:
        dem_prefilled = dem_raw.astype(np.float32, copy=True)

    print(f"{progress_prefix} postprocess: finalize DSM max raster")
    dsm = finalize_max_raster(dsm_max, grid)
    dsm_nan_count = int(np.isnan(dsm).sum())
    if dsm_nan_count > 0:
        print(f"{progress_prefix} postprocess: fill {dsm_nan_count:,} "
              f"DSM NaN cells with nearest-neighbor")
        dsm = nearest_fill(dsm)
    print(f"{progress_prefix} postprocess: finalize density/stat rasters")
    cell_area = resolution * resolution
    all_density = (
        all_count.reshape(grid.height, grid.width) / cell_area
    ).astype(np.float32)
    ground_density = (
        ground_count.reshape(grid.height, grid.width) / cell_area
    ).astype(np.float32)
    vegetation_density = (
        vegetation_count.reshape(grid.height, grid.width) / cell_area
    ).astype(np.float32)
    building_density = (
        building_count.reshape(grid.height, grid.width) / cell_area
    ).astype(np.float32)
    mean_intensity = finalize_mean_raster(intensity_sum, all_count, grid)
    mean_return_num = finalize_mean_raster(return_sum, all_count, grid)
    last_return_ratio = finalize_mean_raster(
        last_return_sum, all_count, grid, fill_value=0.0,
    )
    ground_penetration_ratio = np.divide(
        ground_density,
        np.maximum(all_density, 1.0),
        out=np.zeros_like(ground_density, dtype=np.float32),
        where=all_density > 0,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Build reference DEM
    # ------------------------------------------------------------------
    if use_tin:
        # For TIN mode we bypass build_reference_dem entirely.
        # The TIN surface is already a geometrically correct continuous
        # interpolation; feeding it through the density-based confidence /
        # IDW repair pipeline only degrades quality (ground_density=0 for
        # TIN-only cells → low confidence → IDW overwrites TIN values).
        #
        # Instead:  TIN dem_raw  →  IDW fill remaining NaN gaps  →  done.
        print(f"{progress_prefix} postprocess: TIN mode – "
              f"fill remaining gaps with IDW")
        tin_gaps = ~np.isfinite(dem_prefilled)
        tin_gap_count = int(tin_gaps.sum())
        if tin_gap_count > 0:
            dem_reference = interpolate_missing_values(
                dem_prefilled,
                mask=None,
                method=str(reference_cfg.get("method", "idw")),
                smooth_sigma=float(
                    reference_cfg.get("smooth_sigma", 1.5)),
                max_search_distance=float(
                    reference_cfg.get("max_search_distance", 48.0)),
                smoothing_iterations=int(
                    reference_cfg.get("smoothing_iterations", 1)),
            )
        else:
            dem_reference = dem_prefilled.copy()

        # Also fill dem_raw NaN gaps so it is a complete surface.
        dem_raw_filled = dem_reference.copy()

        print(f"{progress_prefix} postprocess: TIN gaps filled: "
              f"{tin_gap_count:,} cells")

        tin_covered = np.isfinite(dem_raw)
        support_mask_out = (all_density > 0) | tin_covered
        confidence_out = np.where(
            raw_observed_mask, 1.0,
            np.where(tin_covered, 0.7, 0.0),
        ).astype(np.float32)

        reference_products = {
            "dem_reference": dem_reference.astype(np.float32),
            "dem_valid_mask": raw_observed_mask.astype(np.float32),
            "dem_support_mask": support_mask_out.astype(np.float32),
            "dem_lowconf_mask": np.zeros_like(
                dem_reference, dtype=np.float32),
            "dem_tinyhole_mask": np.zeros_like(
                dem_reference, dtype=np.float32),
            "dem_repair_weight": (~tin_covered).astype(np.float32),
            "dem_confidence": confidence_out,
        }
    else:
        support_mask_input = all_density > 0
        print(f"{progress_prefix} postprocess: build reference DEM "
              f"(density-aware repair)")
        reference_products = build_reference_dem(
            dem=dem_prefilled,
            ground_density=ground_density,
            ground_penetration_ratio=ground_penetration_ratio,
            observed_mask=raw_observed_mask,
            support_mask=support_mask_input,
            method=str(reference_cfg.get("method", "idw")),
            smooth_sigma=float(
                reference_cfg.get("smooth_sigma", 1.5)),
            max_search_distance=float(
                reference_cfg.get("max_search_distance", 48.0)),
            smoothing_iterations=int(
                reference_cfg.get("smoothing_iterations", 1)),
            support_dilation=int(
                reference_cfg.get("support_dilation", 0)),
            support_gap_area=int(
                reference_cfg.get("support_gap_area", 4)),
            support_gap_neighbor_count=int(
                reference_cfg.get("support_gap_neighbor_count", 4)),
            post_median_size=int(
                reference_cfg.get("post_median_size", 3)),
            post_median_blend=float(
                reference_cfg.get("post_median_blend", 0.75)),
            final_hole_area=int(
                reference_cfg.get("final_hole_area", 8)),
            final_hole_neighbor_count=int(
                reference_cfg.get("final_hole_neighbor_count", 5)),
            final_hole_search_distance=float(
                reference_cfg.get("final_hole_search_distance", 12.0)),
            final_hole_smoothing_iterations=int(
                reference_cfg.get(
                    "final_hole_smoothing_iterations", 1)),
            low_ground_density=float(
                reference_cfg.get("low_ground_density", 2.0)),
            trusted_ground_density=float(
                reference_cfg.get("trusted_ground_density", 5.0)),
            low_penetration_ratio=float(
                reference_cfg.get("low_penetration_ratio", 0.03)),
            trusted_penetration_ratio=float(
                reference_cfg.get("trusted_penetration_ratio", 0.2)),
            confidence_threshold=float(
                reference_cfg.get("confidence_threshold", 0.55)),
        )
        dem_raw_filled = dem_raw

    rasters = {
        "dem": reference_products["dem_reference"].astype(np.float32),
        "dem_raw": dem_raw_filled.astype(np.float32),
        "dem_reference": reference_products[
            "dem_reference"].astype(np.float32),
        "dsm": dsm.astype(np.float32),
        "all_density": all_density,
        "ground_density": ground_density,
        "vegetation_density": vegetation_density,
        "building_density": building_density,
        "mean_intensity": mean_intensity.astype(np.float32),
        "mean_return_num": mean_return_num.astype(np.float32),
        "last_return_ratio": last_return_ratio.astype(np.float32),
        "ground_penetration_ratio": ground_penetration_ratio.astype(
            np.float32),
        "dem_valid_mask": reference_products[
            "dem_valid_mask"].astype(np.float32),
        "dem_support_mask": reference_products[
            "dem_support_mask"].astype(np.float32),
        "dem_lowconf_mask": reference_products[
            "dem_lowconf_mask"].astype(np.float32),
        "dem_tinyhole_mask": reference_products[
            "dem_tinyhole_mask"].astype(np.float32),
        "dem_repair_weight": reference_products[
            "dem_repair_weight"].astype(np.float32),
        "dem_confidence": reference_products[
            "dem_confidence"].astype(np.float32),
    }
    support_mask = reference_products["dem_support_mask"] > 0.5
    reference_valid = np.isfinite(
        reference_products["dem_reference"])
    missing_cells = int((~raw_observed_mask).sum())
    stats = {
        "total_points": total_points,
        "valid_surface_points": valid_surface_points,
        "ground_points": ground_points,
        "grid_width": grid.width,
        "grid_height": grid.height,
        "dem_observed_cells": int(raw_observed_mask.sum()),
        "dem_missing_cells": missing_cells,
        "dem_support_cells": int(support_mask.sum()),
        "dem_lowconf_cells": int(
            (reference_products["dem_lowconf_mask"] > 0.5).sum()),
        "dem_tinyhole_cells": int(
            (reference_products["dem_tinyhole_mask"] > 0.5).sum()),
        "dem_reference_valid_cells": int(reference_valid.sum()),
        "dem_unresolved_cells": max(
            0, int(support_mask.sum()) - int(reference_valid.sum())),
    }
    return rasters, grid, stats


def main() -> None:
    args = parse_args()
    dataset_config = load_yaml(args.config)
    class_map = load_yaml(args.classes)

    raw_las_root = resolve_path(dataset_config["raw_las_dir"])
    raw_dem_root = resolve_path(dataset_config["raw_dem_dir"])
    raw_dsm_root = resolve_path(dataset_config["raw_dsm_dir"])
    processed_raster_root = resolve_path(dataset_config["processed_raster_dir"])
    splits = [args.split] if args.split else [dataset_config["train_split"], dataset_config["val_split"], dataset_config["test_split"]]
    chunk_size = int(dataset_config.get("las_chunk_size", 1_000_000))
    chunk_progress_interval = int(dataset_config.get("chunk_progress_interval", 5))
    raster_compression = dataset_config.get("raster_compression", "deflate")
    reference_cfg = {
        "method": dataset_config.get("reference_dem_interpolation_method", "idw"),
        "smooth_sigma": dataset_config.get("reference_dem_smooth_sigma", 1.5),
        "max_search_distance": dataset_config.get("reference_dem_max_search_distance", 48.0),
        "smoothing_iterations": dataset_config.get("reference_dem_smoothing_iterations", 1),
        "support_dilation": dataset_config.get("reference_dem_support_dilation", 0),
        "support_gap_area": dataset_config.get("reference_dem_support_gap_area", 4),
        "support_gap_neighbor_count": dataset_config.get("reference_dem_support_gap_neighbor_count", 4),
        "post_median_size": dataset_config.get("reference_dem_post_median_size", 3),
        "post_median_blend": dataset_config.get("reference_dem_post_median_blend", 0.75),
        "final_hole_area": dataset_config.get("reference_dem_final_hole_area", 8),
        "final_hole_neighbor_count": dataset_config.get("reference_dem_final_hole_neighbor_count", 5),
        "final_hole_search_distance": dataset_config.get("reference_dem_final_hole_search_distance", 12.0),
        "final_hole_smoothing_iterations": dataset_config.get("reference_dem_final_hole_smoothing_iterations", 1),
        "low_ground_density": dataset_config.get("reference_dem_low_ground_density", 2.0),
        "trusted_ground_density": dataset_config.get("reference_dem_trusted_ground_density", 5.0),
        "low_penetration_ratio": dataset_config.get("reference_dem_low_penetration_ratio", 0.03),
        "trusted_penetration_ratio": dataset_config.get("reference_dem_trusted_penetration_ratio", 0.2),
        "confidence_threshold": dataset_config.get("reference_dem_confidence_threshold", 0.55),
    }

    for split in splits:
        las_dir = raw_las_root / split
        dem_dir = raw_dem_root / split
        dsm_dir = raw_dsm_root / split
        aux_dir = processed_raster_root / split
        las_files = list_files(las_dir, patterns=["*.las", "*.laz"])
        print(f"[build_rasters] split={split} tiles={len(las_files)}")
        split_start_time = time.perf_counter()
        for index, las_path in enumerate(las_files, start=1):
            tile_stem = las_path.stem
            dem_path = dem_dir / f"{tile_stem}_dem.tif"
            dem_raw_path = dem_dir / f"{tile_stem}_dem_raw.tif"
            dem_reference_path = dem_dir / f"{tile_stem}_dem_reference.tif"
            dsm_path = dsm_dir / f"{tile_stem}_dsm.tif"
            aux_outputs = [aux_dir / f"{tile_stem}_{name}.tif" for name in AUXILIARY_RASTER_NAMES]
            diagnostic_outputs = [aux_dir / f"{tile_stem}_{name}.tif" for name in DIAGNOSTIC_RASTER_NAMES]
            progress_prefix = f"[build_rasters][split={split}][{index}/{len(las_files)}][tile={tile_stem}]"
            required_outputs = [dem_path, dem_raw_path, dem_reference_path, dsm_path, *aux_outputs, *diagnostic_outputs]
            if not args.overwrite and all(path.exists() for path in required_outputs):
                print(f"{progress_prefix} skip: outputs already exist")
                continue

            tile_start_time = time.perf_counter()
            file_size_gb = las_path.stat().st_size / (1024**3)
            header_info = read_las_header_info(las_path)
            point_count_m = header_info["point_count"] / 1_000_000.0
            grid_preview = build_grid_spec_from_bounds(
                float(header_info["mins"][0]),
                float(header_info["maxs"][0]),
                float(header_info["mins"][1]),
                float(header_info["maxs"][1]),
                float(dataset_config["resolution"]),
                crs=header_info.get("crs"),
            )
            print(
                f"{progress_prefix} start: source={las_path.name} size={file_size_gb:.2f} GB "
                f"points={point_count_m:.1f}M grid={grid_preview.width}x{grid_preview.height} "
                f"chunk_size={chunk_size:,}"
            )
            rasters, grid, stats = compute_tile_rasters_streaming(
                las_path=las_path,
                header_info=header_info,
                class_map=class_map,
                resolution=float(dataset_config["resolution"]),
                small_gap_area=int(dataset_config.get("small_gap_area", 32)),
                reference_cfg=reference_cfg,
                chunk_size=chunk_size,
                chunk_progress_interval=chunk_progress_interval,
                progress_prefix=progress_prefix,
                dem_generation_method=str(dataset_config.get("dem_generation_method", "ground_mean")),
                tin_max_edge_length=float(dataset_config.get("tin_max_edge_length", 0.0)),
            )
            print(f"{progress_prefix} write: DEM(reference) -> {dem_path.name}")
            write_raster(dem_path, rasters["dem_reference"], grid, crs=header_info.get("crs"), compress=raster_compression)
            print(f"{progress_prefix} write: DEM raw -> {dem_raw_path.name}")
            write_raster(dem_raw_path, rasters["dem_raw"], grid, crs=header_info.get("crs"), compress=raster_compression)
            print(f"{progress_prefix} write: DEM reference -> {dem_reference_path.name}")
            write_raster(
                dem_reference_path,
                rasters["dem_reference"],
                grid,
                crs=header_info.get("crs"),
                compress=raster_compression,
            )
            print(f"{progress_prefix} write: DSM -> {dsm_path.name}")
            write_raster(dsm_path, rasters["dsm"], grid, crs=header_info.get("crs"), compress=raster_compression)
            for aux_index, name in enumerate(AUXILIARY_RASTER_NAMES, start=1):
                aux_path = aux_dir / f"{tile_stem}_{name}.tif"
                print(f"{progress_prefix} write: aux {aux_index}/{len(AUXILIARY_RASTER_NAMES)} -> {aux_path.name}")
                write_raster(
                    aux_path,
                    rasters[name],
                    grid,
                    crs=header_info.get("crs"),
                    compress=raster_compression,
                )
            for diag_index, name in enumerate(DIAGNOSTIC_RASTER_NAMES, start=1):
                diagnostic_path = aux_dir / f"{tile_stem}_{name}.tif"
                print(f"{progress_prefix} write: diagnostic {diag_index}/{len(DIAGNOSTIC_RASTER_NAMES)} -> {diagnostic_path.name}")
                write_raster(
                    diagnostic_path,
                    rasters[name],
                    grid,
                    crs=header_info.get("crs"),
                    compress=raster_compression,
                )
            tile_elapsed = time.perf_counter() - tile_start_time
            print(
                f"{progress_prefix} done: elapsed={tile_elapsed:.1f}s "
                f"outputs=DEM(raw/reference)+DSM+{len(AUXILIARY_RASTER_NAMES)} aux+{len(DIAGNOSTIC_RASTER_NAMES)} diagnostics "
                f"valid_surface={stats['valid_surface_points']:,} ground={stats['ground_points']:,} "
                f"raw_missing_cells={stats['dem_missing_cells']:,} lowconf_cells={stats['dem_lowconf_cells']:,} "
                f"support_cells={stats['dem_support_cells']:,} tinyhole_cells={stats['dem_tinyhole_cells']:,} "
                f"unresolved_cells={stats['dem_unresolved_cells']:,}"
            )
        split_elapsed = time.perf_counter() - split_start_time
        print(f"[build_rasters] split={split} completed in {split_elapsed / 60.0:.1f} min")


if __name__ == "__main__":
    main()
