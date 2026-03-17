"""Build NPZ training samples by simulating DEM voids from reference rasters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.confidence import build_confidence_map
from utils.config_utils import load_yaml, resolve_path
from utils.interpolation import interpolate_coarse_dem, interpolate_missing_values
from utils.io_utils import ensure_dir, save_npz
from utils.mask_utils import apply_mask_to_dem, simulate_void_mask
from utils.raster_utils import read_raster
from utils.terrain_features import compute_terrain_feature_stack
from utils.vis_utils import save_raster_grid


REQUIRED_AUX_MAPS = [
    "all_density",
    "ground_density",
    "vegetation_density",
    "building_density",
    "mean_intensity",
    "mean_return_num",
    "last_return_ratio",
    "ground_penetration_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DEM repair NPZ samples.")
    parser.add_argument("--config", default="configs/data/dataset.yaml", help="Dataset config path.")
    parser.add_argument("--split", default=None, help="Optional split to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing samples.")
    return parser.parse_args()


def _resolve_dem_tiles(split_dem_dir: Path, prefer_reference: bool) -> list[tuple[str, Path, str]]:
    """Resolve one DEM source per tile, preferring training-ready reference DEMs."""
    dem_sources: dict[str, tuple[Path, str]] = {}

    if prefer_reference:
        for path in sorted(split_dem_dir.glob("*_dem_reference.tif")):
            tile_stem = path.stem.removesuffix("_dem_reference")
            dem_sources[tile_stem] = (path, "reference")

    for path in sorted(split_dem_dir.glob("*_dem.tif")):
        if path.stem.endswith("_dem_reference") or path.stem.endswith("_dem_raw"):
            continue
        tile_stem = path.stem.removesuffix("_dem")
        dem_sources.setdefault(tile_stem, (path, "default"))

    if not dem_sources:
        return []
    return [(tile_stem, path, variant) for tile_stem, (path, variant) in sorted(dem_sources.items())]


def _sliding_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def _crop_with_pad(array: np.ndarray, top: int, left: int, tile_size: int, pad_value: float) -> np.ndarray:
    height, width = array.shape
    bottom = min(height, top + tile_size)
    right = min(width, left + tile_size)
    crop = array[top:bottom, left:right]
    pad_h = tile_size - crop.shape[0]
    pad_w = tile_size - crop.shape[1]
    if pad_h == 0 and pad_w == 0:
        return crop.astype(np.float32)
    return np.pad(crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=pad_value).astype(np.float32)


def _load_auxiliary_maps(aux_dir: Path, tile_stem: str) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for name in REQUIRED_AUX_MAPS:
        raster_path = aux_dir / f"{tile_stem}_{name}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(f"Missing auxiliary raster: {raster_path}")
        maps[name], _ = read_raster(raster_path)
    return maps


def _prepare_reference_crop(gt_crop: np.ndarray, min_valid_ratio: float) -> tuple[np.ndarray, np.ndarray] | None:
    valid_mask = np.isfinite(gt_crop)
    if float(valid_mask.mean()) < min_valid_ratio:
        return None
    filled_gt = interpolate_missing_values(gt_crop, method="nearest")
    return filled_gt.astype(np.float32), valid_mask


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dem_root = resolve_path(config["raw_dem_dir"])
    dsm_root = resolve_path(config["raw_dsm_dir"])
    raster_root = resolve_path(config["processed_raster_dir"])
    sample_root = resolve_path(config["processed_sample_dir"])
    figure_root = resolve_path("outputs/figures/samples")

    splits = [args.split] if args.split else [config["train_split"], config["val_split"], config["test_split"]]
    tile_size = int(config["tile_size"])
    stride = int(config["stride"])
    openness_scales = [int(value) for value in config.get("openness_scales", [3, 7, 15, 31])]
    void_strategies = list(config.get("void_strategies", ["mixed"]))
    void_coverage_range = tuple(float(value) for value in config.get("void_coverage_range", [0.05, 0.22]))
    min_mask_ratio = float(config.get("sample_min_mask_ratio", 0.03))
    max_mask_ratio = float(config.get("sample_max_mask_ratio", 0.35))
    min_valid_ratio = float(config.get("sample_min_valid_ratio", 0.95))
    interpolation_cfg = config.get("coarse_interpolation", {})
    visualization_budget = int(config.get("visualization_max_samples", 4))
    vis_count = 0
    prefer_reference_dem = bool(config.get("prefer_reference_dem", True))

    for split in splits:
        split_dem_dir = dem_root / split
        split_dsm_dir = dsm_root / split
        split_aux_dir = raster_root / split
        split_sample_dir = ensure_dir(sample_root / split)

        dem_products = _resolve_dem_tiles(split_dem_dir, prefer_reference=prefer_reference_dem)
        reference_tiles = sum(1 for _, _, variant in dem_products if variant == "reference")
        default_tiles = len(dem_products) - reference_tiles
        print(
            f"[build_samples] split={split} dem_tiles={len(dem_products)} "
            f"reference={reference_tiles} fallback_default={default_tiles}"
        )
        sample_counter = 0

        for tile_idx, (tile_stem, dem_path, dem_variant) in enumerate(dem_products, 1):
            tile_samples = 0
            print(f"  [{tile_idx}/{len(dem_products)}] "
                  f"{tile_stem} loading...",
                  flush=True)
            dsm_path = (split_dsm_dir
                        / f"{tile_stem}_dsm.tif")
            gt_dem, dem_profile = read_raster(dem_path)
            print(f"    DEM loaded: {gt_dem.shape}",
                  flush=True)
            dsm = np.zeros_like(
                gt_dem, dtype=np.float32)
            if bool(config.get("use_dsm", True)):
                if not dsm_path.exists():
                    raise FileNotFoundError(
                        f"Missing DSM: {dsm_path}")
                dsm, _ = read_raster(dsm_path)
                print("    DSM loaded", flush=True)
                dsm = interpolate_missing_values(
                    dsm, method="nearest")
                print("    DSM interpolated", flush=True)

            auxiliary_maps = _load_auxiliary_maps(
                split_aux_dir, tile_stem)
            print(f"    aux loaded: "
                  f"{len(auxiliary_maps)} maps",
                  flush=True)
            row_positions = _sliding_positions(
                gt_dem.shape[0], tile_size, stride)
            col_positions = _sliding_positions(
                gt_dem.shape[1], tile_size, stride)
            total_positions = (
                len(row_positions)
                * len(col_positions))
            print(f"    positions: {total_positions}",
                  flush=True)
            position_count = 0

            import time as _time
            for top in row_positions:
                for left in col_positions:
                    position_count += 1
                    _t0 = _time.time()

                    _dbg = True
                    gt_crop_raw = _crop_with_pad(
                        gt_dem, top, left,
                        tile_size, pad_value=np.nan)
                    reference = _prepare_reference_crop(
                        gt_crop_raw,
                        min_valid_ratio=min_valid_ratio)
                    if reference is None:
                        if _dbg:
                            print(f"    pos {position_count}: "
                                  f"skip (invalid)",
                                  flush=True)
                        continue
                    gt_crop, reference_valid_mask = reference
                    if _dbg:
                        print(f"    pos {position_count}: "
                              f"ref ok", flush=True)
                    dsm_crop = _crop_with_pad(
                        dsm, top, left,
                        tile_size, pad_value=np.nan)
                    dsm_crop = interpolate_missing_values(
                        dsm_crop, method="nearest")
                    aux_crops = {
                        name: _crop_with_pad(
                            raster, top, left,
                            tile_size, pad_value=np.nan)
                        for name, raster
                        in auxiliary_maps.items()
                    }
                    if _dbg:
                        print(f"    pos {position_count}: "
                              f"crops done", flush=True)

                    tile_rng_seed = (
                        abs(hash((tile_stem, split,
                                  top, left)))
                        % (2**32))
                    strategy = str(
                        np.random.default_rng(
                            tile_rng_seed,
                        ).choice(void_strategies))
                    mask = simulate_void_mask(
                        valid_mask=reference_valid_mask,
                        vegetation_density=aux_crops.get(
                            "vegetation_density"),
                        strategy=strategy,
                        coverage_range=void_coverage_range,
                        seed=tile_rng_seed,
                    )
                    mask_ratio = float(mask.mean())
                    if _dbg:
                        print(f"    pos {position_count}: "
                              f"mask {mask_ratio:.2f}",
                              flush=True)
                    if (mask_ratio < min_mask_ratio
                            or mask_ratio > max_mask_ratio):
                        continue

                    partial_dem = apply_mask_to_dem(
                        gt_crop, mask, fill_value=np.nan)
                    if _dbg:
                        print(f"    pos {position_count}: "
                              f"coarse start", flush=True)
                    coarse_dem = interpolate_coarse_dem(
                        partial_dem,
                        mask=mask,
                        method=str(interpolation_cfg.get(
                            "method", "nearest_gaussian")),
                        smooth_sigma=float(
                            interpolation_cfg.get(
                                "smooth_sigma", 2.0)),
                        max_search_distance=float(
                            interpolation_cfg.get(
                                "max_search_distance",
                                64.0)),
                        smoothing_iterations=int(
                            interpolation_cfg.get(
                                "smoothing_iterations",
                                0)),
                    )
                    if _dbg:
                        print(f"    pos {position_count}: "
                              f"features start", flush=True)
                    feature_stack = \
                        compute_terrain_feature_stack(
                            coarse_dem,
                            resolution=float(
                                config["resolution"]),
                            openness_scales=openness_scales,
                            mask=mask,
                        )
                    if _dbg:
                        print(f"    pos {position_count}: "
                              f"confidence start",
                              flush=True)
                    confidence_map = build_confidence_map(
                        mask=mask,
                        ground_penetration_ratio=(
                            np.nan_to_num(
                                aux_crops[
                                    "ground_penetration_ratio"
                                ],
                                nan=0.0)),
                        partial_dem=partial_dem,
                        dsm=dsm_crop,
                        coarse_dem=coarse_dem,
                        distance_scale=float(
                            config.get(
                                "confidence_distance_scale",
                                24.0)),
                    )
                    if _dbg:
                        _elapsed = _time.time() - _t0
                        print(f"    pos {position_count}: "
                              f"done {_elapsed:.2f}s",
                              flush=True)

                    sample_id = f"{tile_stem}_r{top:05d}_c{left:05d}"
                    sample_path = split_sample_dir / f"{sample_id}.npz"
                    if sample_path.exists() and not args.overwrite:
                        tile_samples += 1
                        sample_counter += 1
                        continue

                    meta = {
                        "sample_id": sample_id,
                        "tile_id": tile_stem,
                        "split": split,
                        "row": int(top),
                        "col": int(left),
                        "resolution": float(config["resolution"]),
                        "mask_strategy": strategy,
                        "mask_ratio": mask_ratio,
                        "source_dem": str(dem_path),
                        "source_dem_variant": dem_variant,
                        "source_dsm": str(dsm_path),
                        "transform": tuple(dem_profile["transform"]),
                    }
                    save_npz(
                        sample_path,
                        gt_dem=gt_crop.astype(np.float32),
                        partial_dem=partial_dem.astype(np.float32),
                        coarse_dem=coarse_dem.astype(np.float32),
                        dsm=dsm_crop.astype(np.float32),
                        mask=mask.astype(np.float32),
                        confidence_map=confidence_map.astype(np.float32),
                        meta=np.array(json.dumps(meta)),
                        **{name: value.astype(np.float32) for name, value in aux_crops.items()},
                        **{name: value.astype(np.float32) for name, value in feature_stack.items()},
                    )
                    sample_counter += 1
                    tile_samples += 1

                    if vis_count < visualization_budget:
                        save_raster_grid(
                            {
                                "gt_dem": gt_crop,
                                "partial_dem": np.where(np.isfinite(partial_dem), partial_dem, coarse_dem),
                                "coarse_dem": coarse_dem,
                                "dsm": dsm_crop,
                                "mask": mask.astype(np.float32),
                                "slope": feature_stack["slope"],
                                "curvature": feature_stack["curvature"],
                                "confidence_map": confidence_map,
                            },
                            figure_root / split / f"{sample_id}.png",
                        )
                        vis_count += 1

            print(f"    done: {tile_samples} samples",
                  flush=True)

        print(f"  wrote {sample_counter} samples for split={split}")


if __name__ == "__main__":
    main()
