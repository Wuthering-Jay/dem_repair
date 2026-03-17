"""Inference script: repair DEM voids and output raster + point cloud.

Usage:
    python scripts/predict.py \
        --config configs/train/train_supervised.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --tile 001 \
        --split train \
        --output outputs/predictions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dem_repair_dataset import (
    DENSITY_CHANNELS, DEM_LIKE_CHANNELS,
    DERIVED_CHANNELS, RATIO_CHANNELS,
)
from utils.config_utils import load_yaml, resolve_path
from utils.confidence import build_confidence_map
from utils.interpolation import interpolate_coarse_dem
from utils.raster_utils import read_raster
from utils.runtime_utils import (
    build_generator, load_checkpoint,
)
from utils.terrain_features import (
    compute_terrain_feature_stack,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict / repair DEM voids.")
    p.add_argument("--config", required=True,
                   help="Training config YAML.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint.")
    p.add_argument("--tile", required=True,
                   help="Tile stem, e.g. '001'.")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="outputs/predictions")
    p.add_argument("--device", default="cuda")
    p.add_argument("--points-per-cell", type=float,
                   default=1.0,
                   help="Average points per void cell.")
    p.add_argument("--jitter", type=float, default=0.4,
                   help="XY jitter range (0-0.5).")
    return p.parse_args()


# ---- Normalization helpers (mirror DEMRepairDataset) ----

def _norm_dem(arr, mean, std):
    a = np.where(np.isfinite(arr), arr, mean).astype(np.float32)
    return ((a - mean) / std).astype(np.float32)


def _norm_density(arr, scale):
    a = np.log1p(np.maximum(
        np.nan_to_num(arr, nan=0.0), 0.0)).astype(np.float32)
    return a / max(scale, 1.0)


def _norm_feature(arr, scale):
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    fill = float(np.mean(a[finite])) if finite.any() else 0.0
    a = np.where(finite, a, fill)
    return (a / max(scale, 1.0)).astype(np.float32)


def _norm_ratio(arr):
    return np.clip(
        np.nan_to_num(arr, nan=0.0), 0.0, 1.0,
    ).astype(np.float32)


# ---- Tile loading ----

def _load_tile_data(
    tile_stem: str, split: str, dataset_cfg: dict,
) -> dict[str, Any]:
    """Load all rasters for a tile."""
    dem_root = resolve_path(dataset_cfg["raw_dem_dir"])
    dsm_root = resolve_path(dataset_cfg["raw_dsm_dir"])
    aux_root = resolve_path(dataset_cfg["processed_raster_dir"])

    # DEM: prefer dem_raw (has real voids)
    dem_raw_path = dem_root / split / f"{tile_stem}_dem_raw.tif"
    if not dem_raw_path.exists():
        dem_raw_path = dem_root / split / f"{tile_stem}_dem.tif"
    dem_raw, profile = read_raster(dem_raw_path)

    dsm_path = dsm_root / split / f"{tile_stem}_dsm.tif"
    dsm, _ = read_raster(dsm_path)

    aux_names = [
        "all_density", "ground_density",
        "vegetation_density", "building_density",
        "mean_intensity", "mean_return_num",
        "last_return_ratio", "ground_penetration_ratio",
    ]
    aux = {}
    for name in aux_names:
        p = aux_root / split / f"{tile_stem}_{name}.tif"
        if p.exists():
            aux[name], _ = read_raster(p)

    # Load dem_valid_mask: cells where actual ground
    # LiDAR points exist (not TIN-interpolated).
    valid_mask_path = (
        aux_root / split
        / f"{tile_stem}_dem_valid_mask.tif")
    if valid_mask_path.exists():
        valid_mask, _ = read_raster(valid_mask_path)
        valid_mask = valid_mask > 0.5
    else:
        # Fallback: use ground_density > 0
        gd = aux.get("ground_density")
        if gd is not None:
            valid_mask = gd > 0
        else:
            valid_mask = np.isfinite(dem_raw)

    return {
        "dem_raw": dem_raw,
        "dsm": dsm,
        "aux": aux,
        "valid_mask": valid_mask,
        "profile": profile,
    }


# ---- Sliding window inference ----

def _predict_tile(
    tile_data: dict[str, Any],
    model: torch.nn.Module,
    dataset_cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Run sliding-window inference on a full tile."""
    dem_raw = tile_data["dem_raw"]
    dsm = tile_data["dsm"]
    aux = tile_data["aux"]
    valid_mask = tile_data["valid_mask"]
    H, W = dem_raw.shape

    tile_size = int(dataset_cfg.get("tile_size", 256))
    stride = int(dataset_cfg.get("stride", 128))
    resolution = float(dataset_cfg["resolution"])
    openness_scales = list(
        dataset_cfg.get("openness_scales", [3, 7, 15, 31]))
    required = list(dataset_cfg["required_channels"])
    norm_scales: dict[str, float] = {
        str(k): float(v)
        for k, v in dataset_cfg.get(
            "normalization_scales", {}).items()
    }
    interp_cfg = dataset_cfg.get("interpolation", {})

    # Accumulation buffers for overlapping predictions
    pred_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    # Sliding window positions
    row_pos = list(range(0, H - tile_size + 1, stride))
    if not row_pos or row_pos[-1] + tile_size < H:
        row_pos.append(max(0, H - tile_size))
    col_pos = list(range(0, W - tile_size + 1, stride))
    if not col_pos or col_pos[-1] + tile_size < W:
        col_pos.append(max(0, W - tile_size))

    total = len(row_pos) * len(col_pos)
    count = 0

    model.eval()
    with torch.no_grad():
        for top in row_pos:
            for left in col_pos:
                count += 1
                if count % 50 == 0 or count == 1:
                    print(f"  crop {count}/{total}",
                          flush=True)

                s = np.s_[top:top+tile_size,
                          left:left+tile_size]
                dem_crop = dem_raw[s].copy()
                dsm_crop = dsm[s].copy()
                vm_crop = valid_mask[s].copy()
                aux_crops = {
                    k: v[s].copy() for k, v in aux.items()
                }

                # Mask = cells WITHOUT real ground points
                # (TIN-interpolated or IDW-filled areas).
                mask = (~vm_crop).astype(np.float32)
                if mask.sum() == 0:
                    # Fully observed — use original
                    pred_sum[s] += dem_crop
                    weight_sum[s] += 1.0
                    continue

                # Build partial DEM: keep only observed
                # cells, mask out TIN-only cells as NaN
                partial = np.where(
                    vm_crop, dem_crop, np.nan)
                coarse = interpolate_coarse_dem(
                    partial, mask=mask,
                    method=str(interp_cfg.get(
                        "method", "nearest_gaussian")),
                    smooth_sigma=float(interp_cfg.get(
                        "smooth_sigma", 2.0)),
                    max_search_distance=float(
                        interp_cfg.get(
                            "max_search_distance", 64.0)),
                    smoothing_iterations=int(
                        interp_cfg.get(
                            "smoothing_iterations", 0)),
                )
                features = compute_terrain_feature_stack(
                    coarse, resolution=resolution,
                    openness_scales=openness_scales,
                    mask=mask,
                )
                gpr = np.nan_to_num(
                    aux_crops.get(
                        "ground_penetration_ratio",
                        np.zeros_like(mask)),
                    nan=0.0)
                conf = build_confidence_map(
                    mask=mask,
                    ground_penetration_ratio=gpr,
                    partial_dem=partial,
                    dsm=dsm_crop,
                    coarse_dem=coarse,
                    distance_scale=float(
                        dataset_cfg.get(
                            "confidence_distance_scale",
                            24.0)),
                )

                # Normalization
                valid_px = dem_crop[np.isfinite(dem_crop)]
                dem_mean = float(valid_px.mean()) \
                    if valid_px.size > 0 else 0.0
                dem_std = float(valid_px.std()) \
                    if valid_px.size > 0 else 1.0
                dem_std = max(dem_std, 1e-6)

                # Build channel dict
                all_channels = {
                    "partial_dem": partial,
                    "coarse_dem": coarse,
                    "dsm": dsm_crop,
                    "mask": mask,
                    "confidence_map": conf,
                }
                all_channels.update(aux_crops)
                all_channels.update(features)

                # Assemble input_tensor
                ch_list = []
                for ch_name in required:
                    arr = all_channels.get(ch_name)
                    if arr is None:
                        arr = np.zeros(
                            (tile_size, tile_size),
                            dtype=np.float32)
                    if ch_name in DEM_LIKE_CHANNELS:
                        ch_list.append(
                            _norm_dem(arr, dem_mean,
                                      dem_std))
                    elif ch_name in DENSITY_CHANNELS:
                        sc = norm_scales.get(ch_name, 0.0)
                        ch_list.append(
                            _norm_density(arr, sc))
                    elif ch_name in RATIO_CHANNELS:
                        ch_list.append(_norm_ratio(arr))
                    elif (ch_name in DERIVED_CHANNELS
                          or ch_name in (
                              "mean_intensity",
                              "mean_return_num")):
                        sc = norm_scales.get(ch_name, 0.0)
                        ch_list.append(
                            _norm_feature(arr, sc))
                    else:
                        ch_list.append(
                            np.nan_to_num(
                                arr, nan=0.0,
                            ).astype(np.float32))

                inp = torch.from_numpy(
                    np.stack(ch_list, axis=0),
                )[None].float().to(device)
                coarse_t = torch.from_numpy(
                    _norm_dem(coarse, dem_mean, dem_std),
                )[None, None].float().to(device)
                mask_t = torch.from_numpy(
                    mask,
                )[None, None].float().to(device)

                out = model(inp, coarse_t, mask_t)
                pred_norm = out["pred_final"][0, 0].cpu(
                ).numpy()
                pred_real = pred_norm * dem_std + dem_mean

                # Blend: void=predicted, valid=original
                crop_result = np.where(
                    mask > 0.5, pred_real, dem_crop)
                pred_sum[s] += crop_result
                weight_sum[s] += 1.0

    # Average overlapping predictions
    valid_w = weight_sum > 0
    result = np.full((H, W), np.nan, dtype=np.float32)
    result[valid_w] = (
        pred_sum[valid_w] / weight_sum[valid_w]
    ).astype(np.float32)
    return result


# ---- Point cloud generation ----

def _dem_to_pointcloud(
    repaired_dem: np.ndarray,
    void_mask: np.ndarray,
    profile: dict[str, Any],
    points_per_cell: float = 1.0,
    jitter: float = 0.4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert void-region DEM pixels to XYZ points.

    Returns (x, y, z) arrays with natural-looking
    irregular spacing via XY jitter and variable density.
    """
    rng = np.random.default_rng(seed)
    transform = profile["transform"]
    res = transform.a  # pixel size in X

    void_rows, void_cols = np.where(void_mask > 0.5)
    n_cells = len(void_rows)
    if n_cells == 0:
        return (np.array([]), np.array([]),
                np.array([]))

    # Variable density: sample n_points per cell
    # from Poisson distribution around points_per_cell
    n_per_cell = rng.poisson(
        lam=points_per_cell, size=n_cells)
    n_per_cell = np.maximum(n_per_cell, 0)
    total_pts = int(n_per_cell.sum())

    xs = np.empty(total_pts, dtype=np.float64)
    ys = np.empty(total_pts, dtype=np.float64)
    zs = np.empty(total_pts, dtype=np.float32)

    idx = 0
    for i in range(n_cells):
        n = n_per_cell[i]
        if n == 0:
            continue
        r, c = int(void_rows[i]), int(void_cols[i])
        # Random sub-pixel offsets
        dx = rng.uniform(-jitter, jitter, size=n)
        dy = rng.uniform(-jitter, jitter, size=n)
        # Pixel centre → world coordinates
        cx = transform.c + (c + 0.5) * res
        cy = transform.f + (r + 0.5) * transform.e
        xs[idx:idx+n] = cx + dx * res
        ys[idx:idx+n] = cy + dy * abs(transform.e)
        # Z from repaired DEM (use cell value for all
        # sub-pixel points — the DEM is 1m resolution,
        # sub-pixel Z variation is negligible)
        zs[idx:idx+n] = repaired_dem[r, c]
        idx += n

    return xs[:idx], ys[:idx], zs[:idx]


def _write_las(
    path: Path,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    crs: Any = None,
) -> None:
    """Write XYZ points to a LAS file."""
    import laspy

    header = laspy.LasHeader(point_format=0, version="1.2")
    # Set scale and offset for precision
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [
        float(np.min(x)) if len(x) > 0 else 0.0,
        float(np.min(y)) if len(y) > 0 else 0.0,
        float(np.min(z)) if len(z) > 0 else 0.0,
    ]

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    # Mark as ground (class 2)
    las.classification = np.full(
        len(x), 2, dtype=np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


# ---- Main ----

def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(train_cfg["dataset_config"])
    model_cfg = load_yaml(train_cfg["generator_config"])
    device = torch.device(args.device)

    required = list(dataset_cfg["required_channels"])
    model = build_generator(
        model_cfg, in_channels=len(required),
    ).to(device)
    ckpt = load_checkpoint(
        model, args.checkpoint, device)
    if not ckpt:
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: epoch={epoch}",
          flush=True)

    print(f"Loading tile {args.tile}...", flush=True)
    tile_data = _load_tile_data(
        args.tile, args.split, dataset_cfg)
    dem_raw = tile_data["dem_raw"]
    valid_mask = tile_data["valid_mask"]
    # Void = cells without real ground observations
    # (TIN-interpolated or IDW-filled, not NaN).
    void_mask = ~valid_mask
    n_void = int(void_mask.sum())
    print(f"  shape={dem_raw.shape} "
          f"void={n_void:,} cells "
          f"({n_void/dem_raw.size:.1%})",
          flush=True)

    print("Running inference...", flush=True)
    repaired = _predict_tile(
        tile_data, model, dataset_cfg, device)

    # Save repaired DEM raster
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    raster_path = out_dir / f"{args.tile}_repaired.tif"
    profile = tile_data["profile"].copy()
    profile.update(dtype="float32", count=1,
                   nodata=-9999.0, compress="deflate")
    with rasterio.open(raster_path, "w", **profile) as dst:
        out_arr = repaired.copy()
        out_arr[np.isnan(out_arr)] = -9999.0
        dst.write(out_arr, 1)
    print(f"Saved raster: {raster_path}", flush=True)

    # Generate point cloud for void regions
    print("Generating point cloud...", flush=True)
    x, y, z = _dem_to_pointcloud(
        repaired, void_mask, tile_data["profile"],
        points_per_cell=args.points_per_cell,
        jitter=args.jitter,
    )
    las_path = out_dir / f"{args.tile}_fill_points.las"
    _write_las(las_path, x, y, z,
               crs=tile_data["profile"].get("crs"))
    print(f"Saved point cloud: {las_path} "
          f"({len(x):,} points)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
