"""Verify raster products meet training pipeline requirements.

Checks performed per tile:
  1. All required rasters exist
  2. All rasters share the same shape (height x width)
  3. DEM / DSM / dem_reference have no NaN (complete surfaces)
  4. Auxiliary maps have acceptable NaN ratios
  5. Value ranges are physically plausible
  6. Density maps are non-negative
  7. Ratio maps are in [0, 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_yaml, resolve_path
from utils.raster_utils import read_raster


# Rasters that build_samples.py requires
CORE_RASTERS = ["dem", "dem_raw", "dem_reference", "dsm"]
AUX_RASTERS = [
    "all_density", "ground_density", "vegetation_density",
    "building_density", "mean_intensity", "mean_return_num",
    "last_return_ratio", "ground_penetration_ratio",
]
DIAG_RASTERS = [
    "dem_valid_mask", "dem_support_mask", "dem_confidence",
]

# Rasters that must be 100% finite (no NaN)
MUST_BE_COMPLETE = {"dem", "dem_raw", "dem_reference", "dsm"}

# Density maps: must be >= 0
DENSITY_MAPS = {
    "all_density", "ground_density",
    "vegetation_density", "building_density",
}

# Ratio maps: must be in [0, 1]
RATIO_MAPS = {"last_return_ratio", "ground_penetration_ratio"}

NAN_WARN_THRESHOLD = 0.05   # warn if > 5% NaN
NAN_FAIL_THRESHOLD = 0.50   # fail if > 50% NaN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify raster products for training.")
    p.add_argument("--config", default="configs/data/dataset.yaml")
    p.add_argument("--split", default=None,
                   help="Check one split; default = all splits.")
    return p.parse_args()


def _find_tiles(dem_dir: Path, aux_dir: Path) -> list[str]:
    """Discover tile stems from *_dem.tif in dem_dir.

    Falls back to aux_dir if dem_dir has no matches (some setups
    may store everything in a single directory).
    """
    stems: list[str] = []
    search_dir = dem_dir if dem_dir.exists() else aux_dir
    for p in sorted(search_dir.glob("*_dem.tif")):
        stem = p.stem.removesuffix("_dem")
        if stem.endswith("_dem_reference") or stem.endswith("_dem_raw"):
            continue
        stems.append(stem)
    return stems



def _check_tile(
    dem_dir: Path,
    dsm_dir: Path,
    aux_dir: Path,
    tile: str,
) -> tuple[list[str], list[str]]:
    """Return (warnings, errors) for one tile."""
    warns: list[str] = []
    errs: list[str] = []

    # Map each raster name to its directory.
    dir_map: dict[str, Path] = {}
    for n in CORE_RASTERS:
        if n == "dsm":
            dir_map[n] = dsm_dir
        else:
            dir_map[n] = dem_dir
    for n in AUX_RASTERS + DIAG_RASTERS:
        dir_map[n] = aux_dir

    all_names = CORE_RASTERS + AUX_RASTERS + DIAG_RASTERS
    arrays: dict[str, np.ndarray] = {}
    ref_shape: tuple[int, ...] | None = None

    # 1. Existence + load
    for name in all_names:
        path = dir_map[name] / f"{tile}_{name}.tif"
        if not path.exists():
            errs.append(f"MISSING {name}")
            continue
        arr, _ = read_raster(path)
        arrays[name] = arr
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            errs.append(
                f"SHAPE MISMATCH {name}: "
                f"{arr.shape} vs {ref_shape}")

    if not arrays:
        errs.append("NO RASTERS FOUND")
        return warns, errs

    # 2. Completeness (no NaN)
    for name in MUST_BE_COMPLETE:
        if name not in arrays:
            continue
        nan_ratio = float(np.isnan(arrays[name]).mean())
        if nan_ratio > 0:
            errs.append(
                f"{name} has {nan_ratio:.2%} NaN "
                f"(must be 0%)")

    # 3. NaN ratios for auxiliary maps
    for name in AUX_RASTERS:
        if name not in arrays:
            continue
        nan_ratio = float(np.isnan(arrays[name]).mean())
        if nan_ratio > NAN_FAIL_THRESHOLD:
            errs.append(
                f"{name} NaN={nan_ratio:.1%} "
                f"> {NAN_FAIL_THRESHOLD:.0%}")
        elif nan_ratio > NAN_WARN_THRESHOLD:
            warns.append(f"{name} NaN={nan_ratio:.1%}")

    # 4. Density non-negative
    for name in DENSITY_MAPS:
        if name not in arrays:
            continue
        arr = arrays[name]
        finite = arr[np.isfinite(arr)]
        if finite.size > 0 and float(finite.min()) < 0:
            errs.append(
                f"{name} has negative values "
                f"(min={float(finite.min()):.4f})")

    # 5. Ratio in [0, 1]
    for name in RATIO_MAPS:
        if name not in arrays:
            continue
        arr = arrays[name]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        lo = float(finite.min())
        hi = float(finite.max())
        if lo < -0.01 or hi > 1.01:
            errs.append(
                f"{name} out of [0,1]: "
                f"[{lo:.4f}, {hi:.4f}]")

    # 6. DEM / DSM plausibility
    for name in ("dem", "dem_reference", "dem_raw", "dsm"):
        if name not in arrays:
            continue
        arr = arrays[name]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        lo = float(finite.min())
        hi = float(finite.max())
        span = hi - lo
        if span > 5000:
            warns.append(
                f"{name} elevation span={span:.0f}m "
                f"(unusually large)")
        if lo < -500:
            warns.append(
                f"{name} min={lo:.1f}m (below -500m)")

    # 7. DSM >= DEM check (soft)
    if "dsm" in arrays and "dem" in arrays:
        dsm_arr = arrays["dsm"]
        dem_arr = arrays["dem"]
        both = np.isfinite(dsm_arr) & np.isfinite(dem_arr)
        if both.any():
            below = float(
                (dsm_arr[both] < dem_arr[both] - 1.0).mean())
            if below > 0.1:
                warns.append(
                    f"DSM < DEM-1m in {below:.1%} of cells")

    return warns, errs


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dem_root = resolve_path(config["raw_dem_dir"])
    dsm_root = resolve_path(config["raw_dsm_dir"])
    aux_root = resolve_path(config["processed_raster_dir"])

    if args.split:
        splits = [args.split]
    else:
        splits = [
            config["train_split"],
            config["val_split"],
            config["test_split"],
        ]

    total_tiles = 0
    total_warns = 0
    total_errs = 0

    for split in splits:
        dem_dir = dem_root / split
        dsm_dir = dsm_root / split
        aux_dir = aux_root / split

        if not dem_dir.exists() and not aux_dir.exists():
            print(f"[SKIP] {split}: directory not found")
            continue

        tiles = _find_tiles(dem_dir, aux_dir)
        print(f"\n{'=' * 60}")
        print(f"Split: {split}  ({len(tiles)} tiles)")
        print(f"{'=' * 60}")

        for tile in tiles:
            total_tiles += 1
            warns, errs = _check_tile(
                dem_dir, dsm_dir, aux_dir, tile)
            total_warns += len(warns)
            total_errs += len(errs)

            if not warns and not errs:
                print(f"  OK {tile}")
                continue

            status = "FAIL" if errs else "WARN"
            print(f"  {status} {tile}")
            for e in errs:
                print(f"      ERROR: {e}")
            for w in warns:
                print(f"      WARN:  {w}")

    print(f"\n{'=' * 60}")
    print(
        f"Summary: {total_tiles} tiles, "
        f"{total_errs} errors, {total_warns} warnings")
    if total_errs == 0:
        print("All raster products are training-ready.")
    else:
        print("Fix errors before running build_samples.py.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()