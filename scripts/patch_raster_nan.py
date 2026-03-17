"""Patch existing raster TIFs by filling NaN with nearest-neighbor.

This is a lightweight fix for rasters generated before the DSM/intensity
NaN-fill logic was added to build_rasters.py.  It reads each file,
applies nearest_fill, and writes back in-place — no need to re-run the
full TIN pipeline.

Usage:
    python scripts/patch_raster_nan.py --config configs/data/dataset.yaml
    python scripts/patch_raster_nan.py --config configs/data/dataset.yaml --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_yaml, resolve_path
from utils.raster_utils import nearest_fill

# Rasters to patch and their directories (relative to config keys).
PATCH_TARGETS = {
    "dsm": "raw_dsm_dir",
    "mean_intensity": "processed_raster_dir",
    "mean_return_num": "processed_raster_dir",
}


def _patch_file(path: Path) -> int:
    """Fill NaN in a single raster file.  Returns number of cells filled."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        profile = src.profile.copy()

    nan_count = int(np.isnan(arr).sum())
    if nan_count == 0:
        return 0

    filled = nearest_fill(arr)

    profile.update(dtype="float32", count=1, nodata=-9999.0)
    with rasterio.open(path, "w", **profile) as dst:
        out = filled.copy()
        out[np.isnan(out)] = -9999.0
        dst.write(out, 1)

    return nan_count


def _find_tiles(dem_dir: Path) -> list[str]:
    """Discover tile stems from *_dem.tif files."""
    stems: list[str] = []
    for p in sorted(dem_dir.glob("*_dem.tif")):
        stem = p.stem.removesuffix("_dem")
        if stem.endswith("_dem_reference") or stem.endswith("_dem_raw"):
            continue
        stems.append(stem)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch NaN in existing raster TIFs.")
    parser.add_argument(
        "--config", default="configs/data/dataset.yaml")
    parser.add_argument(
        "--split", default=None,
        help="Patch one split; default = all splits.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    dem_root = resolve_path(config["raw_dem_dir"])

    if args.split:
        splits = [args.split]
    else:
        splits = [
            config["train_split"],
            config["val_split"],
            config["test_split"],
        ]

    total_patched = 0
    total_cells = 0

    for split in splits:
        dem_dir = dem_root / split
        if not dem_dir.exists():
            print(f"[SKIP] {split}: dem directory not found")
            continue

        tiles = _find_tiles(dem_dir)
        print(f"\nSplit: {split}  ({len(tiles)} tiles)")

        for tile in tiles:
            for raster_name, dir_key in PATCH_TARGETS.items():
                root = resolve_path(config[dir_key])
                path = root / split / f"{tile}_{raster_name}.tif"
                if not path.exists():
                    continue
                filled = _patch_file(path)
                if filled > 0:
                    total_patched += 1
                    total_cells += filled
                    print(f"  {tile}_{raster_name}: "
                          f"filled {filled:,} NaN cells")

    print(f"\nDone: patched {total_patched} files, "
          f"filled {total_cells:,} NaN cells total.")


if __name__ == "__main__":
    main()

