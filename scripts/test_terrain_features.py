"""Smoke-test terrain feature generation on a raster or synthetic terrain."""

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
from utils.terrain_features import compute_terrain_feature_stack
from utils.vis_utils import save_raster_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test terrain feature computation.")
    parser.add_argument("--config", default="configs/data/dataset.yaml", help="Dataset config path.")
    parser.add_argument("--raster", default=None, help="Optional DEM raster path.")
    parser.add_argument("--output", default="outputs/figures/terrain_features_test.png", help="Preview output path.")
    return parser.parse_args()


def synthetic_surface(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-3.0, 3.0, size)
    y = np.linspace(-3.0, 3.0, size)
    xx, yy = np.meshgrid(x, y)
    dem = (
        120.0
        + 10.0 * np.sin(xx * 1.5)
        + 8.0 * np.cos(yy * 1.2)
        - 12.0 * np.exp(-((xx - 1.0) ** 2 + (yy + 0.5) ** 2))
        + 9.0 * np.exp(-((xx + 1.4) ** 2 + (yy - 1.1) ** 2))
    ).astype(np.float32)
    mask = np.zeros_like(dem, dtype=np.float32)
    mask[90:150, 110:170] = 1.0
    return dem, mask


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.raster:
        dem, _ = read_raster(resolve_path(args.raster))
        mask = np.zeros_like(dem, dtype=np.float32)
    else:
        dem, mask = synthetic_surface()

    features = compute_terrain_feature_stack(
        dem=dem,
        resolution=float(config["resolution"]),
        openness_scales=[int(value) for value in config.get("openness_scales", [3, 7, 15, 31])],
        mask=mask,
    )
    save_raster_grid({"dem": dem, "mask": mask, **features}, resolve_path(args.output))
    print(f"saved terrain feature preview to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
