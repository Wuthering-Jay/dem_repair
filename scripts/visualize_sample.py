"""Visualize a saved NPZ sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_yaml, resolve_path
from utils.vis_utils import save_raster_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a DEM repair sample.")
    parser.add_argument("--config", default="configs/data/dataset.yaml", help="Dataset config path.")
    parser.add_argument("--sample", default=None, help="Path to a sample NPZ file.")
    parser.add_argument("--output", default="outputs/figures/visualize_sample.png", help="Visualization output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.sample:
        sample_path = resolve_path(args.sample)
    else:
        sample_dir = resolve_path(config["processed_sample_dir"]) / config["train_split"]
        samples = sorted(sample_dir.glob("*.npz"))
        if not samples:
            print("No samples found. Run scripts/build_samples.py first.")
            return
        sample_path = samples[0]

    with np.load(sample_path, allow_pickle=True) as sample:
        preview = {
            "gt_dem": sample["gt_dem"],
            "partial_dem": np.where(np.isfinite(sample["partial_dem"]), sample["partial_dem"], sample["coarse_dem"]),
            "coarse_dem": sample["coarse_dem"],
            "dsm": sample["dsm"],
            "mask": sample["mask"],
            "confidence_map": sample["confidence_map"],
            "slope": sample["slope"],
            "curvature": sample["curvature"],
            "openness_s1": sample["openness_s1"],
            "openness_s2": sample["openness_s2"],
        }
        meta = json.loads(str(sample["meta"].item()))
    save_raster_grid(preview, resolve_path(args.output))
    print(f"visualized sample {meta['sample_id']} -> {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
