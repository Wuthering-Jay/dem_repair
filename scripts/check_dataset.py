"""Inspect DEM repair samples and dataset outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DEMRepairDataset
from utils.config_utils import load_yaml, resolve_path
from utils.vis_utils import save_raster_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DEM repair dataset samples.")
    parser.add_argument("--config", default="configs/data/dataset.yaml", help="Dataset config path.")
    parser.add_argument("--split", default=None, help="Split to inspect.")
    parser.add_argument("--output", default="outputs/figures/check_dataset.png", help="Preview output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    split = args.split or config["train_split"]
    dataset = DEMRepairDataset(resolve_path(config["processed_sample_dir"]), list(config["required_channels"]), split=split)
    print(f"[check_dataset] split={split} samples={len(dataset)}")
    if len(dataset) == 0:
        print("No samples found. Run scripts/build_samples.py first.")
        return

    sample = dataset[0]
    print(f"input_tensor shape={tuple(sample['input_tensor'].shape)}")
    print(f"gt_dem shape={tuple(sample['gt_dem'].shape)}")
    print(f"mask shape={tuple(sample['mask'].shape)}")
    print(f"sample_id={sample['meta']['sample_id']}")

    raw_path = dataset.sample_paths[0]
    import numpy as np

    with np.load(raw_path, allow_pickle=True) as raw:
        preview = {
            "gt_dem": raw["gt_dem"],
            "partial_dem": raw["partial_dem"],
            "coarse_dem": raw["coarse_dem"],
            "dsm": raw["dsm"],
            "mask": raw["mask"],
            "slope": raw["slope"],
            "openness_s1": raw["openness_s1"],
            "confidence_map": raw["confidence_map"],
        }
    save_raster_grid(preview, resolve_path(args.output))
    print(f"saved preview to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
