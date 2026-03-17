"""Verify model construction and a single forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_yaml, resolve_path
from utils.runtime_utils import build_dataset, build_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DEM repair model forward pass.")
    parser.add_argument("--config", default="configs/data/dataset.yaml", help="Dataset config path.")
    parser.add_argument("--model-config", default="configs/model/generator.yaml", help="Generator config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_yaml(args.config)
    model_config = load_yaml(args.model_config)
    dataset = build_dataset(dataset_config, split=dataset_config["train_split"])
    model = build_generator(model_config, in_channels=len(dataset_config["required_channels"]))
    model.eval()

    if len(dataset) > 0:
        sample = dataset[0]
        inputs = sample["input_tensor"].unsqueeze(0)
        coarse_dem = sample["coarse_dem"].unsqueeze(0)
        mask = sample["mask"].unsqueeze(0)
    else:
        size = int(dataset_config["tile_size"])
        channels = len(dataset_config["required_channels"])
        inputs = torch.randn(1, channels, size, size)
        coarse_dem = torch.randn(1, 1, size, size)
        mask = torch.zeros(1, 1, size, size)
        mask[:, :, size // 4 : size // 2, size // 4 : size // 2] = 1.0

    with torch.no_grad():
        outputs = model(inputs, coarse_dem, mask)
    print(f"input shape={tuple(inputs.shape)}")
    print(f"stage1 shape={tuple(outputs['pred_stage1'].shape)}")
    print(f"final shape={tuple(outputs['pred_final'].shape)}")
    print("model forward check passed")


if __name__ == "__main__":
    main()
