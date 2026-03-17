"""Run inference on prepared DEM repair samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config_utils import load_yaml, resolve_path
from utils.io_utils import ensure_dir, save_npz
from utils.runtime_utils import build_dataset, build_generator, denormalize_tensor, load_checkpoint, save_prediction_preview
from utils.seed_utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEM repair inference.")
    parser.add_argument("--config", default="configs/infer/infer.yaml", help="Path to the inference config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_config = load_yaml(args.config)
    dataset_config = load_yaml(infer_config["dataset_config"])
    model_config = load_yaml(infer_config["generator_config"])
    device = get_device(str(infer_config.get("device", "cuda")))

    input_dir = resolve_path(infer_config["input_dir"])
    dataset = build_dataset(dataset_config, split=None, sample_dir=input_dir)
    if len(dataset) == 0:
        print(f"No samples found in {input_dir}")
        return

    dataloader = DataLoader(dataset, batch_size=int(infer_config.get("batch_size", 1)), shuffle=False)
    model = build_generator(model_config, in_channels=len(dataset_config["required_channels"])).to(device)
    checkpoint_path = resolve_path(infer_config["checkpoint_path"])
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    if not checkpoint:
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    output_dir = ensure_dir(resolve_path(infer_config.get("output_dir", "outputs/predictions")))
    figure_dir = ensure_dir(output_dir / "figures")
    model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            input_tensor = batch["input_tensor"].to(device)
            coarse_dem = batch["coarse_dem"].to(device)
            mask = batch["mask"].to(device)
            outputs = model(input_tensor, coarse_dem, mask)
            prediction = denormalize_tensor(outputs["pred_final"], batch["dem_mean"].to(device), batch["dem_std"].to(device))
            prediction_np = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
            mask_np = batch["mask"][0, 0].detach().cpu().numpy().astype(np.float32)
            sample_id = batch["meta"]["sample_id"][0] if isinstance(batch["meta"]["sample_id"], list) else batch["meta"]["sample_id"]

            gt_np = denormalize_tensor(batch["gt_dem"].to(device), batch["dem_mean"].to(device), batch["dem_std"].to(device))[0, 0].detach().cpu().numpy().astype(np.float32)
            save_npz(
                output_dir / f"{sample_id}_prediction.npz",
                pred_dem=prediction_np,
                gt_dem=gt_np,
                mask=mask_np,
                meta=np.array(json.dumps({"sample_id": sample_id, "checkpoint": str(checkpoint_path)})),
            )
            if bool(infer_config.get("save_visualizations", True)):
                save_prediction_preview(
                    figure_dir / f"{sample_id}.png",
                    partial_dem=batch["partial_dem"].to(device),
                    coarse_dem=batch["coarse_dem"].to(device),
                    prediction=outputs["pred_final"],
                    target=batch["gt_dem"].to(device),
                    mask=batch["mask"].to(device),
                    dem_mean=batch["dem_mean"].to(device),
                    dem_std=batch["dem_std"].to(device),
                )
            print(f"[infer] saved {sample_id} ({batch_index + 1}/{len(dataloader)})")


if __name__ == "__main__":
    main()
