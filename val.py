"""Validate the supervised DEM repair MVP on a prepared sample split."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from train import evaluate
from utils.config_utils import load_yaml, resolve_path
from utils.io_utils import ensure_dir, save_json
from utils.runtime_utils import build_dataset, build_generator, load_checkpoint, save_prediction_preview
from utils.seed_utils import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DEM repair models.")
    parser.add_argument("--config", default="configs/train/train_supervised.yaml", help="Path to the training config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.config)
    dataset_config = load_yaml(train_config["dataset_config"])
    model_config = load_yaml(train_config["generator_config"])
    seed_everything(int(train_config.get("seed", 42)))
    device = get_device(str(train_config.get("device", "cuda")))

    dataset = build_dataset(dataset_config, split=dataset_config["val_split"])
    if len(dataset) == 0:
        print("No validation samples found. Run scripts/build_samples.py first.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(train_config.get("num_workers", 0)),
    )
    model = build_generator(model_config, in_channels=len(dataset_config["required_channels"])).to(device)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else resolve_path("outputs/checkpoints/best.pt")
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    if not checkpoint:
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    metrics, preview = evaluate(model, dataloader, device)
    output_root = resolve_path(train_config.get("output_dir", "outputs"))
    ensure_dir(output_root / "logs")
    save_json(output_root / "logs" / "val_metrics.json", metrics)
    if preview is not None:
        save_prediction_preview(
            output_root / "figures" / "val_preview.png",
            partial_dem=preview["batch"]["partial_dem"],
            coarse_dem=preview["batch"]["coarse_dem"],
            prediction=preview["outputs"]["pred_final"],
            target=preview["batch"]["gt_dem"],
            mask=preview["batch"]["mask"],
            dem_mean=preview["batch"]["dem_mean"],
            dem_std=preview["batch"]["dem_std"],
        )

    print(f"[val] checkpoint={checkpoint_path}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
