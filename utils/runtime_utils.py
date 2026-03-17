"""Runtime helpers shared by training, validation, inference, and checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from datasets import DEMRepairDataset
from models import DEMGenerator
from utils.config_utils import resolve_path
from utils.vis_utils import save_raster_grid


def build_dataset(
    dataset_config: dict[str, Any],
    split: str | None,
    transform: Any = None,
    sample_dir: str | Path | None = None,
) -> DEMRepairDataset:
    """Build a dataset from repository config."""
    sample_root = resolve_path(
        sample_dir if sample_dir is not None
        else dataset_config["processed_sample_dir"],
    )
    norm_scales = {
        str(k): float(v)
        for k, v in dataset_config.get(
            "normalization_scales", {},
        ).items()
    }
    return DEMRepairDataset(
        sample_dir=sample_root,
        required_channels=list(
            dataset_config["required_channels"]),
        split=split,
        transform=transform,
        normalization_scales=norm_scales,
    )


def build_generator(
    model_config: dict[str, Any],
    in_channels: int,
) -> DEMGenerator:
    """Instantiate the supervised MVP generator."""
    return DEMGenerator(
        in_channels=in_channels,
        out_channels=int(
            model_config.get("out_channels", 1)),
        base_channels=int(
            model_config.get("base_channels", 32)),
        dropout=float(model_config.get("dropout", 0.0)),
        depth=int(model_config.get("depth", 3)),
    )


def denormalize_tensor(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo per-sample DEM normalization."""
    while mean.ndim < tensor.ndim:
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    return (tensor * std) + mean


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint if it exists."""
    path = Path(checkpoint_path)
    if not path.exists():
        return {}
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def save_prediction_preview(
    output_path: str | Path,
    partial_dem: torch.Tensor,
    coarse_dem: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dem_mean: torch.Tensor,
    dem_std: torch.Tensor,
) -> Path:
    """Save a small raster panel for debugging or reporting."""
    partial = denormalize_tensor(partial_dem[:1], dem_mean[:1], dem_std[:1])[0, 0].detach().cpu().numpy()
    coarse = denormalize_tensor(coarse_dem[:1], dem_mean[:1], dem_std[:1])[0, 0].detach().cpu().numpy()
    pred = denormalize_tensor(prediction[:1], dem_mean[:1], dem_std[:1])[0, 0].detach().cpu().numpy()
    gt = denormalize_tensor(target[:1], dem_mean[:1], dem_std[:1])[0, 0].detach().cpu().numpy()
    mask_np = mask[0, 0].detach().cpu().numpy()
    return save_raster_grid(
        {
            "partial_dem": partial,
            "coarse_dem": coarse,
            "prediction": pred,
            "gt_dem": gt,
            "mask": mask_np,
        },
        output_path,
    )
