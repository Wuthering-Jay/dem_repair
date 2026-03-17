"""DEM-focused evaluation metrics."""

from __future__ import annotations

import torch


def mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean absolute error with optional masking."""
    absolute_error = torch.abs(prediction - target)
    if mask is None:
        return absolute_error.mean()
    weight = mask.to(dtype=prediction.dtype)
    return torch.sum(absolute_error * weight) / torch.clamp(weight.sum(), min=1.0)


def rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Root mean squared error with optional masking."""
    squared_error = (prediction - target) ** 2
    if mask is None:
        return torch.sqrt(squared_error.mean())
    weight = mask.to(dtype=prediction.dtype)
    return torch.sqrt(torch.sum(squared_error * weight) / torch.clamp(weight.sum(), min=1.0))


def slope_rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Slope RMSE computed from first-order finite differences."""
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]

    pred_slope = torch.sqrt(pred_dx[..., :-1, :] ** 2 + pred_dy[..., :, :-1] ** 2)
    target_slope = torch.sqrt(target_dx[..., :-1, :] ** 2 + target_dy[..., :, :-1] ** 2)

    if mask is None:
        return rmse(pred_slope, target_slope)

    slope_mask = mask[..., 1:, 1:]
    return rmse(pred_slope, target_slope, slope_mask)


def compute_dem_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Compute the baseline metric set used by validation."""
    return {
        "mae": float(mae(prediction, target).detach().cpu()),
        "rmse": float(rmse(prediction, target).detach().cpu()),
        "mask_mae": float(mae(prediction, target, mask).detach().cpu()),
        "mask_rmse": float(rmse(prediction, target, mask).detach().cpu()),
        "slope_rmse": float(slope_rmse(prediction, target, mask).detach().cpu()),
    }
