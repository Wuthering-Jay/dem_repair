"""Supervised DEM losses for the first-stage MVP."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute a masked L1 loss."""
    if mask is None:
        return torch.mean(torch.abs(prediction - target))
    weight = mask.to(dtype=prediction.dtype)
    numerator = torch.sum(torch.abs(prediction - target) * weight)
    denominator = torch.clamp(weight.sum(), min=1.0)
    return numerator / denominator


def hole_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss focused on simulated void pixels."""
    return masked_l1_loss(prediction, target, mask)


def valid_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss on observed terrain pixels."""
    return masked_l1_loss(prediction, target, 1.0 - mask)


def _gradient_x(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., :, 1:] - tensor[..., :, :-1]


def _gradient_y(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., 1:, :] - tensor[..., :-1, :]


def grad_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Penalize gradient mismatch between predicted and target DEM."""
    pred_dx = _gradient_x(prediction)
    target_dx = _gradient_x(target)
    pred_dy = _gradient_y(prediction)
    target_dy = _gradient_y(target)

    if mask is None:
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    weight_x = torch.maximum(mask[..., :, 1:], mask[..., :, :-1]).to(dtype=prediction.dtype)
    weight_y = torch.maximum(mask[..., 1:, :], mask[..., :-1, :]).to(dtype=prediction.dtype)
    loss_x = masked_l1_loss(pred_dx, target_dx, weight_x)
    loss_y = masked_l1_loss(pred_dy, target_dy, weight_y)
    return loss_x + loss_y


def curvature_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Simple curvature loss using a Laplacian stencil."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=prediction.device,
        dtype=prediction.dtype,
    ).view(1, 1, 3, 3)
    pred_curv = F.conv2d(prediction, kernel, padding=1)
    target_curv = F.conv2d(target, kernel, padding=1)
    return masked_l1_loss(pred_curv, target_curv, mask)
