"""Auxiliary head placeholder for ridge and valley structure prediction."""

from __future__ import annotations

import torch
from torch import nn


class RidgeValleyHead(nn.Module):
    """Minimal structure head placeholder."""

    def __init__(self, in_channels: int = 32, out_channels: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
