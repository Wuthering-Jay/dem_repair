"""Discriminator placeholder for conditional GAN training."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import DoubleConv as ConvBlock


class DEMDiscriminator(nn.Module):
    """Minimal patch-style discriminator placeholder."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        self.features = ConvBlock(in_channels, base_channels)
        self.classifier = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(x))
        return logits
