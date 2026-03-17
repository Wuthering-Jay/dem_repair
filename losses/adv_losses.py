"""Adversarial loss placeholders."""

from __future__ import annotations

import torch


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """A minimal non-saturating style generator loss."""
    return torch.mean((fake_logits - 1.0) ** 2)


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """A minimal least-squares discriminator loss."""
    real_loss = torch.mean((real_logits - 1.0) ** 2)
    fake_loss = torch.mean(fake_logits**2)
    return 0.5 * (real_loss + fake_loss)
