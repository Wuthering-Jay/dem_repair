"""Structure metric placeholders for DEM repair."""

from __future__ import annotations

import torch


def mask_coverage(mask: torch.Tensor) -> torch.Tensor:
    """Return the valid ratio of a binary mask."""
    return torch.mean(mask.to(dtype=torch.float32))
