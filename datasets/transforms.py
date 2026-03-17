"""Simple tensor transforms for DEM repair datasets."""

from __future__ import annotations

from typing import Any, Iterable

import torch


class IdentityTransform:
    """Pass-through transform."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return sample


class ComposeTransforms:
    """Apply transforms in sequence."""

    def __init__(self, transforms: Iterable[Any]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomFlip:
    """Randomly flip sample tensors across spatial axes.

    Horizontal and vertical flips are sampled independently,
    giving 4 possible outcomes (none, H only, V only, both).
    """

    FLIP_KEYS = (
        "input_tensor", "gt_dem", "mask",
        "coarse_dem", "partial_dem",
    )

    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = True,
        p: float = 0.5,
    ) -> None:
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = float(p)

    def __call__(
        self, sample: dict[str, Any],
    ) -> dict[str, Any]:
        dims: list[int] = []
        if self.horizontal and torch.rand(1).item() < self.p:
            dims.append(-1)
        if self.vertical and torch.rand(1).item() < self.p:
            dims.append(-2)
        if not dims:
            return sample
        for key in self.FLIP_KEYS:
            if key in sample:
                sample[key] = torch.flip(
                    sample[key], dims=dims)
        return sample
