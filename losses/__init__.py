"""Loss package for DEM repair."""

from .adv_losses import discriminator_loss, generator_adversarial_loss
from .dem_losses import curvature_loss, grad_loss, hole_l1_loss, masked_l1_loss, valid_l1_loss
from .structure_losses import structure_consistency_loss

__all__ = [
    "masked_l1_loss",
    "hole_l1_loss",
    "valid_l1_loss",
    "grad_loss",
    "curvature_loss",
    "generator_adversarial_loss",
    "discriminator_loss",
    "structure_consistency_loss",
]
