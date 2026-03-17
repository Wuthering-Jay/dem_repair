"""Two-stage supervised generator for DEM repair."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import DoubleConv, DownBlock, OutputHead, UpBlock


class UNetResidualStage(nn.Module):
    """Variable-depth U-Net that predicts a residual DEM correction.

    Parameters
    ----------
    depth : int
        Number of encoder *down-sampling* levels (default 3).
        depth=3 → enc1, enc2(2×), enc3(4×), bottleneck(4×)
        depth=4 → enc1, enc2(2×), enc3(4×), enc4(8×), bottleneck(8×)
        Larger depth = larger receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        out_channels: int = 1,
        dropout: float = 0.0,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.depth = depth

        # Encoder: enc1 (no downsample) + enc2..enc_{depth}
        ch = base_channels
        self.enc1 = DoubleConv(
            in_channels, ch, dropout=dropout)
        self.encoders = nn.ModuleList()
        for _ in range(depth - 1):
            next_ch = min(ch * 2, base_channels * 8)
            self.encoders.append(
                DownBlock(ch, next_ch, dropout=dropout))
            ch = next_ch

        self.bottleneck = DoubleConv(
            ch, ch, dropout=dropout)

        # Decoder (mirrors encoder)
        self.decoders = nn.ModuleList()
        for i in range(depth - 1):
            # Walk encoder channels in reverse
            enc_idx = depth - 2 - i
            if enc_idx == 0:
                skip_ch = base_channels
            else:
                skip_ch = min(
                    base_channels * (2 ** enc_idx),
                    base_channels * 8,
                )
            out_ch = skip_ch
            self.decoders.append(
                UpBlock(ch, skip_ch, out_ch,
                        dropout=dropout))
            ch = out_ch

        self.head = OutputHead(
            ch, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = self.enc1(x)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        # Remove last (it feeds into bottleneck, not skip)
        skips.pop()
        x = self.bottleneck(x)
        for dec in self.decoders:
            x = dec(x, skips.pop())
        return self.head(x)


class DEMGenerator(nn.Module):
    """Two-stage residual generator on coarse DEM priors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.0,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.stage1 = UNetResidualStage(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=out_channels,
            dropout=dropout,
            depth=depth,
        )
        self.stage2 = UNetResidualStage(
            in_channels=in_channels + out_channels + 1,
            base_channels=base_channels,
            out_channels=out_channels,
            dropout=dropout,
            depth=depth,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        coarse_dem: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        residual1 = self.stage1(inputs)
        pred_stage1 = coarse_dem + (mask * residual1)
        refinement_input = torch.cat(
            [inputs, pred_stage1, mask], dim=1)
        residual2 = self.stage2(refinement_input)
        pred_final = pred_stage1 + (mask * residual2)
        return {
            "pred_stage1": pred_stage1,
            "pred_final": pred_final,
            "residual1": residual1,
            "residual2": residual2,
        }
