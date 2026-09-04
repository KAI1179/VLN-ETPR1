"""UNet for refining a cognitive map with accumulated visual evidence."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.block = _ConvBlock(in_channels + skip_channels, skip_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.block(torch.cat((inputs, skip), dim=1))


class CognitiveMapRefiner(nn.Module):
    """Four-level UNet mapping P0 and visual evidence to 37 map channels."""

    def __init__(
        self,
        in_channels: int = 67,
        out_channels: int = 37,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        widths = [base_width * (2**level) for level in range(4)]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder0 = _ConvBlock(in_channels, widths[0])
        self.encoder1 = _ConvBlock(widths[0], widths[1])
        self.encoder2 = _ConvBlock(widths[1], widths[2])
        self.encoder3 = _ConvBlock(widths[2], widths[3])
        self.decoder2 = _UpBlock(widths[3], widths[2])
        self.decoder1 = _UpBlock(widths[2], widths[1])
        self.decoder0 = _UpBlock(widths[1], widths[0])
        self.output = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level0 = self.encoder0(inputs)
        level1 = self.encoder1(self.pool(level0))
        level2 = self.encoder2(self.pool(level1))
        level3 = self.encoder3(self.pool(level2))
        decoded = self.decoder2(level3, level2)
        decoded = self.decoder1(decoded, level1)
        decoded = self.decoder0(decoded, level0)
        return torch.sigmoid(self.output(decoded))
