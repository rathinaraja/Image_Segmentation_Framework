"""
modules/nnunet/nnunet_parts.py
-------------------------------
Building blocks for nnU-Net (2D, Level-1 architecture only):

Key differences from standard UNet:
  - Instance Normalization  (works on single samples, better for small batches)
  - Leaky ReLU              (prevents dead neurons)
  - Residual connections    (within each conv block)
  - Strided Conv            (learnable downsampling, no MaxPool)
  - Transposed Conv         (learnable upsampling)
"""

import torch
import torch.nn as nn


# ── Core residual block ────────────────────────────────────────────────────────

class ResidualConvBlock(nn.Module):
    """
    Two conv layers with InstanceNorm + LeakyReLU,
    plus a residual (skip) connection.

    If in_channels != out_channels, a 1x1 conv aligns the residual.

        x ──► Conv─IN─LReLU─Conv─IN ──► + ──► LReLU
        │                               ▲
        └─────── (1x1 conv if needed) ──┘
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, negative_slope: float = 0.01):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        )

        # Align residual dimensions if needed
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        ) if (in_channels != out_channels or stride != 1) else nn.Identity()

        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv_block(x) + self.residual(x))


# ── Encoder block ──────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    ResidualConvBlock with strided conv downsampling (stride=2).
    Returns feature map for skip connection + downsampled output.

        x ──► ResidualConvBlock(stride=1) ──► skip
                                         └──► ResidualConvBlock(stride=2) ──► out
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv    = ResidualConvBlock(in_channels,  out_channels, stride=1)
        self.downsample = ResidualConvBlock(out_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        out  = self.downsample(skip)
        return out, skip       # out → next encoder, skip → decoder


# ── Bottleneck ─────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """Deepest block — no skip, no downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = ResidualConvBlock(in_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Decoder block ──────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Transposed conv upsampling → concat skip → ResidualConvBlock.

        x ──► TransposeConv (×2) ──► cat(skip) ──► ResidualConvBlock ──► out
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                       kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(in_channels // 2 + skip_channels,
                                      out_channels, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims mismatch (odd input sizes)
        if x.shape != skip.shape:
            x = nn.functional.pad(x, [0, skip.shape[3] - x.shape[3],
                                       0, skip.shape[2] - x.shape[2]])
        return self.conv(torch.cat([skip, x], dim=1))
