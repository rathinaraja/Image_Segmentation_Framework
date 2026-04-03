"""
modules/segnet/segnet_parts.py
-------------------------------
Building blocks for SegNet:
  - VGGBlock      : Conv → BN → ReLU (repeated)
  - EncoderBlock  : VGGBlock + MaxPool (returns indices for unpooling)
  - DecoderBlock  : MaxUnpool + VGGBlock
"""

import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    """n × (Conv2d → BN → ReLU)."""

    def __init__(self, in_channels: int, out_channels: int, n_convs: int = 2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers += [
                nn.Conv2d(in_channels if i == 0 else out_channels,
                          out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """VGGBlock → MaxPool2d (returns pooled output + indices)."""

    def __init__(self, in_channels: int, out_channels: int, n_convs: int = 2):
        super().__init__()
        self.vgg  = VGGBlock(in_channels, out_channels, n_convs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x: torch.Tensor):
        x       = self.vgg(x)
        pooled, indices = self.pool(x)
        return pooled, indices, x.size()   # size needed for unpool


class DecoderBlock(nn.Module):
    """MaxUnpool2d → VGGBlock."""

    def __init__(self, in_channels: int, out_channels: int, n_convs: int = 2):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.vgg    = VGGBlock(in_channels, out_channels, n_convs)

    def forward(self, x: torch.Tensor, indices: torch.Tensor,
                output_size: torch.Size) -> torch.Tensor:
        x = self.unpool(x, indices, output_size=output_size)
        return self.vgg(x)
