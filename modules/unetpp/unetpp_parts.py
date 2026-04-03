"""
modules/unetpp/unetpp_parts.py
-------------------------------
Building blocks for UNet++ (nested dense skip connections).
Reference: Zhou et al. 2018 (https://arxiv.org/abs/1807.10165)

Dense node x^{i,j}:
  - Takes ALL previous nodes at level i: x^{i,0}, x^{i,1}, ..., x^{i,j-1}
  - Plus upsampled output from below:    Up(x^{i+1, j-1})
  - Concatenates all → DoubleConv → output (filters[i] channels)

  x^{i,0} ─────────────────────────────────────────┐
  x^{i,1} ─────────────────────────────────────┐   │
  x^{i,2} ─────────────────────────────────┐   │   │
  Up(x^{i+1,j-1}) ─────────────────────┐   │   │   │
                                         ▼   ▼   ▼   ▼
                                        cat(all) → DoubleConv → x^{i,j}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class VGGBlock(nn.Module):
    """
    Dense node conv: takes concatenated inputs, outputs out_channels.
    in_channels is computed dynamically based on how many nodes are being concatenated.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x): return self.conv(x)


def upsample_and_pad(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bilinear upsample x to match target's spatial size."""
    x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
    return x
