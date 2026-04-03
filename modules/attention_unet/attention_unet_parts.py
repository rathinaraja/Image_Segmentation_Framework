"""
modules/attention_unet/attention_unet_parts.py
-----------------------------------------------
Building blocks for Attention U-Net.
Reference: Oktay et al. 2018 (https://arxiv.org/abs/1804.03999)

Key addition over standard UNet:
  AttentionGate — placed on every skip connection.
  Learns to suppress irrelevant spatial regions and
  amplify relevant ones before they enter the decoder.

  g (gating)  ──► Wg (1x1 conv) ──┐
                                    ├──► + ──► ReLU ──► psi (1x1 conv) ──► Sigmoid ──► α
  x (skip)    ──► Wx (1x1 conv) ──┘                                                    │
                                                                                         ▼
  x ──────────────────────────────────────────────────────────────────────────── α ⊗ x ──► out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    """MaxPool2d → DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.block(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate for skip connections.

    Args:
        F_g  : channels in gating signal g  (from decoder, coarser)
        F_l  : channels in skip feature x   (from encoder, finer)
        F_int: intermediate channels (typically F_l // 2)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Project gating signal to F_int
        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Project skip feature to F_int (stride=1, same spatial size as x)
        self.Wx = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Collapse to single-channel attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g : gating signal  [B, F_g, H/2, W/2]
            x : skip feature   [B, F_l, H,   W  ]
        Returns:
            attended skip      [B, F_l, H,   W  ]
        """
        # Upsample g to match x's spatial size
        g_up = F.interpolate(self.Wg(g), size=x.shape[2:], mode='bilinear', align_corners=True)
        alpha = self.psi(self.relu(g_up + self.Wx(x)))   # [B, 1, H, W]
        return x * alpha                                   # broadcast multiply


class UpAttention(nn.Module):
    """Upsample → concat attended skip → DoubleConv"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up      = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                          kernel_size=2, stride=2)
        self.attn    = AttentionGate(F_g=in_channels // 2,
                                     F_l=skip_channels,
                                     F_int=skip_channels // 2)
        self.conv    = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        # Pad if spatial mismatch
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        skip = self.attn(g=x, x=skip)           # apply attention to skip
        return self.conv(torch.cat([skip, x], dim=1))
