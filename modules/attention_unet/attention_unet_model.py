"""
modules/attention_unet/attention_unet_model.py
-----------------------------------------------
Attention U-Net: standard UNet with attention gates on all skip connections.
Reference: Oktay et al. 2018 (https://arxiv.org/abs/1804.03999)

Architecture:
    Encoder path  →  identical to UNet
    Skip connections  →  each filtered by AttentionGate
    Decoder path  →  upsampling + attended skip + DoubleConv

    x1 = inc(input)        [B, 64,   H,    W   ]
    x2 = down1(x1)         [B, 128,  H/2,  W/2 ]
    x3 = down2(x2)         [B, 256,  H/4,  W/4 ]
    x4 = down3(x3)         [B, 512,  H/8,  W/8 ]
    x5 = down4(x4)         [B, 1024, H/16, W/16]

    up1(x5, x4→attn) → [B, 512,  H/8,  W/8 ]
    up2(  , x3→attn) → [B, 256,  H/4,  W/4 ]
    up3(  , x2→attn) → [B, 128,  H/2,  W/2 ]
    up4(  , x1→attn) → [B, 64,   H,    W   ]
    outc             → [B, n_classes, H, W  ]
"""

import torch.nn as nn
from modules.attention_unet.attention_unet_parts import (
    DoubleConv, Down, UpAttention
)


class AttentionUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder with attention gates on skip connections
        self.up1 = UpAttention(in_channels=1024, skip_channels=512,  out_channels=512)
        self.up2 = UpAttention(in_channels=512,  skip_channels=256,  out_channels=256)
        self.up3 = UpAttention(in_channels=256,  skip_channels=128,  out_channels=128)
        self.up4 = UpAttention(in_channels=128,  skip_channels=64,   out_channels=64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)
