"""
modules/segnet/segnet_model.py
-------------------------------
SegNet: encoder-decoder architecture using MaxPool indices for upsampling.
Reference: Badrinarayanan et al., 2017 (https://arxiv.org/abs/1511.00561)

Encoder mirrors VGG-16 (without FC layers).
Decoder mirrors encoder in reverse, using stored pool indices.
"""

import torch
import torch.nn as nn
from modules.segnet.segnet_parts import EncoderBlock, DecoderBlock


class SegNet(nn.Module):
    """
    Args:
        n_channels : Number of input channels (e.g. 3 for RGB).
        n_classes  : Number of output segmentation classes.
    """

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes

        # ── Encoder (mirrors VGG-16 block structure) ──────────────────────
        self.enc1 = EncoderBlock(n_channels, 64,  n_convs=2)
        self.enc2 = EncoderBlock(64,         128, n_convs=2)
        self.enc3 = EncoderBlock(128,        256, n_convs=3)
        self.enc4 = EncoderBlock(256,        512, n_convs=3)
        self.enc5 = EncoderBlock(512,        512, n_convs=3)

        # ── Decoder (reverse) ─────────────────────────────────────────────
        self.dec5 = DecoderBlock(512, 512, n_convs=3)
        self.dec4 = DecoderBlock(512, 256, n_convs=3)
        self.dec3 = DecoderBlock(256, 128, n_convs=3)
        self.dec2 = DecoderBlock(128, 64,  n_convs=2)
        self.dec1 = DecoderBlock(64,  64,  n_convs=2)

        # ── Final classifier ──────────────────────────────────────────────
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        x, idx1, s1 = self.enc1(x)
        x, idx2, s2 = self.enc2(x)
        x, idx3, s3 = self.enc3(x)
        x, idx4, s4 = self.enc4(x)
        x, idx5, s5 = self.enc5(x)

        # Decode
        x = self.dec5(x, idx5, s5)
        x = self.dec4(x, idx4, s4)
        x = self.dec3(x, idx3, s3)
        x = self.dec2(x, idx2, s2)
        x = self.dec1(x, idx1, s1)

        return self.classifier(x)
