"""
modules/nnunet/nnunet_model.py
-------------------------------
2D nnU-Net (Level-1 architecture).

Architecture differences vs standard UNet:
  ┌─────────────────────┬──────────────────┬──────────────────────┐
  │ Component           │ Standard UNet    │ nnU-Net              │
  ├─────────────────────┼──────────────────┼──────────────────────┤
  │ Normalization       │ BatchNorm        │ InstanceNorm         │
  │ Activation          │ ReLU             │ Leaky ReLU (0.01)    │
  │ Residual connection │ None             │ Within every block   │
  │ Downsampling        │ MaxPool          │ Strided Conv (2×2)   │
  │ Upsampling          │ Bilinear / TConv │ TransposedConv       │
  │ Depth               │ Fixed 4          │ Configurable         │
  │ Base features       │ Fixed 64         │ Configurable         │
  └─────────────────────┴──────────────────┴──────────────────────┘

Args:
    n_channels    : Input channels (e.g. 3 for RGB)
    n_classes     : Output segmentation classes
    base_features : Feature channels at first encoder level (default 32)
    depth         : Number of encoder/decoder levels (default 5)
"""

import torch
import torch.nn as nn
from modules.nnunet.nnunet_parts import EncoderBlock, Bottleneck, DecoderBlock


class NNUNet(nn.Module):

    def __init__(self, n_channels: int, n_classes: int,
                 base_features: int = 32, depth: int = 5):
        super().__init__()
        self.n_channels    = n_channels
        self.n_classes     = n_classes
        self.base_features = base_features
        self.depth         = depth

        # Feature map sizes at each depth level
        # e.g. base=32, depth=5 → [32, 64, 128, 256, 320]
        # nnU-Net caps features at 320 to control memory
        feats = [min(base_features * (2 ** i), 320) for i in range(depth)]

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        in_ch = n_channels
        for out_ch in feats[:-1]:          # all levels except bottleneck
            self.encoders.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = Bottleneck(feats[-2], feats[-1])

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        dec_feats = list(reversed(feats))  # [320, 256, 128, 64, 32]
        for i in range(len(dec_feats) - 1):
            in_ch   = dec_feats[i]
            skip_ch = dec_feats[i + 1]
            out_ch  = dec_feats[i + 1]
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch))

        # ── Output head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(feats[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode — collect skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode — consume skips in reverse order
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)
