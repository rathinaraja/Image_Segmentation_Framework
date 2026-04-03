"""
modules/segformer/segformer_model.py
-------------------------------------
SegFormer: Hierarchical Transformer + All-MLP Decoder.
Reference: Xie et al. 2021 (https://arxiv.org/abs/2105.15203)

Comparison with other models in the framework:

  Model           Encoder          Decoder       Positional Enc
  ─────────────── ──────────────── ───────────── ──────────────
  UNet            CNN              CNN (skip)    —
  Attention UNet  CNN              CNN (attn)    —
  TransUNet       CNN + ViT        CNN (skip)    1D learnable
  Swin-UNet       Swin Transformer UNet-style    Relative bias
  SegFormer       Mix Transformer  All-MLP       None (DWConv)

SegFormer's key novelty:
  - No positional encoding at all → works on any input resolution
  - Efficient self-attention with sequence reduction ratio R
  - Ultra-lightweight All-MLP decoder (no convolutions, just MLPs)
  - Multi-scale feature fusion without skip connections

Built-in variants (set via config):

  Variant  embed_dims          depths        ~Params
  ───────  ──────────────────  ────────────  ───────
  B0       [32, 64, 160, 256]  [2,2,2,2]     3.8M   ← default (fast)
  B1       [64,128, 320, 512]  [2,2,2,2]    13.7M
  B2       [64,128, 320, 512]  [3,4,6,3]    27.5M
  B3       [64,128, 320, 512]  [3,4,18,3]   47.3M

Args:
    n_channels  : Input image channels (3 for RGB)
    n_classes   : Segmentation output classes
    embed_dims  : Channel sizes per stage
    num_heads   : Attention heads per stage
    mlp_ratios  : FFN expansion ratio per stage
    depths      : Number of MiT blocks per stage
    sr_ratios   : Sequence reduction ratios per stage (8,4,2,1 for B0)
    decoder_dim : Unified MLP decoder dimension (default 256)
    drop_rate   : Dropout in FFN and attention projection
    attn_drop   : Attention weight dropout
    drop_path   : Stochastic depth max rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.segformer.segformer_parts import MixTransformer, AllMLPDecoder


class SegFormer(nn.Module):

    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 embed_dims: list = None,
                 num_heads:  list = None,
                 mlp_ratios: list = None,
                 depths:     list = None,
                 sr_ratios:  list = None,
                 decoder_dim: int  = 256,
                 drop_rate:   float = 0.,
                 attn_drop:   float = 0.,
                 drop_path:   float = 0.1):
        super().__init__()

        # Default to MiT-B0 (smallest variant)
        embed_dims = embed_dims or [32,  64,  160, 256]
        num_heads  = num_heads  or [1,   2,   5,   8]
        mlp_ratios = mlp_ratios or [4,   4,   4,   4]
        depths     = depths     or [2,   2,   2,   2]
        sr_ratios  = sr_ratios  or [8,   4,   2,   1]

        self.encoder = MixTransformer(
            in_chans    = n_channels,
            embed_dims  = embed_dims,
            num_heads   = num_heads,
            mlp_ratios  = mlp_ratios,
            depths      = depths,
            sr_ratios   = sr_ratios,
            drop_rate   = drop_rate,
            attn_drop   = attn_drop,
            drop_path   = drop_path,
        )

        self.decoder = AllMLPDecoder(
            embed_dims  = embed_dims,
            decoder_dim = decoder_dim,
            n_classes   = n_classes,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Multi-scale encoder: [F1(H/4), F2(H/8), F3(H/16), F4(H/32)]
        features = self.encoder(x)

        # All-MLP decoder → output at H/4
        out = self.decoder(features)                           # (B, n_classes, H/4, W/4)

        # Upsample to full input resolution
        out = F.interpolate(out, size=(H, W),
                            mode='bilinear', align_corners=False)
        return out                                             # (B, n_classes, H, W)
