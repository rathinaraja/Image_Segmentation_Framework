"""
modules/swinunet/swinunet_model.py
-----------------------------------
Swin-UNet: Pure Transformer encoder-decoder for segmentation.
Reference: Cao et al. 2021 (https://arxiv.org/abs/2105.05537)

  ┌────────────────────────────────────────────────────────────────────┐
  │  Input (B, 3, H, W)                                                │
  │      ↓                                                             │
  │  PatchEmbed (patch_size=4) → (B, H/4*W/4, C)                      │
  │      ↓                                                             │
  │  Encoder Stage 0 ────────────────────── skip0 (H/4,  C)            │
  │  PatchMerging   → (H/8,  2C)                                       │
  │  Encoder Stage 1 ────────────────────── skip1 (H/8,  2C)           │
  │  PatchMerging   → (H/16, 4C)                                       │
  │  Encoder Stage 2 ────────────────────── skip2 (H/16, 4C)           │
  │  PatchMerging   → (H/32, 8C)                                       │
  │  Bottleneck     ────────────────────── (H/32, 8C)                  │
  │      ↓                                                             │
  │  PatchExpanding → (H/16, 4C) + skip2 → Decoder Stage 2            │
  │  PatchExpanding → (H/8,  2C) + skip1 → Decoder Stage 1            │
  │  PatchExpanding → (H/4,  C)  + skip0 → Decoder Stage 0            │
  │  FinalExpanding → (H, W, C//16)                                    │
  │      ↓                                                             │
  │  Head: Conv1x1 → (n_classes, H, W)                                 │
  └────────────────────────────────────────────────────────────────────┘

Args:
    n_channels  : Input image channels (3 for RGB)
    n_classes   : Segmentation output classes
    img_size    : Input image size (H = W)
    patch_size  : Patch partition size (default 4)
    embed_dim   : Base embedding dimension C (default 96)
    depths      : Number of SwinBlocks per encoder stage (default [2,2,2,2])
    num_heads   : Attention heads per stage (default [3,6,12,24])
    window_size : Local attention window size (default 7)
    mlp_ratio   : FFN expansion ratio (default 4.0)
    drop_rate   : Dropout rate (default 0.0)
    attn_drop   : Attention dropout rate (default 0.0)
"""

import torch
import torch.nn as nn
from modules.swinunet.swinunet_parts import (
    PatchEmbed, BasicLayer, PatchMerging,
    PatchExpanding, FinalExpanding
)


class SwinUNet(nn.Module):

    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 img_size: int = 224, patch_size: int = 4,
                 embed_dim: int = 96,
                 depths: list = None,
                 num_heads: list = None,
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()

        depths    = depths    or [2, 2, 2, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.embed_dim  = embed_dim
        self.num_stages = len(depths) - 1   # last depth = bottleneck

        # ── Patch embedding ───────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_size, patch_size, n_channels, embed_dim)
        self.pos_drop    = nn.Dropout(drop_rate)

        # ── Encoder stages + patch merging ────────────────────────────
        self.enc_layers   = nn.ModuleList()
        self.merging_layers = nn.ModuleList()
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        for i in range(self.num_stages):
            self.enc_layers.append(
                BasicLayer(dims[i], depths[i], num_heads[i],
                           window_size=window_size, mlp_ratio=mlp_ratio,
                           drop=drop_rate, attn_drop=attn_drop))
            self.merging_layers.append(PatchMerging(dims[i]))

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = BasicLayer(dims[-1], depths[-1], num_heads[-1],
                                     window_size=window_size, mlp_ratio=mlp_ratio,
                                     drop=drop_rate, attn_drop=attn_drop)

        # ── Decoder stages + patch expanding ──────────────────────────
        self.dec_layers    = nn.ModuleList()
        self.expanding_layers = nn.ModuleList()
        self.concat_proj   = nn.ModuleList()  # project after skip concat

        for i in range(self.num_stages - 1, -1, -1):
            self.expanding_layers.append(PatchExpanding(dims[i + 1]))
            self.concat_proj.append(nn.Linear(dims[i] * 2, dims[i], bias=False))
            self.dec_layers.append(
                BasicLayer(dims[i], depths[i], num_heads[i],
                           window_size=window_size, mlp_ratio=mlp_ratio,
                           drop=drop_rate, attn_drop=attn_drop))

        # ── Final upsampling + head ────────────────────────────────────
        self.final_expand = FinalExpanding(embed_dim, embed_dim // 4)
        self.head         = nn.Conv2d(embed_dim // 4, n_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed
        x, H, W = self.patch_embed(x)   # (B, H/4*W/4, C)
        x        = self.pos_drop(x)

        # Encoder: collect (x, H, W) at each stage for skip connections
        skips = []
        for enc, merge in zip(self.enc_layers, self.merging_layers):
            x = enc(x, H, W)
            skips.append((x, H, W))
            x = merge(x, H, W)
            H, W = H // 2, W // 2

        # Bottleneck
        x = self.bottleneck(x, H, W)

        # Decoder: expand → concat skip → project → decode
        for expand, proj, dec, (skip_x, sH, sW) in zip(
                self.expanding_layers, self.concat_proj,
                self.dec_layers, reversed(skips)):
            x  = expand(x, H, W)        # (B, sH*sW, C_lower)
            H, W = sH, sW
            x  = proj(torch.cat([skip_x, x], dim=-1))  # concat + project
            x  = dec(x, H, W)

        # Final 4× expansion + head
        x = self.final_expand(x, H, W)  # (B, embed_dim//4, 4H, 4W) = (B, C//4, H_orig, W_orig)
        return self.head(x)
