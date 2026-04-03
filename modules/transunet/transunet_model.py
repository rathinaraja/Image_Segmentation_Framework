"""
modules/transunet/transunet_model.py
-------------------------------------
TransUNet: Hybrid CNN + Transformer segmentation model.
Reference: Chen et al. 2021 (https://arxiv.org/abs/2102.04306)

  ┌────────────────────────────────────────────────────────────────────┐
  │  Input (B, 3, H, W)                                                │
  │      ↓                                                             │
  │  Stem Conv ──────────────────────────────────── skip0 (64, H/2)   │
  │      ↓                                                             │
  │  Encoder Stage 1 ────────────────────────────── skip1 (64, H/4)   │
  │      ↓                                                             │
  │  Encoder Stage 2 ────────────────────────────── skip2 (128, H/8)  │
  │      ↓                                                             │
  │  Encoder Stage 3 ────────────────────────────── skip3 (256, H/16) │
  │      ↓                                                             │
  │  Flatten → (B, (H/16)*(W/16), 256)                                 │
  │      ↓                                                             │
  │  Transformer Encoder (depth layers)                                │
  │      ↓                                                             │
  │  Reshape → (B, 256, H/16, W/16)                                    │
  │      ↓                                                             │
  │  Decoder 3: TConv + skip3 → (128, H/8)                            │
  │  Decoder 2: TConv + skip2 → (64,  H/4)                            │
  │  Decoder 1: TConv + skip1 → (64,  H/2)                            │
  │  Decoder 0: TConv + skip0 → (16,  H)                              │
  │      ↓                                                             │
  │  Head: Conv1x1 → (n_classes, H, W)                                 │
  └────────────────────────────────────────────────────────────────────┘

Args:
    n_channels    : Input channels (3 for RGB)
    n_classes     : Output segmentation classes
    img_size      : Input image size (H=W assumed)
    embed_dim     : Transformer embedding dimension (default 256)
    trans_depth   : Number of transformer layers (default 12)
    num_heads     : Transformer attention heads (default 8)
    mlp_ratio     : FFN hidden size multiplier (default 4.0)
    dropout       : Dropout rate (default 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transunet.transunet_parts import (
    EncoderStage, TransformerEncoder, DecoderBlock
)
 
 
class TransUNet(nn.Module):
 
    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 img_size: int = 512, embed_dim: int = 256,
                 trans_depth: int = 6, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.embed_dim  = embed_dim
 
        # ── CNN Encoder ───────────────────────────────────────────────
        # stem:   H → H/2  (stride=2)
        self.stem   = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # stage1: H/2 → H/2  (stride=1, no downsampling)
        self.stage1 = EncoderStage(64,  64,  n_blocks=2, stride=1)
        # stage2: H/2 → H/4  (stride=2)
        self.stage2 = EncoderStage(64,  128, n_blocks=2, stride=2)
        # stage3: H/4 → H/8  (stride=2)  → fed to transformer
        self.stage3 = EncoderStage(128, embed_dim, n_blocks=2, stride=2)
 
        # ── Transformer ───────────────────────────────────────────────
        # After stage3: (B, embed_dim, H/8, W/8)
        # Downsampling: stem(÷2) × stage1(÷1) × stage2(÷2) × stage3(÷2) = ÷8
        patch_h     = img_size // 8
        patch_w     = img_size // 8
        num_patches = patch_h * patch_w    # 64*64=4096 for img_size=512
 
        self.transformer = TransformerEncoder(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            depth       = trans_depth,
            num_patches = num_patches,
            mlp_ratio   = mlp_ratio,
            dropout     = dropout,
        )
 
        # Project transformer output to 512 channels
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
 
        # ── CNN Decoder (3 upsampling steps) ─────────────────────────
        # dec2: (512, H/8) → up → (256, H/4) + skip2(128, H/4) → (128, H/4)
        self.dec2 = DecoderBlock(in_channels=512, skip_channels=128, out_channels=128)
 
        # dec1: (128, H/4) → up → (64, H/2) + skip0(64, H/2)  → (64, H/2)
        self.dec1 = DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)
 
        # up_final: (64, H/2) → up → (32, H) → conv → (16, H)  [no skip at full res]
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
 
        # ── Output head ───────────────────────────────────────────────
        self.head = nn.Conv2d(16, n_classes, kernel_size=1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
 
        # ── Encode ────────────────────────────────────────────────────
        skip0 = self.stem(x)           # (B, 64,  H/2, W/2)
        skip0 = self.stage1(skip0)     # (B, 64,  H/2, W/2)  ← used in dec1
        skip2 = self.stage2(skip0)     # (B, 128, H/4, W/4)  ← used in dec2
        feat  = self.stage3(skip2)     # (B, 256, H/8, W/8)  ← fed to transformer
 
        # ── Transformer ───────────────────────────────────────────────
        H8, W8  = feat.shape[2], feat.shape[3]
        tokens  = feat.flatten(2).transpose(1, 2)            # (B, H8*W8, embed_dim)
        tokens  = self.transformer(tokens)                    # (B, H8*W8, embed_dim)
        feat    = tokens.transpose(1, 2).reshape(B, self.embed_dim, H8, W8)
        feat    = self.proj(feat)                             # (B, 512, H/8, W/8)
 
        # ── Decode ────────────────────────────────────────────────────
        x = self.dec2(feat,  skip2)    # (B, 128, H/4, W/4)
        x = self.dec1(x,     skip0)    # (B, 64,  H/2, W/2)
        x = self.up_final(x)           # (B, 16,  H,   W)
 
        return self.head(x)            # (B, n_classes, H, W)