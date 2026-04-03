"""
modules/transunet/transunet_parts.py
-------------------------------------
Building blocks for TransUNet.
Reference: Chen et al. 2021 (https://arxiv.org/abs/2102.04306)

Architecture flow:
  Input
    ↓
  ResNet Encoder  (3 stages → skip1, skip2, skip3)
    ↓
  Reshape feature map → token sequence
    ↓
  Transformer Encoder  (L layers of MSA + FFN)
    ↓
  Reshape back → 2D feature map
    ↓
  CNN Decoder  (upsampling + skip connections)
    ↓
  Output head

Key insight: CNN handles local texture, Transformer captures global context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CNN Encoder blocks ─────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-activation ResNet block (ResNet-V2 style)."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if (stride != 1 or in_channels != out_channels) else nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + self.shortcut(x)


class EncoderStage(nn.Module):
    """Stack of ResBlocks with optional downsampling at the first block."""
    def __init__(self, in_channels, out_channels, n_blocks=2, stride=2):
        super().__init__()
        blocks = [ResBlock(in_channels, out_channels, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_channels, out_channels))
        self.stage = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stage(x)


# ── Transformer blocks ─────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.qkv     = nn.Linear(embed_dim, embed_dim * 3)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer layer: LayerNorm → MSA → residual → LayerNorm → FFN → residual."""
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of TransformerBlocks with learnable 1D positional encoding.
    Applied to flattened CNN feature map tokens.
    """
    def __init__(self, embed_dim: int, num_heads: int, depth: int,
                 num_patches: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return self.norm(x)


# ── CNN Decoder blocks ─────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Upsample × 2 → concat skip → DoubleConv."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                        kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                               0, skip.shape[2] - x.shape[2]])
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)
