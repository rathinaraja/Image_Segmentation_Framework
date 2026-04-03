"""
modules/segformer/segformer_parts.py
-------------------------------------
Building blocks for SegFormer.
Reference: Xie et al. 2021 (https://arxiv.org/abs/2105.15203)

Two main components:

1. Mix Transformer Encoder (MiT)
   ─────────────────────────────
   4 hierarchical stages, each containing:

   ┌── OverlapPatchEmbed ──────────────────────────────────────────────┐
   │   Overlapping conv (no positional encoding needed — DWConv        │
   │   in Mix-FFN implicitly encodes local position)                   │
   └───────────────────────────────────────────────────────────────────┘
   ┌── EfficientSelfAttention ─────────────────────────────────────────┐
   │   Reduces key/value sequence by factor R² via strided conv        │
   │   (much faster than standard attention for large feature maps)    │
   │   Q: full sequence  K,V: reduced by ratio R                       │
   └───────────────────────────────────────────────────────────────────┘
   ┌── Mix-FFN ────────────────────────────────────────────────────────┐
   │   Linear → DWConv (3×3 depth-wise, provides local info) → GELU   │
   │   → Linear  (replaces positional encoding with local conv)        │
   └───────────────────────────────────────────────────────────────────┘

   Output per stage: feature map at H/4, H/8, H/16, H/32

2. All-MLP Decoder
   ────────────────
   - Linear projection of each stage output → unified decoder_dim
   - Upsample all to H/4 (coarsest to finest)
   - Concatenate → Conv fusion → head
   - Upsample to H×W (done in model forward)

   Unlike UNet decoder, no skip connections — pure MLP fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


# ── Stage 1: Overlapping Patch Embedding ──────────────────────────────────────

class OverlapPatchEmbed(nn.Module):
    """
    Partition image into overlapping patches via strided conv.
    Overlap provides continuity across patch boundaries.

    Input : (B, in_chans, H,    W)
    Output: (B, H'*W',   embed_dim)  where H' = H/stride
    """
    def __init__(self, in_chans: int, embed_dim: int,
                 patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                           # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, H'*W', C)
        return self.norm(x), H, W


# ── Stage 2: Efficient Self-Attention ─────────────────────────────────────────

class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with sequence reduction ratio R.
    K and V are computed on a spatially-reduced sequence (÷R²),
    while Q attends at full resolution.

    Complexity: O(N * N/R²) instead of O(N²)
    """
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.sr_ratio  = sr_ratio

        self.q         = nn.Linear(dim, dim)
        self.kv        = nn.Linear(dim, dim * 2)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Sequence reduction: spatial pooling via strided conv
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Reduce K, V sequence
        if self.sr_ratio > 1:
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            x_2d = self.sr(x_2d).reshape(B, C, -1).transpose(1, 2)  # (B, N/R², C)
            x_2d = self.norm(x_2d)
        else:
            x_2d = x

        kv = self.kv(x_2d).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ── Stage 3: Mix-FFN ──────────────────────────────────────────────────────────

class MixFFN(nn.Module):
    """
    Feed-Forward Network with a depth-wise conv in the middle.
    The DWConv encodes local positional information — replaces
    SegFormer's need for explicit positional encoding.

    Linear(dim→hidden) → DWConv(3×3) → GELU → Linear(hidden→dim)
    """
    def __init__(self, dim: int, mlp_ratio: float = 4., drop: float = 0.):
        super().__init__()
        hidden    = int(dim * mlp_ratio)
        self.fc1  = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        # Reshape to 2D for depth-wise conv
        x = self.dwconv(x.transpose(1, 2).reshape(B, C, H, W))
        x = self.act(x.flatten(2).transpose(1, 2))
        return self.drop(self.fc2(self.drop(x)))


# ── MiT Block ─────────────────────────────────────────────────────────────────

class MiTBlock(nn.Module):
    """One Mix Transformer block: Norm→ESA→residual, Norm→MixFFN→residual."""
    def __init__(self, dim: int, num_heads: int, sr_ratio: int,
                 mlp_ratio: float = 4., drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0.):
        super().__init__()
        self.norm1     = nn.LayerNorm(dim)
        self.attn      = EfficientSelfAttention(dim, num_heads, sr_ratio, attn_drop, drop)
        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = MixFFN(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


# ── Mix Transformer Encoder ───────────────────────────────────────────────────

class MixTransformer(nn.Module):
    """
    Hierarchical encoder with 4 stages producing multi-scale features.

    Stage  Patch  Stride  Resolution  Channels
    ─────  ─────  ──────  ──────────  ────────
      1      7      4       H/4        C1
      2      3      2       H/8        C2
      3      3      2       H/16       C3
      4      3      2       H/32       C4
    """
    def __init__(self, in_chans: int = 3,
                 embed_dims: list = None,
                 num_heads:  list = None,
                 mlp_ratios: list = None,
                 depths:     list = None,
                 sr_ratios:  list = None,
                 drop_rate:  float = 0.,
                 attn_drop:  float = 0.,
                 drop_path:  float = 0.1):
        super().__init__()

        embed_dims = embed_dims or [32, 64, 160, 256]
        num_heads  = num_heads  or [1, 2, 5, 8]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        depths     = depths     or [2, 2, 2, 2]
        sr_ratios  = sr_ratios  or [8, 4, 2, 1]

        # Stochastic depth decay
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        self.patch_embeds = nn.ModuleList()
        self.layers       = nn.ModuleList()
        self.norms        = nn.ModuleList()

        patch_sizes = [7, 3, 3, 3]
        strides     = [4, 2, 2, 2]
        in_ch       = in_chans
        cur         = 0

        for i in range(4):
            self.patch_embeds.append(
                OverlapPatchEmbed(in_ch, embed_dims[i], patch_sizes[i], strides[i]))
            self.layers.append(nn.ModuleList([
                MiTBlock(embed_dims[i], num_heads[i], sr_ratios[i],
                         mlp_ratios[i], drop_rate, attn_drop, dp_rates[cur + j])
                for j in range(depths[i])
            ]))
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            in_ch = embed_dims[i]
            cur  += depths[i]

    def forward(self, x: torch.Tensor):
        """Returns list of 4 feature maps: [F1(H/4), F2(H/8), F3(H/16), F4(H/32)]"""
        B    = x.shape[0]
        outs = []
        for pe, blocks, norm in zip(self.patch_embeds, self.layers, self.norms):
            x, H, W = pe(x)                                    # (B, N, C)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
            outs.append(x)
        return outs


# ── All-MLP Decoder ───────────────────────────────────────────────────────────

class AllMLPDecoder(nn.Module):
    """
    Lightweight MLP decoder — no UNet-style skip connections.

    For each scale:
      Linear projection → unified decoder_dim → upsample to H/4
    Then:
      Concatenate all 4 → Conv1×1 fusion → head → output at H/4
    Final bilinear upsample to H×W done in SegFormer.forward()

    This simple design beats heavy decoders because the MiT encoder
    already captures rich multi-scale context.
    """
    def __init__(self, embed_dims: list, decoder_dim: int, n_classes: int,
                 drop: float = 0.1):
        super().__init__()
        # One linear projection per encoder stage
        self.linear_projs = nn.ModuleList([
            nn.Linear(dim, decoder_dim) for dim in embed_dims
        ])
        # Fuse all 4 projected+upsampled features
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.head = nn.Conv2d(decoder_dim, n_classes, kernel_size=1)

    def forward(self, features: list) -> torch.Tensor:
        # Upsample all feature maps to match the finest scale (H/4)
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        projected = []
        for feat, linear in zip(features, self.linear_projs):
            B, C, H, W = feat.shape
            # Flatten → project → reshape → upsample
            x = linear(feat.flatten(2).transpose(1, 2))       # (B, H*W, decoder_dim)
            x = x.transpose(1, 2).reshape(B, -1, H, W)        # (B, decoder_dim, H, W)
            x = F.interpolate(x, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
            projected.append(x)

        x = self.fuse(torch.cat(projected, dim=1))             # (B, decoder_dim, H/4, W/4)
        return self.head(x)                                    # (B, n_classes, H/4, W/4)
