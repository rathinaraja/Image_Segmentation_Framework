"""
modules/swinunet/swinunet_parts.py
-----------------------------------
Building blocks for Swin-UNet.
Reference: Cao et al. 2021 (https://arxiv.org/abs/2105.05537)

Key components:
  PatchEmbed    : Split image into non-overlapping patches + linear projection
  WindowAttention : Local window multi-head self-attention (W-MSA / SW-MSA)
  SwinBlock     : Two consecutive blocks — regular window + shifted window attention
  PatchMerging  : 2× spatial downsampling (merge 2×2 patch regions)
  PatchExpanding: 2× spatial upsampling  (expand each patch into 2×2)
  BasicLayer    : Stack of SwinBlocks for one encoder/decoder stage
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


# ── Utility ────────────────────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) into windows (num_windows*B, Ws, Ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """Reverse window partition back to (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── Patch embedding ────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Split image into non-overlapping patches and project to embed_dim.
    Input : (B, C, H, W)
    Output: (B, H/patch_size * W/patch_size, embed_dim)
    """
    def __init__(self, img_size: int = 224, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size   = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size  = (self.img_size[0] // self.patch_size[0],
                           self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                         # (B, embed_dim, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, H*W, embed_dim)
        return self.norm(x), H, W


# ── Window attention ───────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    Supports both regular (shift=0) and shifted (shift=shift_size) windows.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # Relative position bias table
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Relative position index for each token pair in a window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Ws, Ws)
        coords_flat = coords.flatten(1)                                              # (2, Ws*Ws)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]              # (2, Ws*Ws, Ws*Ws)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", rel_coords.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias
        rel_pos_bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        rel_pos_bias = rel_pos_bias.view(
            self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── Swin Transformer Block ─────────────────────────────────────────────────────

class SwinBlock(nn.Module):
    """
    One Swin Transformer block.
    shift_size=0       → regular W-MSA
    shift_size=window//2 → shifted SW-MSA
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.dim        = dim
        self.shift_size = shift_size
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int,
                attn_mask=None) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to multiple of window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window partition → attention → reverse
        x_win  = window_partition(x, self.window_size)
        x_win  = x_win.view(-1, self.window_size ** 2, C)
        x_win  = self.attn(x_win, mask=attn_mask)
        x_win  = x_win.view(-1, self.window_size, self.window_size, C)
        x      = window_reverse(x_win, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x.view(B, H * W, C))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ── Attention mask helper ──────────────────────────────────────────────────────

def compute_attn_mask(H: int, W: int, window_size: int,
                      shift_size: int, device) -> torch.Tensor:
    """Compute attention mask for SW-MSA to prevent cross-region attention."""
    img_mask = torch.zeros(1, H, W, 1, device=device)
    slices_h = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    slices_w = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in slices_h:
        for w in slices_w:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size ** 2)
    mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


# ── BasicLayer (encoder/decoder stage) ────────────────────────────────────────

class BasicLayer(nn.Module):
    """
    A stack of SwinBlocks for one stage (alternating W-MSA and SW-MSA).
    """
    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = window_size // 2
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[i],
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int):
        # Compute attention mask once per layer (only for shifted blocks)
        attn_mask = compute_attn_mask(H, W, self.window_size, self.shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask if blk.shift_size > 0 else None)
        return x


# ── Patch Merging (downsampling) ───────────────────────────────────────────────

class PatchMerging(nn.Module):
    """
    Merge 2×2 neighboring patches → halve spatial resolution, double channels.
    (B, H*W, C) → (B, H/2 * W/2, 2C)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        # Pad if odd spatial dims
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x  = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


# ── Patch Expanding (upsampling) ───────────────────────────────────────────────

class PatchExpanding(nn.Module):
    """
    Expand each patch into 2×2 patches → double spatial resolution, halve channels.
    (B, H*W, C) → (B, 2H * 2W, C//2)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = self.expand(x)                                   # (B, H*W, 2C)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = x.view(B, H, W, 2, 2, C // 4)                  # split into 2×2 sub-patches
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 4)
        x = x.view(B, -1, C // 4)
        return self.norm(x)


class FinalExpanding(nn.Module):
    """
    4× upsampling for the final decoder stage (patch_size=4 reversal).
    (B, H*W, C) → (B, 4H * 4W, C//16)  then reshape to (B, C//16, 4H, 4W)
    """
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 16 * out_dim, bias=False)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = self.expand(x)                                   # (B, H*W, 16*out_dim)
        B, L, C_new = x.shape
        out_dim = C_new // 16
        x = x.view(B, H, W, 4, 4, out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 4 * H, 4 * W, out_dim)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)                        # (B, out_dim, 4H, 4W)
