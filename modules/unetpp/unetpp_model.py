"""
modules/unetpp/unetpp_model.py
-------------------------------
UNet++ (UNet with nested dense skip connections).
Reference: Zhou et al. 2018 (https://arxiv.org/abs/1807.10165)

Dense skip pathway redesign:
  Standard UNet:  Encoder level i ────────────────────────► Decoder level i
  UNet++:         Encoder level i → X^{i,1} → X^{i,2} → X^{i,3} → Decoder

Channel layout (filters = [64, 128, 256, 512, 1024]):

  Level 0 (64ch) : x00 → x01 → x02 → x03 → x04  ← final output used
  Level 1 (128ch): x10 → x11 → x12 → x13
  Level 2 (256ch): x20 → x21 → x22
  Level 3 (512ch): x30 → x31
  Level 4 (1024ch): x40  ← bottleneck

Each x^{i,j} input channels:
  filters[i] * j          (all j previous nodes at level i)
  + filters[i]            (upsampled node from level i+1, reduced to filters[i])
  = filters[i] * (j + 1)

Args:
    n_channels : Input image channels
    n_classes  : Output segmentation classes
    deep_supervision : If True, average outputs from x04, x03, x02, x01 (4 heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.unetpp.unetpp_parts import VGGBlock, upsample_and_pad


class UNetPP(nn.Module):

    def __init__(self, n_channels: int, n_classes: int,
                 deep_supervision: bool = False):
        super().__init__()
        self.n_channels       = n_channels
        self.n_classes        = n_classes
        self.deep_supervision = deep_supervision

        nb_filter = [64, 128, 256, 512, 1024]

        # ── Pooling ───────────────────────────────────────────────────────
        self.pool = nn.MaxPool2d(2, 2)

        # ── Encoder nodes x^{i,0} ─────────────────────────────────────────
        self.x00 = VGGBlock(n_channels,       nb_filter[0])
        self.x10 = VGGBlock(nb_filter[0],     nb_filter[1])
        self.x20 = VGGBlock(nb_filter[1],     nb_filter[2])
        self.x30 = VGGBlock(nb_filter[2],     nb_filter[3])
        self.x40 = VGGBlock(nb_filter[3],     nb_filter[4])

        # ── Up-conv: reduce channels from level i+1 → level i ─────────────
        # These project the upsampled tensor to filters[i] channels
        self.up10 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)
        self.up20 = nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1)
        self.up30 = nn.Conv2d(nb_filter[3], nb_filter[2], kernel_size=1)
        self.up40 = nn.Conv2d(nb_filter[4], nb_filter[3], kernel_size=1)

        self.up11 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)
        self.up21 = nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1)
        self.up31 = nn.Conv2d(nb_filter[3], nb_filter[2], kernel_size=1)

        self.up12 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)
        self.up22 = nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1)

        self.up13 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)

        # ── Dense nodes x^{i,j} for j > 0 ────────────────────────────────
        # Input channels = nb_filter[i] * (j existing nodes + 1 upsampled)

        # j = 1
        self.x01 = VGGBlock(nb_filter[0] * 2, nb_filter[0])   # x00 + up(x10)
        self.x11 = VGGBlock(nb_filter[1] * 2, nb_filter[1])   # x10 + up(x20)
        self.x21 = VGGBlock(nb_filter[2] * 2, nb_filter[2])   # x20 + up(x30)
        self.x31 = VGGBlock(nb_filter[3] * 2, nb_filter[3])   # x30 + up(x40)

        # j = 2
        self.x02 = VGGBlock(nb_filter[0] * 3, nb_filter[0])   # x00, x01 + up(x11)
        self.x12 = VGGBlock(nb_filter[1] * 3, nb_filter[1])   # x10, x11 + up(x21)
        self.x22 = VGGBlock(nb_filter[2] * 3, nb_filter[2])   # x20, x21 + up(x31)

        # j = 3
        self.x03 = VGGBlock(nb_filter[0] * 4, nb_filter[0])   # x00,x01,x02 + up(x12)
        self.x13 = VGGBlock(nb_filter[1] * 4, nb_filter[1])   # x10,x11,x12 + up(x22)

        # j = 4 (final)
        self.x04 = VGGBlock(nb_filter[0] * 5, nb_filter[0])   # x00..x03 + up(x13)

        # ── Output heads ──────────────────────────────────────────────────
        if deep_supervision:
            self.head1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.head2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.head3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.head4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.head  = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def _up(self, x, conv, target):
        """Upsample x, apply 1x1 conv to reduce channels, pad to target size."""
        x = upsample_and_pad(x, target)
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ───────────────────────────────────────────────────────
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        # ── Dense nodes j=1 ───────────────────────────────────────────────
        x01 = self.x01(torch.cat([x00, self._up(x10, self.up10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up(x20, self.up20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self._up(x30, self.up30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, self._up(x40, self.up40, x30)], dim=1))

        # ── Dense nodes j=2 ───────────────────────────────────────────────
        x02 = self.x02(torch.cat([x00, x01, self._up(x11, self.up11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, self.up21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self._up(x31, self.up31, x20)], dim=1))

        # ── Dense nodes j=3 ───────────────────────────────────────────────
        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, self.up12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self._up(x22, self.up22, x10)], dim=1))

        # ── Dense nodes j=4 ───────────────────────────────────────────────
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self._up(x13, self.up13, x00)], dim=1))

        # ── Output ────────────────────────────────────────────────────────
        if self.deep_supervision:
            o1 = self.head1(x01)
            o2 = self.head2(x02)
            o3 = self.head3(x03)
            o4 = self.head4(x04)
            return (o1 + o2 + o3 + o4) / 4   # average over all supervision heads
        else:
            return self.head(x04)
