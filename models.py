"""
模型定义：多帧早期融合（Early Fusion）UNet，输出单通道 heatmap。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """
    Conv2d + BatchNorm + SiLU。
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        """
        @param {int} in_ch - 输入通道
        @param {int} out_ch - 输出通道
        @param {int} k - kernel size
        @param {int} s - stride
        @param {int} p - padding
        """

        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param {torch.Tensor} x - 输入
        @returns {torch.Tensor} - 输出
        """

        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """
    UNet 的 double conv block。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """
        @param {int} in_ch - 输入通道
        @param {int} out_ch - 输出通道
        """

        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param {torch.Tensor} x - 输入
        @returns {torch.Tensor} - 输出
        """

        return self.net(x)


class Down(nn.Module):
    """
    下采样模块：MaxPool + DoubleConv。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    上采样模块：Upsample + concat skip + DoubleConv。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad to match
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


@dataclass
class ModelOutput:
    """
    模型输出结构。

    @param {torch.Tensor} heatmap_logits - (B,1,Hm,Wm) logits
    """

    heatmap_logits: torch.Tensor


class EarlyFusionUNet(nn.Module):
    """
    多帧早期融合 UNet：

    输入： (B, window*3, H, W)
    输出： (B, 1, Hm, Wm) heatmap logits
    """

    def __init__(
        self,
        window: int = 5,
        base_ch: int = 32,
        out_size: Tuple[int, int] = (160, 160),
    ) -> None:
        """
        @param {int} window - 多帧窗口长度
        @param {int} base_ch - UNet 基础通道数（越小越省显存，推荐 16/24/32）
        @param {Tuple[int,int]} out_size - 输出 heatmap 的 (H,W)
        """

        super().__init__()
        self.window = int(window)
        self.out_size = (int(out_size[0]), int(out_size[1]))

        in_ch = self.window * 3

        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.mid = DoubleConv(base_ch * 8, base_ch * 8)
        self.up1 = Up(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up3 = Up(base_ch * 2 + base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        @param {torch.Tensor} x - (B, window*3, H, W)
        @returns {ModelOutput} - 输出
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xm = self.mid(x4)
        x = self.up1(xm, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        # 统一输出到指定 heatmap 尺寸（便于坐标映射与训练稳定）
        logits = F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)
        return ModelOutput(heatmap_logits=logits)


