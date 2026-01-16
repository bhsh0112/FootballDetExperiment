"""
数据集：从帧目录 + 点标注 CSV 构建多帧滑窗样本，并生成 heatmap 监督。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    LetterboxInfo,
    bgr_to_rgb_float,
    letterbox,
    make_gaussian_heatmap,
    normalize_imagenet,
    orig_xy_to_input_xy,
    safe_float,
)


@dataclass(frozen=True)
class PointAnno:
    """
    单帧球心点标注。

    @param {str} frame - 帧文件名（相对 frames_dir）
    @param {int} visible - 1 可见，0 不可见
    @param {Optional[float]} x - 原图像素坐标 x
    @param {Optional[float]} y - 原图像素坐标 y
    """

    frame: str
    visible: int
    x: Optional[float]
    y: Optional[float]


def load_points_csv(points_csv: str) -> List[PointAnno]:
    """
    读取点标注 CSV。

    CSV 字段：frame,x,y,visible

    @param {str} points_csv - CSV 路径
    @returns {List[PointAnno]} - 按行顺序的标注列表
    """

    out: List[PointAnno] = []
    with open(points_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = str(row.get("frame", "")).strip()
            if frame == "":
                continue
            visible = int(str(row.get("visible", "1")).strip() or "1")
            x = safe_float(row.get("x"))
            y = safe_float(row.get("y"))
            if visible == 0:
                x, y = None, None
            out.append(PointAnno(frame=frame, visible=visible, x=x, y=y))
    return out


class SequencePointDataset(Dataset):
    """
    多帧滑窗数据集：以“窗口最后一帧”为预测目标。

    输入：window 帧（letterbox 到 img_size），按时间顺序拼成 (window*3,H,W)
    标签：最后一帧的 heatmap（heatmap_size），不可见时为全零
    """

    def __init__(
        self,
        frames_dir: str,
        points_csv: str,
        window: int = 5,
        img_size: int = 640,
        heatmap_size: int = 160,
        sigma: float = 2.0,
        lambda_no_ball: float = 0.05,
    ) -> None:
        """
        @param {str} frames_dir - 帧目录
        @param {str} points_csv - 点标注 CSV
        @param {int} window - 滑窗长度
        @param {int} img_size - 网络输入尺寸（正方形）
        @param {int} heatmap_size - heatmap 尺寸（正方形）
        @param {float} sigma - heatmap 高斯 sigma（heatmap 像素）
        @param {float} lambda_no_ball - 不可见帧的 heatmap loss 权重（防止全零崩塌）
        """

        super().__init__()
        self.frames_dir = frames_dir
        self.window = int(window)
        self.img_size = int(img_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.lambda_no_ball = float(lambda_no_ball)

        annos = load_points_csv(points_csv)
        # 按 frame 名称排序（确保时间顺序，适配 frame_000001.jpg / *_frame0007200.jpg 等）
        annos = sorted(annos, key=lambda a: a.frame)
        self.annos: List[PointAnno] = annos

        if len(self.annos) == 0:
            raise ValueError(f"points_csv 为空或无有效行：{points_csv}")

    def __len__(self) -> int:
        return len(self.annos)

    def _read_frame(self, frame_rel: str) -> Tuple[np.ndarray, LetterboxInfo]:
        """
        读取帧并做 letterbox 预处理。

        @param {str} frame_rel - 相对 frames_dir 的文件名
        @returns {Tuple[np.ndarray, LetterboxInfo]} - (RGB normalized CHW, info)
        """

        path = os.path.join(self.frames_dir, frame_rel)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")

        lb, info = letterbox(img, (self.img_size, self.img_size))
        rgb01 = bgr_to_rgb_float(lb)
        rgbn = normalize_imagenet(rgb01)
        chw = np.transpose(rgbn, (2, 0, 1)).astype(np.float32)  # (3,H,W)
        return chw, info

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        @param {int} idx - 样本索引（对应目标帧）
        @returns {Dict[str, torch.Tensor]} - batch 字典
        """

        # 取 window 帧：左侧不足则重复第一帧
        start = idx - self.window + 1
        indices = [max(0, i) for i in range(start, idx + 1)]

        frames: List[np.ndarray] = []
        infos: List[LetterboxInfo] = []
        for i in indices:
            chw, info = self._read_frame(self.annos[i].frame)
            frames.append(chw)
            infos.append(info)

        # 输入：按时间拼接到 channel 维度 (window*3,H,W)
        x = np.concatenate(frames, axis=0).astype(np.float32)

        # 标签：窗口最后一帧
        target_anno = self.annos[idx]
        last_info = infos[-1]

        if target_anno.visible == 1 and target_anno.x is not None and target_anno.y is not None:
            x_in, y_in = orig_xy_to_input_xy(target_anno.x, target_anno.y, last_info)
            sx = self.heatmap_size / float(self.img_size)
            cx = x_in * sx
            cy = y_in * sx
            y_hm = make_gaussian_heatmap(
                heatmap_h=self.heatmap_size,
                heatmap_w=self.heatmap_size,
                center_x=cx,
                center_y=cy,
                sigma=self.sigma,
            )
            visible = 1.0
            loss_w = 1.0
        else:
            y_hm = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)
            visible = 0.0
            loss_w = float(self.lambda_no_ball)

        sample = {
            "x": torch.from_numpy(x),  # (C,H,W)
            "y": torch.from_numpy(y_hm[None, ...]),  # (1,Hm,Wm)
            "visible": torch.tensor([visible], dtype=torch.float32),
            "loss_w": torch.tensor([loss_w], dtype=torch.float32),
        }
        return sample


