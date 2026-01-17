"""
通用工具：图像预处理、letterbox 变换、heatmap 生成与坐标映射。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    """
    letterbox 变换信息（用于坐标正/逆变换）。

    @param {float} scale - 原图到输入图的缩放比例（等比）
    @param {int} pad_x - 左侧 padding（像素）
    @param {int} pad_y - 顶部 padding（像素）
    @param {int} in_w - 网络输入宽
    @param {int} in_h - 网络输入高
    @param {int} orig_w - 原图宽
    @param {int} orig_h - 原图高
    """

    scale: float
    pad_x: int
    pad_y: int
    in_w: int
    in_h: int
    orig_w: int
    orig_h: int


def letterbox(
    image_bgr: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxInfo]:
    """
    将图像等比缩放到 new_shape，并用 padding 补齐（YOLO 风格 letterbox）。

    @param {np.ndarray} image_bgr - 输入 BGR 图像（H,W,3）
    @param {Tuple[int,int]} new_shape - (new_h, new_w)
    @param {Tuple[int,int,int]} color - padding 颜色
    @returns {Tuple[np.ndarray, LetterboxInfo]} - (letterbox 后图像, 变换信息)
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("letterbox: 输入图像为空")

    orig_h, orig_w = image_bgr.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])

    scale = min(new_w / orig_w, new_h / orig_h)
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))

    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    out = np.full((new_h, new_w, 3), color, dtype=resized.dtype)
    out[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

    info = LetterboxInfo(
        scale=float(scale),
        pad_x=int(pad_x),
        pad_y=int(pad_y),
        in_w=int(new_w),
        in_h=int(new_h),
        orig_w=int(orig_w),
        orig_h=int(orig_h),
    )
    return out, info


def orig_xy_to_input_xy(x: float, y: float, info: LetterboxInfo) -> Tuple[float, float]:
    """
    原图坐标 -> 网络输入坐标（letterbox 后的像素坐标）。

    @param {float} x - 原图 x（像素）
    @param {float} y - 原图 y（像素）
    @param {LetterboxInfo} info - letterbox 信息
    @returns {Tuple[float,float]} - (x_in, y_in)
    """

    x_in = x * info.scale + info.pad_x
    y_in = y * info.scale + info.pad_y
    return float(x_in), float(y_in)


def input_xy_to_orig_xy(x: float, y: float, info: LetterboxInfo) -> Tuple[float, float]:
    """
    网络输入坐标 -> 原图坐标（逆 letterbox）。

    @param {float} x - 输入图 x（像素）
    @param {float} y - 输入图 y（像素）
    @param {LetterboxInfo} info - letterbox 信息
    @returns {Tuple[float,float]} - (x_orig, y_orig)
    """

    x0 = (x - info.pad_x) / max(info.scale, 1e-12)
    y0 = (y - info.pad_y) / max(info.scale, 1e-12)
    x0 = float(np.clip(x0, 0, info.orig_w - 1))
    y0 = float(np.clip(y0, 0, info.orig_h - 1))
    return x0, y0


def make_gaussian_heatmap(
    heatmap_h: int,
    heatmap_w: int,
    center_x: float,
    center_y: float,
    sigma: float,
) -> np.ndarray:
    """
    生成以 (center_x, center_y) 为中心的二维高斯 heatmap（值域 0~1）。

    @param {int} heatmap_h - heatmap 高
    @param {int} heatmap_w - heatmap 宽
    @param {float} center_x - 中心 x（heatmap 坐标系，像素）
    @param {float} center_y - 中心 y（heatmap 坐标系，像素）
    @param {float} sigma - 高斯标准差（heatmap 像素）
    @returns {np.ndarray} - (H,W) float32
    """

    yy, xx = np.mgrid[0:heatmap_h, 0:heatmap_w]
    d2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    hm = np.exp(-d2 / (2.0 * (sigma**2) + 1e-12)).astype(np.float32)
    hm = np.clip(hm, 0.0, 1.0)
    return hm


def heatmap_argmax_to_xy(heatmap: np.ndarray) -> Tuple[float, float, float]:
    """
    从 heatmap 中取峰值坐标与峰值置信度。

    @param {np.ndarray} heatmap - (H,W) 或 (1,H,W)
    @returns {Tuple[float,float,float]} - (x, y, conf) in heatmap coords
    """

    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        hm = heatmap[0]
    else:
        hm = heatmap

    if hm.ndim != 2:
        raise ValueError(f"heatmap_argmax_to_xy: 期望二维 heatmap，实际 shape={hm.shape}")

    idx = int(np.argmax(hm))
    y, x = divmod(idx, hm.shape[1])
    conf = float(hm[y, x])
    return float(x), float(y), conf


def bgr_to_rgb_float(image_bgr: np.ndarray) -> np.ndarray:
    """
    BGR uint8 -> RGB float32 [0,1]。

    @param {np.ndarray} image_bgr - BGR 图像
    @returns {np.ndarray} - RGB float32
    """

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def normalize_imagenet(rgb01: np.ndarray) -> np.ndarray:
    """
    ImageNet 均值方差归一化（与 torchvision 默认一致）。

    @param {np.ndarray} rgb01 - RGB float32, range [0,1]
    @returns {np.ndarray} - normalized RGB float32
    """

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = (rgb01 - mean) / std
    return out.astype(np.float32)


def ensure_odd_int(x: float, min_val: int = 1) -> int:
    """
    将数值转为 >=min_val 的奇数整数（用于一些核大小参数）。

    @param {float} x - 输入
    @param {int} min_val - 最小值
    @returns {int} - 奇数整数
    """

    v = int(round(float(x)))
    v = max(int(min_val), v)
    if v % 2 == 0:
        v += 1
    return int(v)


def safe_float(x: Optional[str]) -> Optional[float]:
    """
    安全解析 float（空字符串/None -> None）。

    @param {Optional[str]} x - 输入字符串
    @returns {Optional[float]} - float 或 None
    """

    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


