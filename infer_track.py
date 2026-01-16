"""
方案A推理 + 跟踪入口：
- 从视频/帧目录读取连续帧
- 多帧输入 -> heatmap -> 球心点（峰值）
- 低置信度视为缺测 -> 卡尔曼预测补点
- 导出 tracks.csv + 叠加轨迹可视化

示例：
python -m scheme_a_heatmap_tracker.infer_track \
  --weights /abs/out/best.pt \
  --input /abs/video.mp4 \
  --out_dir /abs/out_vis
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .common import (
    bgr_to_rgb_float,
    heatmap_argmax_to_xy,
    input_xy_to_orig_xy,
    letterbox,
    normalize_imagenet,
)
from .kalman import ConstantVelocityKalman
from .models import EarlyFusionUNet


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="Scheme-A Heatmap Inference + Tracking")
    p.add_argument("--weights", type=str, required=True, help="训练输出的 best.pt/last.pt")
    p.add_argument("--input", type=str, required=True, help="输入视频(.mp4) 或帧目录")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")

    p.add_argument("--window", type=int, default=5, help="多帧窗口长度（应与训练一致）")
    p.add_argument("--img_size", type=int, default=640, help="网络输入尺寸（正方形）")
    p.add_argument("--heatmap_size", type=int, default=160, help="heatmap 尺寸（正方形）")
    p.add_argument("--conf_thresh", type=float, default=0.25, help="低于阈值视为缺测")

    p.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    p.add_argument("--trail", type=int, default=60, help="轨迹尾巴长度（帧）")

    # kalman 参数
    p.add_argument("--process_var", type=float, default=200.0, help="卡尔曼过程噪声强度")
    p.add_argument("--meas_var", type=float, default=100.0, help="卡尔曼观测噪声方差")

    return p.parse_args()


def is_video_file(path: str) -> bool:
    """
    判断是否视频文件。

    @param {str} path - 路径
    @returns {bool} - 是否视频
    """

    exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}
    return Path(path).suffix.lower() in exts


def load_model(weights: str, window: int, heatmap_size: int, device: torch.device) -> EarlyFusionUNet:
    """
    加载模型权重。

    @param {str} weights - checkpoint 路径
    @param {int} window - 窗口长度
    @param {int} heatmap_size - heatmap 尺寸
    @param {torch.device} device - 设备
    @returns {EarlyFusionUNet} - 模型
    """

    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = EarlyFusionUNet(window=window, base_ch=32, out_size=(heatmap_size, heatmap_size))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame_bgr: np.ndarray, img_size: int) -> Tuple[np.ndarray, object]:
    """
    单帧预处理：letterbox + 归一化，输出 CHW float32。

    @param {np.ndarray} frame_bgr - BGR
    @param {int} img_size - 输入尺寸
    @returns {Tuple[np.ndarray, object]} - (chw, letterbox_info)
    """

    lb, info = letterbox(frame_bgr, (img_size, img_size))
    rgb01 = bgr_to_rgb_float(lb)
    rgbn = normalize_imagenet(rgb01)
    chw = np.transpose(rgbn, (2, 0, 1)).astype(np.float32)
    return chw, info


def draw_track(
    frame_bgr: np.ndarray,
    meas_xy: Optional[Tuple[float, float]],
    filt_xy: Tuple[float, float],
    used_meas: bool,
    trail: List[Tuple[float, float]],
) -> np.ndarray:
    """
    绘制当前点与轨迹。

    @param {np.ndarray} frame_bgr - 原图
    @param {Optional[Tuple[float,float]]} meas_xy - 观测点（原图坐标）或 None
    @param {Tuple[float,float]} filt_xy - 滤波点（原图坐标）
    @param {bool} used_meas - 是否使用了观测
    @param {List[Tuple[float,float]]} trail - 轨迹点列表（原图坐标）
    @returns {np.ndarray} - 绘制后的图像
    """

    out = frame_bgr.copy()

    # 轨迹
    for i in range(1, len(trail)):
        x1, y1 = trail[i - 1]
        x2, y2 = trail[i]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)

    fx, fy = filt_xy
    # 滤波点：绿色（用观测）/红色（预测）
    color = (0, 255, 0) if used_meas else (0, 0, 255)
    cv2.circle(out, (int(fx), int(fy)), 5, color, -1)

    if meas_xy is not None:
        mx, my = meas_xy
        cv2.circle(out, (int(mx), int(my)), 4, (255, 255, 255), 2)

    return out


def run_on_video(args: argparse.Namespace) -> None:
    """
    视频推理。

    @param {argparse.Namespace} args - 参数
    @returns {None}
    """

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_model(args.weights, args.window, args.heatmap_size, device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {args.input}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = os.path.join(args.out_dir, "tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # 卡尔曼
    kf = ConstantVelocityKalman(dt=1.0 / max(fps, 1e-6), process_var=args.process_var, meas_var=args.meas_var)

    q: Deque[np.ndarray] = deque(maxlen=args.window)
    infos: Deque[object] = deque(maxlen=args.window)
    trail: Deque[Tuple[float, float]] = deque(maxlen=int(args.trail))

    csv_path = os.path.join(args.out_dir, "tracks.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fcsv:
        writer_csv = csv.writer(fcsv)
        writer_csv.writerow(
            [
                "frame_idx",
                "time_sec",
                "meas_x",
                "meas_y",
                "meas_conf",
                "filt_x",
                "filt_y",
                "used_measurement",
            ]
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            chw, info = preprocess_frame(frame, args.img_size)
            q.append(chw)
            infos.append(info)

            meas_xy = None
            meas_conf = 0.0
            if len(q) == args.window:
                x = np.concatenate(list(q), axis=0)[None, ...]  # (1,C,H,W)
                xt = torch.from_numpy(x).to(device)
                with torch.no_grad():
                    out = model(xt)
                    hm = torch.sigmoid(out.heatmap_logits)[0, 0].detach().cpu().numpy().astype(np.float32)

                hx, hy, conf = heatmap_argmax_to_xy(hm)
                meas_conf = float(conf)
                if meas_conf >= float(args.conf_thresh):
                    # heatmap 坐标 -> 输入图坐标
                    scale = float(args.img_size) / float(args.heatmap_size)
                    x_in = (hx + 0.5) * scale
                    y_in = (hy + 0.5) * scale
                    # 输入图坐标 -> 原图坐标（用最后一帧的 letterbox info）
                    x0, y0 = input_xy_to_orig_xy(x_in, y_in, infos[-1])
                    meas_xy = (float(x0), float(y0))

            filt_x, filt_y, used = kf.step(meas_xy)
            trail.append((float(filt_x), float(filt_y)))

            vis = draw_track(frame, meas_xy, (filt_x, filt_y), used, list(trail))
            writer.write(vis)

            t = frame_idx / max(fps, 1e-6)
            writer_csv.writerow(
                [
                    frame_idx,
                    f"{t:.6f}",
                    "" if meas_xy is None else f"{meas_xy[0]:.3f}",
                    "" if meas_xy is None else f"{meas_xy[1]:.3f}",
                    f"{meas_conf:.6f}",
                    f"{filt_x:.3f}",
                    f"{filt_y:.3f}",
                    int(1 if used else 0),
                ]
            )

            frame_idx += 1
            if total > 0 and frame_idx % 100 == 0:
                print(f"[{frame_idx}/{total}] processing...")

    cap.release()
    writer.release()
    print(f"输出完成：{out_path}")
    print(f"轨迹 CSV：{csv_path}")


def run_on_frames_dir(args: argparse.Namespace) -> None:
    """
    帧目录推理。

    @param {argparse.Namespace} args - 参数
    @returns {None}
    """

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_model(args.weights, args.window, args.heatmap_size, device)

    frames = sorted(
        [p for p in Path(args.input).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}],
        key=lambda p: p.name,
    )
    if len(frames) == 0:
        raise FileNotFoundError(f"帧目录为空或无图片：{args.input}")

    # 读第一帧拿尺寸
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"无法读取第一帧：{frames[0]}")
    h, w = first.shape[:2]

    kf = ConstantVelocityKalman(dt=1.0, process_var=args.process_var, meas_var=args.meas_var)
    q: Deque[np.ndarray] = deque(maxlen=args.window)
    infos: Deque[object] = deque(maxlen=args.window)
    trail: Deque[Tuple[float, float]] = deque(maxlen=int(args.trail))

    csv_path = os.path.join(args.out_dir, "tracks.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fcsv:
        writer_csv = csv.writer(fcsv)
        writer_csv.writerow(
            [
                "frame_name",
                "meas_x",
                "meas_y",
                "meas_conf",
                "filt_x",
                "filt_y",
                "used_measurement",
            ]
        )

        for i, fp in enumerate(frames):
            frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            chw, info = preprocess_frame(frame, args.img_size)
            q.append(chw)
            infos.append(info)

            meas_xy = None
            meas_conf = 0.0
            if len(q) == args.window:
                x = np.concatenate(list(q), axis=0)[None, ...]
                xt = torch.from_numpy(x).to(device)
                with torch.no_grad():
                    out = model(xt)
                    hm = torch.sigmoid(out.heatmap_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
                hx, hy, conf = heatmap_argmax_to_xy(hm)
                meas_conf = float(conf)
                if meas_conf >= float(args.conf_thresh):
                    scale = float(args.img_size) / float(args.heatmap_size)
                    x_in = (hx + 0.5) * scale
                    y_in = (hy + 0.5) * scale
                    x0, y0 = input_xy_to_orig_xy(x_in, y_in, infos[-1])
                    meas_xy = (float(x0), float(y0))

            filt_x, filt_y, used = kf.step(meas_xy)
            trail.append((float(filt_x), float(filt_y)))

            vis = draw_track(frame, meas_xy, (filt_x, filt_y), used, list(trail))
            out_img = os.path.join(args.out_dir, fp.name)
            cv2.imwrite(out_img, vis)

            writer_csv.writerow(
                [
                    fp.name,
                    "" if meas_xy is None else f"{meas_xy[0]:.3f}",
                    "" if meas_xy is None else f"{meas_xy[1]:.3f}",
                    f"{meas_conf:.6f}",
                    f"{filt_x:.3f}",
                    f"{filt_y:.3f}",
                    int(1 if used else 0),
                ]
            )

            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(frames)}] processing...")

    print(f"输出完成：{args.out_dir}")
    print(f"轨迹 CSV：{csv_path}")


def main() -> None:
    """
    主函数。

    @returns {None}
    """

    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入不存在：{args.input}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重不存在：{args.weights}")

    if is_video_file(args.input):
        run_on_video(args)
    else:
        run_on_frames_dir(args)


if __name__ == "__main__":
    main()


