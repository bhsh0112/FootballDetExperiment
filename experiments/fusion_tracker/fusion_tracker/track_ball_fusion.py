import argparse
import csv
import os
import sys
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from scheme_a_heatmap_tracker.common import (
        bgr_to_rgb_float,
        heatmap_argmax_to_xy,
        input_xy_to_orig_xy,
        letterbox,
        normalize_imagenet,
    )
    from scheme_a_heatmap_tracker.kalman import ConstantVelocityKalman
    from scheme_a_heatmap_tracker.models import EarlyFusionUNet
    from ttnet_small_object.models import TTNetSmallObject
else:
    from scheme_a_heatmap_tracker.common import (
        bgr_to_rgb_float,
        heatmap_argmax_to_xy,
        input_xy_to_orig_xy,
        letterbox,
        normalize_imagenet,
    )
    from scheme_a_heatmap_tracker.kalman import ConstantVelocityKalman
    from scheme_a_heatmap_tracker.models import EarlyFusionUNet
    from ttnet_small_object.models import TTNetSmallObject


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数对象
    """

    p = argparse.ArgumentParser(description="Heatmap + LK Fusion Tracker")
    p.add_argument("--weights", type=str, required=True, help="TTNet 权重路径")
    p.add_argument("--input", type=str, required=True, help="输入视频(.mp4)")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")
    p.add_argument("--window", type=int, default=5, help="多帧窗口长度")
    p.add_argument("--img_size", type=int, default=640, help="输入尺寸")
    p.add_argument("--heatmap_size", type=int, default=160, help="heatmap 尺寸")
    p.add_argument("--conf_thresh", type=float, default=0.25, help="热力图置信阈值")
    p.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    p.add_argument("--roi_size", type=int, default=28, help="LK 跟踪 ROI 边长")
    p.add_argument("--lk_win", type=int, default=21, help="LK 窗口大小")
    p.add_argument("--lk_levels", type=int, default=3, help="LK 金字塔层数")
    p.add_argument("--lk_iters", type=int, default=20, help="LK 迭代次数")
    p.add_argument("--max_move", type=float, default=50.0, help="最大位移限制")
    p.add_argument("--min_lk_points", type=int, default=6, help="LK 最少有效点数")
    p.add_argument("--process_var", type=float, default=200.0, help="卡尔曼过程噪声强度")
    p.add_argument("--meas_var", type=float, default=120.0, help="卡尔曼观测噪声方差")
    p.add_argument("--fps", type=float, default=20.0, help="输出视频帧率")
    p.add_argument("--make_video", action="store_true", help="输出可视化视频")
    return p.parse_args()


def load_model(
    weights: str, window: int, heatmap_size: int, device: torch.device
) -> Tuple[str, torch.nn.Module, dict]:
    """
    加载模型权重，自动识别模型类型。

    @param {str} weights - 权重路径
    @param {int} window - 窗口长度
    @param {int} heatmap_size - heatmap 尺寸
    @param {torch.device} device - 设备
    @returns {Tuple[str,torch.nn.Module,dict]} - (model_kind, model, ckpt)
    """

    try:
        ckpt = torch.load(weights, map_location="cpu")
    except Exception:
        if hasattr(torch, "serialization"):
            from argparse import Namespace

            with torch.serialization.safe_globals([Namespace]):
                ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        else:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    state_keys = list(state.keys()) if isinstance(state, dict) else []
    if any(k.startswith("global_stage.") for k in state_keys):
        use_local = False
        if isinstance(ckpt, dict) and "args" in ckpt:
            args = ckpt["args"]
            if hasattr(args, "use_local"):
                use_local = bool(args.use_local)
            elif isinstance(args, dict):
                use_local = bool(args.get("use_local", False))
        model = TTNetSmallObject(num_classes=1, input_size=640, use_local=use_local)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return "ttnet", model, ckpt

    model = EarlyFusionUNet(window=window, base_ch=32, out_size=(heatmap_size, heatmap_size))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return "heatmap", model, ckpt


def preprocess_frame(frame_bgr: np.ndarray, img_size: int) -> Tuple[np.ndarray, object]:
    """
    预处理：letterbox + 归一化，输出 CHW float32。

    @param {np.ndarray} frame_bgr - BGR
    @param {int} img_size - 输入尺寸
    @returns {Tuple[np.ndarray, object]} - (chw, letterbox_info)
    """

    lb, info = letterbox(frame_bgr, (img_size, img_size))
    rgb01 = bgr_to_rgb_float(lb)
    rgbn = normalize_imagenet(rgb01)
    chw = np.transpose(rgbn, (2, 0, 1)).astype(np.float32)
    return chw, info


def preprocess_ttnet(frame_bgr: np.ndarray, input_size: int) -> Tuple[torch.Tensor, Tuple[float, float]]:
    """
    TTNet 预处理：resize + 归一化。

    @param {np.ndarray} frame_bgr - BGR
    @param {int} input_size - 输入尺寸
    @returns {Tuple[torch.Tensor,Tuple[float,float]]} - (tensor, scale)
    """

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    scale = (float(w) / float(input_size), float(h) / float(input_size))
    return tensor, scale


def ttnet_best_center(
    predictions: dict, conf_thresh: float, original_size: Tuple[int, int]
) -> Tuple[Optional[Tuple[float, float]], float]:
    """
    从 TTNet 输出中取最可信中心点。

    @param {dict} predictions - TTNet 输出
    @param {float} conf_thresh - 置信阈值
    @param {Tuple[int,int]} original_size - 原图尺寸 (w,h)
    @returns {Tuple[Optional[Tuple[float,float]],float]} - (中心点, 置信度)
    """

    global_pred = predictions.get("global", {})
    cls_output = global_pred.get("cls", None)
    loc_output = global_pred.get("loc", None)
    if cls_output is None:
        return None, 0.0
    cls_prob = torch.sigmoid(cls_output)
    input_h, input_w = cls_prob.shape[-2:]
    kernel_size = 5
    max_pool = torch.nn.functional.max_pool2d(
        cls_prob, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
    )
    mask = (cls_prob == max_pool) & (cls_prob > conf_thresh)
    if mask.sum() == 0:
        return None, 0.0
    y_coords, x_coords = torch.where(mask[0, 0])
    confs = cls_prob[0, 0, y_coords, x_coords]
    best_idx = torch.argmax(confs)
    y = y_coords[best_idx].item()
    x = x_coords[best_idx].item()
    conf = float(confs[best_idx].item())

    grid_x_norm = (x + 0.5) / float(input_w)
    grid_y_norm = (y + 0.5) / float(input_h)
    x_center_norm = grid_x_norm
    y_center_norm = grid_y_norm
    if loc_output is not None and loc_output.shape[1] >= 2:
        loc_x = loc_output[0, 0, int(y), int(x)].item()
        loc_y = loc_output[0, 1, int(y), int(x)].item()
        cand_raw_x = max(0.0, min(1.0, loc_x))
        cand_raw_y = max(0.0, min(1.0, loc_y))
        if conf > 1e-6:
            cand_div_x = max(0.0, min(1.0, loc_x / conf))
            cand_div_y = max(0.0, min(1.0, loc_y / conf))
        else:
            cand_div_x, cand_div_y = cand_raw_x, cand_raw_y
        dist_raw = abs(cand_raw_x - grid_x_norm) + abs(cand_raw_y - grid_y_norm)
        dist_div = abs(cand_div_x - grid_x_norm) + abs(cand_div_y - grid_y_norm)
        best_x, best_y = (cand_raw_x, cand_raw_y) if dist_raw <= dist_div else (cand_div_x, cand_div_y)
        if min(dist_raw, dist_div) <= 0.1:
            x_center_norm, y_center_norm = best_x, best_y

    x_center = x_center_norm * original_size[0]
    y_center = y_center_norm * original_size[1]
    return (float(x_center), float(y_center)), conf


def to_bbox(center: Tuple[float, float], size: int) -> Tuple[int, int, int, int]:
    """
    中心点转 ROI bbox。

    @param {Tuple[float,float]} center - 中心点
    @param {int} size - ROI 边长
    @returns {Tuple[int,int,int,int]} - (x,y,w,h)
    """

    cx, cy = center
    half = size / 2.0
    x = int(round(cx - half))
    y = int(round(cy - half))
    return x, y, int(round(size)), int(round(size))


def clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    """
    裁剪 ROI 到图像边界内。

    @param {Tuple[int,int,int,int]} bbox - 原 bbox
    @param {int} w - 图像宽
    @param {int} h - 图像高
    @returns {Tuple[int,int,int,int]} - 裁剪后的 bbox
    """

    x, y, bw, bh = bbox
    x0 = max(0, min(w - 1, x))
    y0 = max(0, min(h - 1, y))
    x1 = max(1, min(w, x + bw))
    y1 = max(1, min(h, y + bh))
    return x0, y0, x1 - x0, y1 - y0


def sample_features(gray: np.ndarray, bbox: Tuple[int, int, int, int], max_points: int) -> np.ndarray:
    """
    ROI 内采样角点。

    @param {np.ndarray} gray - 灰度图
    @param {Tuple[int,int,int,int]} bbox - ROI
    @param {int} max_points - 最大点数
    @returns {np.ndarray} - 角点集合 (N,1,2)
    """

    x, y, w, h = bbox
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=3,
        useHarrisDetector=False,
    )
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    points[:, 0, 0] += x
    points[:, 0, 1] += y
    return points.astype(np.float32)


def lk_track(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_points: np.ndarray,
    prev_center: Tuple[float, float],
    lk_win: int,
    lk_levels: int,
    lk_iters: int,
    max_move: float,
    min_points: int,
) -> Tuple[Optional[Tuple[float, float]], np.ndarray]:
    """
    基于 LK 的单步跟踪。

    @param {np.ndarray} prev_gray - 上一帧灰度图
    @param {np.ndarray} gray - 当前帧灰度图
    @param {np.ndarray} prev_points - 上一帧角点
    @param {Tuple[float,float]} prev_center - 上一帧中心
    @param {int} lk_win - LK 窗口
    @param {int} lk_levels - 金字塔层
    @param {int} lk_iters - 迭代次数
    @param {float} max_move - 最大位移
    @param {int} min_points - 最少有效点
    @returns {Tuple[Optional[Tuple[float,float]],np.ndarray]} - (新中心, 新角点)
    """

    if prev_points.size == 0:
        return None, prev_points
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(lk_win, lk_win),
        maxLevel=lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_iters, 0.03),
    )
    if p1 is None or st is None:
        return None, prev_points
    good_new = p1[st[:, 0] == 1]
    good_old = prev_points[st[:, 0] == 1]
    if good_new.size < min_points:
        return None, prev_points
    shifts = good_new - good_old
    median_shift = np.median(shifts.reshape(-1, 2), axis=0)
    cand = (prev_center[0] + float(median_shift[0]), prev_center[1] + float(median_shift[1]))
    if max_move > 0 and np.hypot(cand[0] - prev_center[0], cand[1] - prev_center[1]) > max_move:
        return None, prev_points
    return cand, good_new.reshape(-1, 1, 2)


def run(args: argparse.Namespace) -> None:
    """
    融合推理入口。

    @param {argparse.Namespace} args - 参数
    @returns {None}
    """

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames_annotated")
    os.makedirs(frames_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_kind, model, _ = load_model(args.weights, args.window, args.heatmap_size, device)
    kf = ConstantVelocityKalman(dt=1.0, process_var=args.process_var, meas_var=args.meas_var)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = float(args.fps)

    writer = None
    if args.make_video:
        out_path = os.path.join(args.out_dir, "tracked.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    csv_path = os.path.join(args.out_dir, "tracks.csv")
    fcsv = open(csv_path, "w", encoding="utf-8", newline="")
    writer_csv = csv.writer(fcsv)
    writer_csv.writerow(
        [
            "frame_idx",
            "det_x",
            "det_y",
            "det_conf",
            "lk_x",
            "lk_y",
            "filt_x",
            "filt_y",
            "used_meas",
            "method",
            "model_kind",
        ]
    )

    q: Deque[np.ndarray] = deque(maxlen=args.window)
    infos: Deque[object] = deque(maxlen=args.window)

    prev_gray = None
    prev_points = np.empty((0, 1, 2), dtype=np.float32)
    lk_center: Optional[Tuple[float, float]] = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det_xy = None
        det_conf = 0.0
        if model_kind == "heatmap":
            chw, info = preprocess_frame(frame, args.img_size)
            q.append(chw)
            infos.append(info)
            if len(q) == args.window:
                x = np.concatenate(list(q), axis=0)[None, ...]
                xt = torch.from_numpy(x).to(device)
                with torch.no_grad():
                    out = model(xt)
                    hm = torch.sigmoid(out.heatmap_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
                hx, hy, conf = heatmap_argmax_to_xy(hm)
                det_conf = float(conf)
                scale = float(args.img_size) / float(args.heatmap_size)
                x_in = (hx + 0.5) * scale
                y_in = (hy + 0.5) * scale
                x0, y0 = input_xy_to_orig_xy(x_in, y_in, infos[-1])
                det_xy = (float(x0), float(y0))
        else:
            tensor, _ = preprocess_ttnet(frame, args.img_size)
            tensor = tensor.to(device)
            with torch.no_grad():
                out = model(tensor)
            det_xy, det_conf = ttnet_best_center(
                out, float(args.conf_thresh), (frame.shape[1], frame.shape[0])
            )

        lk_xy = None
        if prev_gray is not None and lk_center is not None:
            lk_xy, prev_points = lk_track(
                prev_gray,
                gray,
                prev_points,
                lk_center,
                args.lk_win,
                args.lk_levels,
                args.lk_iters,
                args.max_move,
                args.min_lk_points,
            )
        if det_xy is not None and det_conf >= float(args.conf_thresh):
            lk_center = det_xy
            bbox = to_bbox(det_xy, args.roi_size)
            bbox = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
            prev_points = sample_features(gray, bbox, max_points=max(8, args.min_lk_points * 2))
            measurement = det_xy
            method = "detect"
        elif lk_xy is not None:
            lk_center = lk_xy
            measurement = lk_xy
            method = "lk"
        else:
            measurement = None
            method = "predict"

        filt_x, filt_y, used = kf.step(measurement)

        vis = frame.copy()
        if det_xy is not None:
            cv2.circle(vis, (int(det_xy[0]), int(det_xy[1])), 6, (255, 0, 0), 2)
        if lk_xy is not None:
            cv2.circle(vis, (int(lk_xy[0]), int(lk_xy[1])), 5, (0, 255, 255), 1)
        color = (0, 255, 0) if used else (0, 0, 255)
        cv2.circle(vis, (int(filt_x), int(filt_y)), 5, color, -1)
        cv2.putText(
            vis,
            f"{method} {det_conf:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out_img = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_img, vis)
        if writer is not None:
            writer.write(vis)

        writer_csv.writerow(
            [
                frame_idx,
                "" if det_xy is None else f"{det_xy[0]:.2f}",
                "" if det_xy is None else f"{det_xy[1]:.2f}",
                f"{det_conf:.4f}",
                "" if lk_xy is None else f"{lk_xy[0]:.2f}",
                "" if lk_xy is None else f"{lk_xy[1]:.2f}",
                f"{filt_x:.2f}",
                f"{filt_y:.2f}",
                int(used),
                method,
                model_kind,
            ]
        )

        prev_gray = gray
        frame_idx += 1

    if writer is not None:
        writer.release()
    fcsv.close()
    cap.release()


if __name__ == "__main__":
    run(parse_args())

