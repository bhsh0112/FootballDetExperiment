#!/usr/bin/env python3
"""
从视频中抽取关键帧（用于推理/可视化测试）。

思路：
1) 低频采样读取视频帧，计算相邻采样帧的“变化强度”分数（灰度缩放后做像素绝对差均值）。
2) 从高分帧中做间隔约束选择，避免关键帧集中在同一小段。
3) 若不足指定数量，则用均匀时间采样补齐。

用法示例：
python3 tools/extract_keyframes.py \
  --video /home/buaa/football_detect/test_game/0108.mp4 \
  --out_dir /home/buaa/football_detect/keyframes/0108 \
  --num 5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    """关键帧候选信息。"""

    frame_idx: int
    score: float


def _safe_makedirs(path: str) -> None:
    """创建输出目录（若已存在则忽略）。"""

    os.makedirs(path, exist_ok=True)


def _video_meta(cap: cv2.VideoCapture) -> Tuple[float, int, int, float]:
    """读取视频元信息：fps、总帧数、宽高、时长（秒）。"""

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
    return fps, frame_count, max(width, height), duration_s


def _diff_score(prev_small_gray: np.ndarray, curr_small_gray: np.ndarray) -> float:
    """计算两帧之间的变化强度分数（像素绝对差均值）。"""

    return float(np.mean(np.abs(curr_small_gray.astype(np.int16) - prev_small_gray.astype(np.int16))))


def _iter_candidates(
    video_path: str,
    sample_fps: float,
    resize_w: int,
    resize_h: int,
) -> Tuple[List[Candidate], float, int]:
    """
    扫描视频得到候选关键帧列表。

    返回：
    - candidates: 每个候选的 (frame_idx, score)
    - fps: 原视频 fps
    - frame_count: 总帧数
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        # 常见于某些编码/容器；回退一个合理值，至少保证 stride 可用
        fps = 25.0

    # 采样步长（按帧）
    stride = max(1, int(round(fps / max(sample_fps, 0.1))))

    candidates: List[Candidate] = []
    prev_gray: np.ndarray | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        small = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            score = _diff_score(prev_gray, gray)
            candidates.append(Candidate(frame_idx=frame_idx, score=score))

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return candidates, fps, frame_count


def _select_top_with_gap(
    candidates: List[Candidate],
    num: int,
    min_gap_frames: int,
) -> List[int]:
    """按分数从高到低选取，且保证所选帧之间的最小间隔。"""

    if not candidates or num <= 0:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: List[int] = []
    for c in sorted_cands:
        if len(selected) >= num:
            break
        if all(abs(c.frame_idx - s) >= min_gap_frames for s in selected):
            selected.append(c.frame_idx)

    selected.sort()
    return selected


def _uniform_frame_indices(frame_count: int, num: int) -> List[int]:
    """均匀采样得到帧索引（尽量避开 0 帧）。"""

    if frame_count <= 0 or num <= 0:
        return []
    if num == 1:
        return [max(0, frame_count // 2)]
    # 从 5% 到 95% 均匀取点，减少黑屏/片头片尾概率
    start = int(frame_count * 0.05)
    end = max(start + 1, int(frame_count * 0.95))
    idxs = np.linspace(start, end, num=num, dtype=np.int64).tolist()
    return [int(i) for i in idxs]


def _write_frames(video_path: str, frame_indices: List[int], out_dir: str, prefix: str) -> List[str]:
    """根据帧索引导出图片文件，返回输出路径列表。"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    out_paths: List[str] = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue

        out_path = os.path.join(out_dir, f"{prefix}_{i+1:02d}_frame{frame_idx:07d}.jpg")
        # JPEG 质量设置稍高，利于检测
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        out_paths.append(out_path)

    cap.release()
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="从视频中抽取指定数量的关键帧。")
    parser.add_argument("--video", required=True, help="视频路径（绝对路径建议）。")
    parser.add_argument("--out_dir", required=True, help="输出目录（会自动创建）。")
    parser.add_argument("--num", type=int, default=5, help="关键帧数量，默认 5。")
    parser.add_argument("--sample_fps", type=float, default=2.0, help="扫描阶段采样帧率（越大越慢但更精细），默认 2。")
    parser.add_argument("--min_gap_s", type=float, default=2.0, help="关键帧最小时间间隔（秒），默认 2 秒。")
    parser.add_argument("--resize_w", type=int, default=160, help="扫描阶段缩放宽度，默认 160。")
    parser.add_argument("--resize_h", type=int, default=90, help="扫描阶段缩放高度，默认 90。")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    out_dir = os.path.abspath(args.out_dir)
    _safe_makedirs(out_dir)

    candidates, fps, frame_count = _iter_candidates(
        video_path=video_path,
        sample_fps=args.sample_fps,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
    )

    min_gap_frames = max(1, int(round(max(args.min_gap_s, 0.0) * fps)))
    selected = _select_top_with_gap(candidates=candidates, num=args.num, min_gap_frames=min_gap_frames)

    # 不足则用均匀采样补齐（并去重）
    if len(selected) < args.num:
        uniform = _uniform_frame_indices(frame_count=frame_count, num=args.num)
        merged = list(dict.fromkeys(selected + uniform))  # 保序去重
        merged = sorted(set(merged))  # 再排序更稳
        # 若超出则再按均匀分布抽回 num 个
        if len(merged) > args.num:
            # 重新均匀选取 merged 的索引
            pick_pos = np.linspace(0, len(merged) - 1, num=args.num, dtype=np.int64).tolist()
            selected = [merged[int(p)] for p in pick_pos]
        else:
            selected = merged

    prefix = os.path.splitext(os.path.basename(video_path))[0]
    out_paths = _write_frames(video_path=video_path, frame_indices=selected, out_dir=out_dir, prefix=prefix)

    print(f"video={video_path}")
    print(f"fps={fps:.3f} frame_count={frame_count}")
    print(f"selected_frames={selected}")
    print("outputs:")
    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()


