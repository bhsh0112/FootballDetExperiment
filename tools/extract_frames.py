"""
从视频抽帧到图片目录。

示例：
python -m scheme_a_heatmap_tracker.tools.extract_frames \
  --video /abs/video.mp4 \
  --out_dir /abs/frames \
  --every 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="Extract frames from video")
    p.add_argument("--video", type=str, required=True, help="输入视频路径")
    p.add_argument("--out_dir", type=str, required=True, help="输出帧目录")
    p.add_argument("--every", type=int, default=1, help="每 N 帧保存 1 帧（默认每帧）")
    p.add_argument("--prefix", type=str, default="frame_", help="输出文件名前缀")
    p.add_argument("--ext", type=str, default=".jpg", help="输出扩展名：.jpg/.png")
    return p.parse_args()


def main() -> None:
    """
    主函数。

    @returns {None}
    """

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every = max(1, int(args.every))
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"

    idx = 0
    saved = 0
    pbar = tqdm(total=total if total > 0 else None, desc="extract")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every == 0:
            name = f"{args.prefix}{idx:06d}{ext}"
            out_path = str(Path(args.out_dir) / name)
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"完成：读取 {idx} 帧，保存 {saved} 帧到 {args.out_dir}")


if __name__ == "__main__":
    main()


