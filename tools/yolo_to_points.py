"""
将 YOLO bbox 标签转换为“球心点标注 CSV”（frame,x,y,visible）。

说明：
- 输入是静态图片数据集（如 train/images + train/labels）
- 对每张图片：若标签里存在 ball_class_id，则取该类的一个实例作为球
  - 默认策略：取面积最大的 bbox（更鲁棒，避免把噪声小框当球）
- 输出 CSV：frame 使用图片文件名（basename）

示例：
python -m scheme_a_heatmap_tracker.tools.yolo_to_points \
  --images_dir /home/buaa/football_detect/train/images \
  --labels_dir /home/buaa/football_detect/train/labels \
  --out_csv /home/buaa/football_detect/train_ball_points.csv \
  --ball_class_id 0
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # type: ignore
from tqdm import tqdm  # type: ignore


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="Convert YOLO labels to ball center CSV")
    p.add_argument("--images_dir", type=str, required=True, help="images 目录")
    p.add_argument("--labels_dir", type=str, required=True, help="labels 目录（YOLO txt）")
    p.add_argument("--out_csv", type=str, required=True, help="输出 CSV 路径")
    p.add_argument("--ball_class_id", type=int, default=0, help="Ball 类别 id（默认 0）")
    p.add_argument(
        "--pick_policy",
        type=str,
        default="largest",
        choices=["largest", "smallest"],
        help="同一张图出现多个 Ball 框时的选择策略：largest/smallest（默认 largest）",
    )
    return p.parse_args()


def list_images(images_dir: str) -> List[Path]:
    """
    列出 images_dir 下的图片。

    @param {str} images_dir - 目录
    @returns {List[Path]} - 图片路径列表
    """

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in Path(images_dir).iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(imgs, key=lambda p: p.name)


def read_yolo_label(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    读取一张图的 YOLO 标签。

    @param {str} label_path - 标签路径
    @returns {List[Tuple[int,float,float,float,float]]} - (cls,xc,yc,w,h) 均为归一化
    """

    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            out.append((cls, xc, yc, w, h))
    return out


def pick_ball_center(
    labels: List[Tuple[int, float, float, float, float]],
    ball_class_id: int,
    img_w: int,
    img_h: int,
    pick_policy: str = "largest",
) -> Optional[Tuple[float, float]]:
    """
    从 YOLO 标签中选择一个球心点。

    策略：筛出 ball_class_id，再按 pick_policy 选择 bbox。

    @param {List[Tuple]} labels - 标签
    @param {int} ball_class_id - ball 类别 id
    @param {int} img_w - 图像宽
    @param {int} img_h - 图像高
    @param {str} pick_policy - largest 或 smallest
    @returns {Optional[Tuple[float,float]]} - (x,y) 像素坐标或 None
    """

    candidates = [(xc, yc, w, h) for (cls, xc, yc, w, h) in labels if cls == int(ball_class_id)]
    if len(candidates) == 0:
        return None
    # 选择 bbox
    policy = str(pick_policy).strip().lower()
    if policy == "smallest":
        xc, yc, w, h = min(candidates, key=lambda t: t[2] * t[3])
    else:
        xc, yc, w, h = max(candidates, key=lambda t: t[2] * t[3])
    x = float(xc) * float(img_w)
    y = float(yc) * float(img_h)
    return x, y


def main() -> None:
    """
    主函数。

    @returns {None}
    """

    args = parse_args()
    imgs = list_images(args.images_dir)
    if len(imgs) == 0:
        raise FileNotFoundError(f"images_dir 下没有图片：{args.images_dir}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "visible"])

        for img_path in tqdm(imgs, desc="convert"):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, wimg = img.shape[:2]

            label_path = str(Path(args.labels_dir) / f"{img_path.stem}.txt")
            labels = read_yolo_label(label_path)
            xy = pick_ball_center(labels, args.ball_class_id, wimg, h, pick_policy=args.pick_policy)
            if xy is None:
                w.writerow([img_path.name, "", "", 0])
            else:
                x, y = xy
                w.writerow([img_path.name, f"{x:.3f}", f"{y:.3f}", 1])

    print(f"完成：输出 {args.out_csv}")


if __name__ == "__main__":
    main()


