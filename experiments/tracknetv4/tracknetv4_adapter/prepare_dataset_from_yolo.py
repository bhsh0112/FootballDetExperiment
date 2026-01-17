"""
将 YOLO 标注数据集转换为 TrackNetV4 风格的点标注 CSV。

输出 CSV 字段：frame,visibility,x,y
其中 x,y 为像素坐标，visibility 为 0/1。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore

from scheme_a_heatmap_tracker.tools.yolo_to_points import list_images, pick_ball_center, read_yolo_label


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="Prepare TrackNetV4 CSV from YOLO dataset")
    p.add_argument("--data_root", type=str, required=True, help="YOLO 数据集根目录（包含 train/valid/test）")
    p.add_argument("--data_yaml", type=str, default="", help="data.yaml 路径（可选，用于解析类别）")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")
    p.add_argument("--ball_class_id", type=int, default=-1, help="Ball 类别 id（优先级高于 data.yaml）")
    p.add_argument("--ball_class_name", type=str, default="Ball", help="Ball 类别名（用于 data.yaml）")
    p.add_argument(
        "--pick_policy",
        type=str,
        default="smallest",
        choices=["largest", "smallest"],
        help="同图多个 Ball 框选择策略",
    )
    p.add_argument(
        "--image_mode",
        type=str,
        default="none",
        choices=["none", "copy", "symlink"],
        help="是否输出 images（none/copy/symlink）",
    )
    return p.parse_args()


def load_class_id(data_yaml: str, ball_class_name: str) -> Optional[int]:
    """
    从 data.yaml 中读取 Ball 类别 id。

    @param {str} data_yaml - data.yaml 路径
    @param {str} ball_class_name - 类别名
    @returns {Optional[int]} - 类别 id（找不到返回 None）
    """

    if not data_yaml or not os.path.exists(data_yaml):
        return None
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", [])
    if isinstance(names, list):
        for i, name in enumerate(names):
            if str(name).strip().lower() == str(ball_class_name).strip().lower():
                return int(i)
    return None


def ensure_dir(path: str) -> None:
    """
    确保目录存在。

    @param {str} path - 目录路径
    @returns {None}
    """

    os.makedirs(path, exist_ok=True)


def maybe_place_images(
    images: List[Path],
    out_images_dir: str,
    mode: str,
) -> None:
    """
    复制或软链接 images 到输出目录。

    @param {List[Path]} images - 图片路径列表
    @param {str} out_images_dir - 输出 images 目录
    @param {str} mode - none/copy/symlink
    @returns {None}
    """

    if mode == "none":
        return
    ensure_dir(out_images_dir)
    for img_path in tqdm(images, desc=f"images:{os.path.basename(out_images_dir)}"):
        dst = Path(out_images_dir) / img_path.name
        if dst.exists():
            continue
        if mode == "copy":
            dst.write_bytes(img_path.read_bytes())
        else:
            os.symlink(img_path, dst)


def export_split_csv(
    images_dir: str,
    labels_dir: str,
    out_csv: str,
    ball_class_id: int,
    pick_policy: str,
) -> int:
    """
    导出一个 split 的 TrackNetV4 风格 CSV。

    @param {str} images_dir - images 目录
    @param {str} labels_dir - labels 目录
    @param {str} out_csv - 输出 CSV
    @param {int} ball_class_id - Ball 类别 id
    @param {str} pick_policy - largest/smallest
    @returns {int} - 写入行数
    """

    imgs = list_images(images_dir)
    ensure_dir(os.path.dirname(out_csv) or ".")
    rows = 0

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("frame,visibility,x,y\n")
        for img_path in tqdm(imgs, desc=f"csv:{os.path.basename(os.path.dirname(images_dir))}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                f.write(f"{img_path.name},0,,\n")
                continue
            h, wimg = img.shape[:2]
            label_path = os.path.join(labels_dir, f"{img_path.stem}.txt")
            labels = read_yolo_label(label_path)
            xy = pick_ball_center(labels, ball_class_id, wimg, h, pick_policy=pick_policy)
            if xy is None:
                f.write(f"{img_path.name},0,,\n")
            else:
                x, y = xy
                f.write(f"{img_path.name},1,{x:.3f},{y:.3f}\n")
            rows += 1
    return rows


def main() -> None:
    """
    主入口。

    @returns {None}
    """

    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(str(out_dir))

    ball_class_id = args.ball_class_id
    if ball_class_id < 0:
        data_yaml = args.data_yaml or str(data_root / "data.yaml")
        maybe_id = load_class_id(data_yaml, args.ball_class_name)
        ball_class_id = int(maybe_id) if maybe_id is not None else 0

    split_map = {
        "train": ("train", "train"),
        "valid": ("valid", "val"),
        "test": ("test", "test"),
    }

    info: Dict[str, Dict[str, str]] = {}
    for split, (src_name, dst_name) in split_map.items():
        images_dir = data_root / src_name / "images"
        labels_dir = data_root / src_name / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        out_split_dir = out_dir / dst_name
        out_images_dir = out_split_dir / "images"
        out_csv = out_split_dir / "labels.csv"
        ensure_dir(str(out_split_dir))

        export_split_csv(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            out_csv=str(out_csv),
            ball_class_id=ball_class_id,
            pick_policy=args.pick_policy,
        )
        maybe_place_images(list_images(str(images_dir)), str(out_images_dir), args.image_mode)

        info[dst_name] = {
            "images_dir": str(out_images_dir if args.image_mode != "none" else images_dir),
            "labels_csv": str(out_csv),
        }

    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"完成：TrackNetV4 数据已导出到 {out_dir}")


if __name__ == "__main__":
    main()

