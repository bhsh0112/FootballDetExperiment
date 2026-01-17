"""
YOLOv13 足球（Ball）单类检测训练脚本

说明：
- 你的原始数据集 `data.yaml` 是 3 类：Ball / Player / Ref。
- 本脚本会在本地自动生成一个“只保留 Ball 标注”的临时数据集目录（不会改动原始数据）。
- 然后使用 YOLOv13（来自 iMoonLab/yolov13 的 ultralytics 分支）进行训练。
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from ultralytics import YOLO


@dataclass(frozen=True)
class DatasetSplit:
    """数据集划分路径结构。"""

    images_dir: Path
    labels_dir: Path


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Train YOLOv13 for soccer ball detection (single-class).")

    parser.add_argument("--data", type=str, default="data.yaml", help="原始数据集 data.yaml（多类也可以）")
    parser.add_argument(
        "--out-dataset",
        type=str,
        default=str(Path(__file__).resolve().parent / ".dataset_ball_only"),
        help="输出的临时单类数据集根目录（会创建/复用）",
    )
    parser.add_argument(
        "--ball-class-id",
        type=int,
        default=0,
        help="原始标签里 Ball 的类别 id（你的 data.yaml 里 Ball 是 0）",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov13n.pt",
        help="YOLOv13 模型：yolov13n.pt / yolov13s.pt / yolov13l.pt / yolov13x.pt 或 yolov13n.yaml 等",
    )

    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size（按显存调整）")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--device", type=str, default="", help="训练设备，如 '0' / '0,1' / 'cpu'；默认自动选择")

    parser.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "detect"),
        help="runs 保存目录",
    )
    parser.add_argument("--name", type=str, default="yolov13_ball", help="实验名")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名实验目录")

    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--amp", action="store_true", help="启用 AMP（默认关闭，避免部分环境不兼容）")

    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="强制重建临时单类数据集（会先删除 --out-dataset）",
    )

    return parser.parse_args()


def _read_yaml(path: Path) -> Dict:
    """读取 YAML 文件。"""

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, obj: Dict) -> None:
    """写入 YAML 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _auto_device(device_arg: str) -> str:
    """自动选择训练 device。"""

    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_paths(data_yaml_path: Path, data_obj: Dict) -> Tuple[Path, Dict[str, Path]]:
    """
    解析 ultralytics data.yaml 的路径字段。

    支持：
    - train/val/test 是相对路径（相对于 data.yaml 所在目录）
    - train/val/test 是绝对路径
    """

    base_dir = data_yaml_path.parent
    split_keys = ["train", "val", "test"]
    resolved: Dict[str, Path] = {}

    for k in split_keys:
        if k not in data_obj or data_obj[k] is None:
            continue
        p = Path(str(data_obj[k]))
        resolved[k] = p if p.is_absolute() else (base_dir / p)

    return base_dir, resolved


def _images_to_labels_dir(images_dir: Path) -> Path:
    """
    从 images 目录推导 labels 目录：
    - .../train/images  -> .../train/labels
    - .../valid/images  -> .../valid/labels
    """

    if images_dir.name != "images":
        # 尽量兼容，但默认还是用父目录下的 labels
        return images_dir.parent / "labels"
    return images_dir.parent / "labels"


def _iter_label_files(labels_dir: Path) -> List[Path]:
    """获取 labels 目录下所有 .txt 标签文件。"""

    if not labels_dir.exists():
        return []
    return sorted(labels_dir.rglob("*.txt"))


def _filter_one_label_file(src_txt: Path, dst_txt: Path, keep_class_id: int) -> None:
    """
    过滤单个 YOLO 标签文件，只保留指定类别。

    YOLO label 行格式：<cls> <x> <y> <w> <h> （可带置信度/track id 等扩展，本脚本只处理前 5 列）
    """

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    kept_lines: List[str] = []

    with src_txt.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            if cls_id != keep_class_id:
                continue

            # 单类数据集：类别 id 统一写成 0
            parts[0] = "0"
            kept_lines.append(" ".join(parts))

    with dst_txt.open("w", encoding="utf-8") as f:
        if kept_lines:
            f.write("\n".join(kept_lines) + "\n")
        else:
            # 写空文件也 OK：表示该图片没有 ball 标注
            f.write("")


def _symlink_or_copy_dir(src: Path, dst: Path) -> None:
    """优先创建目录软链接；若失败则复制（尽量不占用太多空间）。"""

    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.as_posix(), dst.as_posix(), target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def _prepare_ball_only_dataset(
    data_yaml_path: Path,
    out_root: Path,
    ball_class_id: int,
    rebuild: bool,
) -> Path:
    """
    生成一个只包含 Ball 的临时数据集目录，并返回新 data.yaml 的路径。
    """

    if rebuild and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    data_obj = _read_yaml(data_yaml_path)
    _, split_images_dirs = _resolve_dataset_paths(data_yaml_path, data_obj)

    # 生成临时数据集结构：out_root/{train,valid,test}/{images,labels}
    # 注意：你的原始结构是 train/images, valid/images, test/images
    split_map = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    for k, out_split_name in split_map.items():
        if k not in split_images_dirs:
            continue
        images_dir = split_images_dirs[k]
        labels_dir = _images_to_labels_dir(images_dir)

        out_images = out_root / out_split_name / "images"
        out_labels = out_root / out_split_name / "labels"

        _symlink_or_copy_dir(images_dir, out_images)
        out_labels.mkdir(parents=True, exist_ok=True)

        for src_txt in _iter_label_files(labels_dir):
            rel = src_txt.relative_to(labels_dir)
            dst_txt = out_labels / rel
            _filter_one_label_file(src_txt, dst_txt, keep_class_id=ball_class_id)

    # 写新的 data.yaml（单类）
    out_data = {
        "path": str(out_root.as_posix()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["Ball"],
    }
    out_yaml_path = out_root / "data_ball.yaml"
    _write_yaml(out_yaml_path, out_data)
    return out_yaml_path


def main() -> None:
    """入口函数。"""

    args = _parse_args()

    data_yaml_path = Path(args.data).expanduser().resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"找不到 data.yaml: {data_yaml_path}")

    out_root = Path(args.out_dataset).expanduser().resolve()

    print(f"[INFO] 原始 data.yaml: {data_yaml_path}")
    print(f"[INFO] 临时单类数据集目录: {out_root}")
    print(f"[INFO] 仅保留 Ball 类别 id: {args.ball_class_id}")

    ball_data_yaml = _prepare_ball_only_dataset(
        data_yaml_path=data_yaml_path,
        out_root=out_root,
        ball_class_id=args.ball_class_id,
        rebuild=args.rebuild_dataset,
    )

    device = _auto_device(args.device)
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 训练 data.yaml(单类): {ball_data_yaml}")
    print(f"[INFO] YOLOv13 模型: {args.model}")

    model = YOLO(args.model)

    results = model.train(
        data=str(ball_data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=device,
        workers=int(args.workers),
        project=str(args.project),
        name=str(args.name),
        exist_ok=bool(args.exist_ok),
        pretrained=True,
        single_cls=True,  # 单类训练
        seed=int(args.seed),
        deterministic=True,
        amp=bool(args.amp),
        close_mosaic=10,
    )

    print("\n[INFO] 训练完成，开始验证...")
    metrics = model.val(data=str(ball_data_yaml), device=device)
    print(f"[INFO] 验证完成: {metrics}")
    print(f"[INFO] best 权重: {getattr(model.trainer, 'best', '(unknown)')}")
    print(f"[INFO] 保存目录: {Path(args.project) / args.name}")

    _ = results  # 避免部分环境下未使用变量告警


if __name__ == "__main__":
    main()


