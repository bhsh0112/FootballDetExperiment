"""
从 Roboflow YOLO 数据集（如本仓库的 data/）一键完成：
1) YOLO bbox -> 球心点 CSV（train/valid）
2) 训练 heatmap 网络（默认 window=1，适合非连续图片数据）

示例：
conda run -n yolo python -m scheme_a_heatmap_tracker.train_from_data_yolo \
  --data_root /home/buaa/football_detect/data \
  --out_dir /home/buaa/football_detect/runs/scheme_a/run1 \
  --epochs 50
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore

"""
兼容两种启动方式：
- 推荐：在仓库根目录执行 `python -m scheme_a_heatmap_tracker.train_from_data_yolo ...`
- 兼容：在本文件所在目录执行 `python train_from_data_yolo.py ...`
"""

if __package__ is None or __package__ == "":
    # 直接运行脚本时，确保能找到顶层包 scheme_a_heatmap_tracker
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from scheme_a_heatmap_tracker.datasets import SequencePointDataset
    from scheme_a_heatmap_tracker.models import EarlyFusionUNet
    from scheme_a_heatmap_tracker.tools.yolo_to_points import list_images, pick_ball_center, read_yolo_label
else:
    from .datasets import SequencePointDataset
    from .models import EarlyFusionUNet
    from .tools.yolo_to_points import list_images, pick_ball_center, read_yolo_label


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="Train scheme-A from data/ YOLO dataset")
    p.add_argument("--data_root", type=str, required=True, help="包含 train/valid/test 的数据根目录（如 data/）")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")

    p.add_argument("--ball_class_id", type=int, default=0, help="Ball 类别 id（默认 0）")
    p.add_argument(
        "--pick_policy",
        type=str,
        default="smallest",
        choices=["largest", "smallest"],
        help="同图多个 Ball 框的选择策略（默认 smallest 更符合小球）",
    )

    # 训练参数
    p.add_argument("--window", type=int, default=1, help="多帧窗口长度（此数据通常非连续，建议 1）")
    p.add_argument("--img_size", type=int, default=640, help="网络输入尺寸（正方形）")
    p.add_argument("--heatmap_size", type=int, default=160, help="heatmap 尺寸（正方形）")
    p.add_argument("--base_ch", type=int, default=32, help="UNet 基础通道数（越小越省显存：16/24/32）")
    p.add_argument("--sigma", type=float, default=2.0, help="heatmap 高斯 sigma（heatmap 像素）")
    p.add_argument("--lambda_no_ball", type=float, default=0.05, help="不可见帧 loss 权重")

    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=16, help="batch size（单帧训练可适当调大）")
    p.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数（=1表示不累积）")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    p.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    p.add_argument("--amp", action="store_true", help="启用 AMP 混合精度（显存更省，推荐）")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return p.parse_args()


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor, loss_w: torch.Tensor) -> torch.Tensor:
    """
    heatmap loss（MSE on sigmoid）。

    @param {torch.Tensor} logits - (B,1,Hm,Wm)
    @param {torch.Tensor} target - (B,1,Hm,Wm)
    @param {torch.Tensor} loss_w - (B,1)
    @returns {torch.Tensor} - scalar
    """

    pred = torch.sigmoid(logits)
    per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    w = loss_w.view(-1).to(per.dtype)
    return (per * w).mean()


def export_points_csv(
    images_dir: str,
    labels_dir: str,
    out_csv: str,
    ball_class_id: int,
    pick_policy: str,
) -> None:
    """
    将一个 split 的 YOLO 标签导出为点标注 CSV。

    @param {str} images_dir - images 目录
    @param {str} labels_dir - labels 目录
    @param {str} out_csv - 输出 CSV
    @param {int} ball_class_id - ball 类别 id
    @param {str} pick_policy - largest/smallest
    @returns {None}
    """

    import csv
    import cv2  # type: ignore

    imgs = list_images(images_dir)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "visible"])
        for img_path in tqdm(imgs, desc=f"points:{os.path.basename(os.path.dirname(images_dir))}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                w.writerow([img_path.name, "", "", 0])
                continue
            h, wimg = img.shape[:2]
            label_path = os.path.join(labels_dir, f"{img_path.stem}.txt")
            labels = read_yolo_label(label_path)
            xy = pick_ball_center(labels, ball_class_id, wimg, h, pick_policy=pick_policy)
            if xy is None:
                w.writerow([img_path.name, "", "", 0])
            else:
                x, y = xy
                w.writerow([img_path.name, f"{x:.3f}", f"{y:.3f}", 1])


def save_checkpoint(path: str, model: torch.nn.Module, meta: dict) -> None:
    """
    保存 checkpoint。

    @param {str} path - 路径
    @param {torch.nn.Module} model - 模型
    @param {dict} meta - 元信息
    @returns {None}
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def main() -> None:
    """
    主函数。

    @returns {None}
    """

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_images = os.path.join(args.data_root, "train", "images")
    train_labels = os.path.join(args.data_root, "train", "labels")
    val_images = os.path.join(args.data_root, "valid", "images")
    val_labels = os.path.join(args.data_root, "valid", "labels")

    if not os.path.isdir(train_images) or not os.path.isdir(train_labels):
        raise FileNotFoundError(f"缺少 train/images 或 train/labels：{args.data_root}")
    if not os.path.isdir(val_images) or not os.path.isdir(val_labels):
        raise FileNotFoundError(f"缺少 valid/images 或 valid/labels：{args.data_root}")

    # 1) 导出点标注 CSV
    train_csv = os.path.join(args.out_dir, "train_points.csv")
    val_csv = os.path.join(args.out_dir, "valid_points.csv")
    export_points_csv(train_images, train_labels, train_csv, args.ball_class_id, args.pick_policy)
    export_points_csv(val_images, val_labels, val_csv, args.ball_class_id, args.pick_policy)

    # 2) 构造数据集
    train_ds = SequencePointDataset(
        frames_dir=train_images,
        points_csv=train_csv,
        window=args.window,
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        lambda_no_ball=args.lambda_no_ball,
    )
    val_ds = SequencePointDataset(
        frames_dir=val_images,
        points_csv=val_csv,
        window=args.window,
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        lambda_no_ball=args.lambda_no_ball,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = EarlyFusionUNet(
        window=args.window,
        base_ch=int(args.base_ch),
        out_size=(args.heatmap_size, args.heatmap_size),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    meta = {
        "data_root": args.data_root,
        "ball_class_id": int(args.ball_class_id),
        "pick_policy": str(args.pick_policy),
        "window": int(args.window),
        "img_size": int(args.img_size),
        "heatmap_size": int(args.heatmap_size),
        "base_ch": int(args.base_ch),
        "sigma": float(args.sigma),
        "lambda_no_ball": float(args.lambda_no_ball),
        "amp": bool(args.amp),
        "grad_accum": int(args.grad_accum),
    }

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_losses = []
        opt.zero_grad(set_to_none=True)
        for batch in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            lw = batch["loss_w"].to(device, non_blocking=True)
            accum = max(1, int(args.grad_accum))
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x)
                loss = heatmap_loss(out.heatmap_logits, y, lw) / float(accum)
            scaler.scale(loss).backward()
            tr_losses.append(float(loss.item() * float(accum)))

            if (len(tr_losses) % accum) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                lw = batch["loss_w"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(x)
                    loss = heatmap_loss(out.heatmap_logits, y, lw)
                v_losses.append(float(loss.item()))

        tr = float(np.mean(tr_losses)) if tr_losses else 0.0
        va = float(np.mean(v_losses)) if v_losses else float("inf")
        print(f"[epoch {epoch}] train_loss={tr:.6f} val_loss={va:.6f} best_val={best_val:.6f}")

        save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, {**meta, "epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val:
            best_val = va
            save_checkpoint(os.path.join(args.out_dir, "best.pt"), model, {**meta, "epoch": epoch, "train_loss": tr, "val_loss": va})


if __name__ == "__main__":
    main()


