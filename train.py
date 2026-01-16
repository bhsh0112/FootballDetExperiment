"""
方案A训练入口：多帧 Early-Fusion UNet -> heatmap。

示例：
python -m scheme_a_heatmap_tracker.train \
  --frames_dir /abs/frames \
  --points_csv /abs/points.csv \
  --out_dir /abs/out/run1
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .datasets import SequencePointDataset
from .models import EarlyFusionUNet


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="Scheme-A Heatmap Tracker Training")
    p.add_argument("--frames_dir", type=str, required=True, help="帧目录（抽帧后的图片）")
    p.add_argument("--points_csv", type=str, required=True, help="点标注 CSV（frame,x,y,visible）")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")

    p.add_argument("--window", type=int, default=5, help="多帧窗口长度")
    p.add_argument("--img_size", type=int, default=640, help="网络输入尺寸（正方形）")
    p.add_argument("--heatmap_size", type=int, default=160, help="heatmap 尺寸（正方形）")
    p.add_argument("--sigma", type=float, default=2.0, help="heatmap 高斯 sigma（heatmap 像素）")
    p.add_argument("--lambda_no_ball", type=float, default=0.05, help="不可见帧 loss 权重（0~0.2 常用）")

    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=8, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    p.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """
    设置随机种子。

    @param {int} seed - 种子
    @returns {None}
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_w: torch.Tensor,
) -> torch.Tensor:
    """
    heatmap 回归 loss：对可见帧正常回归，对不可见帧用较小权重约束为 0。

    @param {torch.Tensor} logits - (B,1,Hm,Wm)
    @param {torch.Tensor} target - (B,1,Hm,Wm)
    @param {torch.Tensor} loss_w - (B,1) 每个样本权重
    @returns {torch.Tensor} - scalar loss
    """

    pred = torch.sigmoid(logits)
    per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))  # (B,)
    w = loss_w.view(-1).to(per.dtype)
    return (per * w).mean()


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[list, list]:
    """
    切分 train/val 索引。

    @param {int} n - 总长度
    @param {float} val_ratio - 验证比例
    @param {int} seed - 随机种子
    @returns {Tuple[list,list]} - (train_idx, val_idx)
    """

    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = int(round(n * float(val_ratio)))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return train_idx, val_idx


def save_checkpoint(path: str, model: torch.nn.Module, meta: Dict) -> None:
    """
    保存 checkpoint。

    @param {str} path - 保存路径
    @param {torch.nn.Module} model - 模型
    @param {Dict} meta - 元信息
    @returns {None}
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model.state_dict(), "meta": meta}
    torch.save(payload, path)


def main() -> None:
    """
    主函数。

    @returns {None}
    """

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ds = SequencePointDataset(
        frames_dir=args.frames_dir,
        points_csv=args.points_csv,
        window=args.window,
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        lambda_no_ball=args.lambda_no_ball,
    )

    train_idx, val_idx = split_indices(len(ds), args.val_ratio, args.seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if len(val_idx) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        if val_ds is not None
        else None
    )

    model = EarlyFusionUNet(window=args.window, base_ch=32, out_size=(args.heatmap_size, args.heatmap_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    meta = {
        "window": int(args.window),
        "img_size": int(args.img_size),
        "heatmap_size": int(args.heatmap_size),
        "sigma": float(args.sigma),
        "lambda_no_ball": float(args.lambda_no_ball),
    }

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            lw = batch["loss_w"].to(device, non_blocking=True)

            out = model(x)
            loss = heatmap_loss(out.heatmap_logits, y, lw)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            train_losses.append(float(loss.item()))
            pbar.set_postfix({"loss": np.mean(train_losses[-50:]) if train_losses else 0.0})

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_loss = None
        if val_loader is not None:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="val", leave=False):
                    x = batch["x"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True)
                    lw = batch["loss_w"].to(device, non_blocking=True)
                    out = model(x)
                    l = heatmap_loss(out.heatmap_logits, y, lw)
                    v_losses.append(float(l.item()))
            val_loss = float(np.mean(v_losses)) if v_losses else float("inf")

        # save checkpoints
        save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **meta})
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(args.out_dir, "best.pt"), model, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **meta})

        print(f"[epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss if val_loss is not None else 'N/A'} best_val={best_val:.6f}")

    # 保存训练配置，方便推理复现
    with open(os.path.join(args.out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")


if __name__ == "__main__":
    main()


