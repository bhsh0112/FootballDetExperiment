"""
基于TTNet架构的小目标检测训练脚本
专门针对足球（Ball）小目标检测优化
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ttnet_small_object.models import TTNetSmallObject
from ttnet_small_object.dataset import YOLODataset
from ttnet_small_object.losses import TTNetLoss


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='TTNet小目标检测训练')
    
    # 数据参数
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--input-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--target-class', type=int, default=0,
                       help='目标类别ID（0=Ball）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--use-local', action='store_true',
                       help='是否使用局部阶段')
    parser.add_argument('--crop-size', type=int, default=128,
                       help='局部裁剪尺寸')
    
    # 损失函数参数
    parser.add_argument('--cls-weight', type=float, default=1.0,
                       help='分类损失权重')
    parser.add_argument('--loc-weight', type=float, default=1.0,
                       help='位置损失权重')
    parser.add_argument('--size-weight', type=float, default=1.0,
                       help='尺寸损失权重')
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='是否使用Focal Loss')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    default_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'ttnet')
    parser.add_argument('--save-dir', type=str, default=default_save_dir,
                       help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--print-freq', type=int, default=10,
                       help='打印频率')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='保存频率（epoch）')
    
    return parser.parse_args()


def load_data_config(data_yaml):
    """
    加载数据集配置
    
    @param {str} data_yaml - 配置文件路径
    @returns {dict} - 配置字典
    """
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config, args):
    """
    创建数据加载器
    
    @param {dict} config - 数据集配置
    @param {argparse.Namespace} args - 参数
    @returns {tuple} - (train_loader, val_loader)
    """
    # 训练集
    train_dataset = YOLODataset(
        images_dir=config['train'],
        labels_dir=config['train'].replace('/images', '/labels'),
        input_size=args.input_size,
        target_class=args.target_class,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 验证集
    val_dataset = YOLODataset(
        images_dir=config['val'],
        labels_dir=config['val'].replace('/images', '/labels'),
        input_size=args.input_size,
        target_class=args.target_class,
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """
    训练一个epoch
    
    @param {nn.Module} model - 模型
    @param {DataLoader} train_loader - 训练数据加载器
    @param {nn.Module} criterion - 损失函数
    @param {optim.Optimizer} optimizer - 优化器
    @param {str} device - 设备
    @param {int} epoch - 当前epoch
    @param {argparse.Namespace} args - 参数
    @returns {dict} - 训练统计信息
    """
    model.train()
    
    total_loss = 0.0
    cls_loss_sum = 0.0
    loc_loss_sum = 0.0
    size_loss_sum = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = {
            'cls': batch['target']['cls'].to(device),
            'loc_x': batch['target']['loc_x'].to(device),
            'loc_y': batch['target']['loc_y'].to(device),
            'size': batch['target']['size'].to(device)
        }
        
        # 前向传播
        optimizer.zero_grad()
        predictions = model(images)
        
        # 计算损失
        losses = criterion(predictions, targets)
        
        # 反向传播
        losses['total_loss'].backward()
        optimizer.step()
        
        # 统计
        total_loss += losses['total_loss'].item()
        cls_loss_sum += losses['cls_loss'].item()
        loc_loss_sum += losses['loc_loss'].item()
        size_loss_sum += losses['size_loss'].item()
        
        # 更新进度条
        if (batch_idx + 1) % args.print_freq == 0:
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'cls': f"{losses['cls_loss'].item():.4f}",
                'loc': f"{losses['loc_loss'].item():.4f}"
            })
    
    num_batches = len(train_loader)
    
    return {
        'total_loss': total_loss / num_batches,
        'cls_loss': cls_loss_sum / num_batches,
        'loc_loss': loc_loss_sum / num_batches,
        'size_loss': size_loss_sum / num_batches
    }


def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    @param {nn.Module} model - 模型
    @param {DataLoader} val_loader - 验证数据加载器
    @param {nn.Module} criterion - 损失函数
    @param {str} device - 设备
    @returns {dict} - 验证统计信息
    """
    model.eval()
    
    total_loss = 0.0
    cls_loss_sum = 0.0
    loc_loss_sum = 0.0
    size_loss_sum = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            targets = {
                'cls': batch['target']['cls'].to(device),
                'loc_x': batch['target']['loc_x'].to(device),
                'loc_y': batch['target']['loc_y'].to(device),
                'size': batch['target']['size'].to(device)
            }
            
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            losses = criterion(predictions, targets)
            
            # 统计
            total_loss += losses['total_loss'].item()
            cls_loss_sum += losses['cls_loss'].item()
            loc_loss_sum += losses['loc_loss'].item()
            size_loss_sum += losses['size_loss'].item()
    
    num_batches = len(val_loader)
    
    return {
        'total_loss': total_loss / num_batches,
        'cls_loss': cls_loss_sum / num_batches,
        'loc_loss': loc_loss_sum / num_batches,
        'size_loss': size_loss_sum / num_batches
    }


def main():
    """
    主训练函数
    """
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据集配置
    config = load_data_config(args.data)
    print(f'数据集配置: {config}')
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(config, args)
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'验证集大小: {len(val_loader.dataset)}')
    
    # 创建模型
    model = TTNetSmallObject(
        num_classes=1,
        input_size=args.input_size,
        crop_size=args.crop_size,
        use_local=args.use_local
    ).to(device)
    
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 创建损失函数
    criterion = TTNetLoss(
        cls_weight=args.cls_weight,
        loc_weight=args.loc_weight,
        size_weight=args.size_weight,
        use_focal_loss=args.use_focal_loss
    )
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f'从epoch {start_epoch}恢复训练')
    
    # 训练循环
    print('开始训练...')
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )
        
        # 验证
        val_stats = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_stats['total_loss'])
        
        # 记录到tensorboard
        writer.add_scalar('Loss/Train', train_stats['total_loss'], epoch)
        writer.add_scalar('Loss/Val', val_stats['total_loss'], epoch)
        writer.add_scalar('Loss/Cls_Train', train_stats['cls_loss'], epoch)
        writer.add_scalar('Loss/Cls_Val', val_stats['cls_loss'], epoch)
        writer.add_scalar('Loss/Loc_Train', train_stats['loc_loss'], epoch)
        writer.add_scalar('Loss/Loc_Val', val_stats['loc_loss'], epoch)
        
        # 打印统计信息
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  训练损失: {train_stats["total_loss"]:.4f}')
        print(f'  验证损失: {val_stats["total_loss"]:.4f}')
        print(f'  分类损失: {val_stats["cls_loss"]:.4f}')
        print(f'  位置损失: {val_stats["loc_loss"]:.4f}')
        
        # 保存最佳模型
        if val_stats['total_loss'] < best_val_loss:
            best_val_loss = val_stats['total_loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  保存最佳模型 (val_loss: {best_val_loss:.4f})')
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print('训练完成！')


if __name__ == '__main__':
    main()

