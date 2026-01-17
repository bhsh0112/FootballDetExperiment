"""
基于TTNet架构的小目标检测模型
实现全局和局部两阶段检测，专门针对小目标（Ball）优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GlobalStage(nn.Module):
    """
    全局阶段检测模块
    用于粗定位小目标的位置
    """
    
    def __init__(self, num_classes=1, input_size=640):
        """
        初始化全局检测阶段
        
        @param {int} num_classes - 类别数量（默认1，只检测Ball）
        @param {int} input_size - 输入图像尺寸
        """
        super(GlobalStage, self).__init__()
        
        # 使用ResNet作为backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 特征金字塔网络（FPN）用于多尺度特征融合
        self.fpn = FeaturePyramidNetwork(2048, 256)
        
        # 检测头
        self.detection_head = DetectionHead(256, num_classes, input_size)
        
    def forward(self, x):
        """
        前向传播
        
        @param {torch.Tensor} x - 输入图像 [B, 3, H, W]
        @returns {dict} - 包含检测结果的字典
        """
        # 提取特征
        features = self.backbone(x)
        
        # FPN特征融合
        fpn_features = self.fpn(features)
        
        # 检测
        detection = self.detection_head(fpn_features)
        
        return detection


class LocalStage(nn.Module):
    """
    局部阶段检测模块
    用于精确定位和细化小目标检测
    """
    
    def __init__(self, num_classes=1, crop_size=128):
        """
        初始化局部检测阶段
        
        @param {int} num_classes - 类别数量
        @param {int} crop_size - 裁剪区域尺寸
        """
        super(LocalStage, self).__init__()
        
        # 使用更轻量的backbone进行局部细化
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 局部检测头
        self.refinement_head = RefinementHead(512, num_classes, crop_size)
        
    def forward(self, x):
        """
        前向传播
        
        @param {torch.Tensor} x - 裁剪后的图像块 [B, 3, crop_size, crop_size]
        @returns {dict} - 包含细化检测结果的字典
        """
        features = self.backbone(x)
        refinement = self.refinement_head(features)
        
        return refinement


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络
    用于多尺度特征融合，提高小目标检测能力
    """
    
    def __init__(self, in_channels=2048, out_channels=256):
        """
        初始化FPN
        
        @param {int} in_channels - 输入通道数
        @param {int} out_channels - 输出通道数
        """
        super(FeaturePyramidNetwork, self).__init__()
        
        # 侧边连接
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # 输出层
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, x):
        """
        前向传播
        
        @param {torch.Tensor} x - 输入特征 [B, C, H, W]
        @returns {torch.Tensor} - 融合后的特征
        """
        # 侧边连接
        lateral = self.lateral_conv(x)
        
        # 上采样到相同尺寸
        lateral = F.interpolate(lateral, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 输出
        output = self.output_conv(lateral)
        
        return output


class DetectionHead(nn.Module):
    """
    检测头
    输出目标的位置和类别
    """
    
    def __init__(self, in_channels=256, num_classes=1, input_size=640):
        """
        初始化检测头
        
        @param {int} in_channels - 输入通道数
        @param {int} num_classes - 类别数量
        @param {int} input_size - 输入图像尺寸
        """
        super(DetectionHead, self).__init__()
        
        self.input_size = input_size
        
        # 分类分支
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # 位置回归分支（使用1D高斯分布表示目标位置）
        self.loc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # x和y坐标
        )
        
        # 尺寸回归分支
        self.size_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # width和height
        )
        
    def forward(self, x):
        """
        前向传播
        
        @param {torch.Tensor} x - 输入特征 [B, C, H, W]
        @returns {dict} - 检测结果
        """
        # 获取输入特征图尺寸
        _, _, h, w = x.shape
        
        # 分类输出
        cls_output = self.cls_conv(x)
        
        # 位置输出
        loc_output = self.loc_conv(x)
        
        # 尺寸输出
        size_output = self.size_conv(x)
        
        # 上采样到输入尺寸（如果特征图尺寸小于输入尺寸）
        if h != self.input_size or w != self.input_size:
            cls_output = F.interpolate(
                cls_output, 
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
            loc_output = F.interpolate(
                loc_output,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
            size_output = F.interpolate(
                size_output,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
        
        return {
            'cls': cls_output,
            'loc': loc_output,
            'size': size_output
        }


class RefinementHead(nn.Module):
    """
    细化检测头
    用于局部阶段的精确检测
    """
    
    def __init__(self, in_channels=512, num_classes=1, crop_size=128):
        """
        初始化细化检测头
        
        @param {int} in_channels - 输入通道数
        @param {int} num_classes - 类别数量
        @param {int} crop_size - 裁剪尺寸
        """
        super(RefinementHead, self).__init__()
        
        self.crop_size = crop_size
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类分支
        self.cls_fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 位置回归分支
        self.loc_fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # x, y坐标（相对于crop区域）
        )
        
        # 尺寸回归分支
        self.size_fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # width, height
        )
        
    def forward(self, x):
        """
        前向传播
        
        @param {torch.Tensor} x - 输入特征 [B, C, H, W]
        @returns {dict} - 细化检测结果
        """
        # 全局池化
        pooled = self.global_pool(x).view(x.size(0), -1)
        
        # 分类
        cls_output = self.cls_fc(pooled)
        
        # 位置
        loc_output = self.loc_fc(pooled)
        
        # 尺寸
        size_output = self.size_fc(pooled)
        
        return {
            'cls': cls_output,
            'loc': loc_output,
            'size': size_output
        }


class TTNetSmallObject(nn.Module):
    """
    基于TTNet架构的小目标检测模型
    结合全局和局部两阶段检测
    """
    
    def __init__(self, num_classes=1, input_size=640, crop_size=128, use_local=True):
        """
        初始化TTNet小目标检测模型
        
        @param {int} num_classes - 类别数量（默认1，只检测Ball）
        @param {int} input_size - 输入图像尺寸
        @param {int} crop_size - 局部裁剪尺寸
        @param {bool} use_local - 是否使用局部阶段
        """
        super(TTNetSmallObject, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.crop_size = crop_size
        self.use_local = use_local
        
        # 全局阶段
        self.global_stage = GlobalStage(num_classes, input_size)
        
        # 局部阶段（可选）
        if use_local:
            self.local_stage = LocalStage(num_classes, crop_size)
        else:
            self.local_stage = None
        
    def forward(self, x, crop_regions=None):
        """
        前向传播
        
        @param {torch.Tensor} x - 输入图像 [B, 3, H, W]
        @param {list} crop_regions - 裁剪区域列表（用于局部阶段）
        @returns {dict} - 检测结果
        """
        # 全局阶段检测
        global_output = self.global_stage(x)
        
        # 局部阶段细化（如果启用）
        local_output = None
        if self.use_local and self.local_stage is not None and crop_regions is not None:
            # 从全局检测结果中提取感兴趣区域
            crops = self._extract_crops(x, global_output, crop_regions)
            if crops is not None and crops.size(0) > 0:
                local_output = self.local_stage(crops)
        
        return {
            'global': global_output,
            'local': local_output
        }
    
    def _extract_crops(self, x, global_output, crop_regions):
        """
        从输入图像中提取裁剪区域
        
        @param {torch.Tensor} x - 输入图像
        @param {dict} global_output - 全局检测结果
        @param {list} crop_regions - 裁剪区域
        @returns {torch.Tensor} - 裁剪后的图像块
        """
        # 这里简化实现，实际应该根据global_output的检测结果来提取
        # 暂时返回None，需要根据具体需求实现
        return None

