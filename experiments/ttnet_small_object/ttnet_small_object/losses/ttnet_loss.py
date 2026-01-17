"""
TTNet损失函数
结合分类损失和位置回归损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTNetLoss(nn.Module):
    """
    TTNet损失函数
    包含分类损失和位置回归损失
    """
    
    def __init__(self, cls_weight=1.0, loc_weight=1.0, size_weight=1.0, 
                 use_focal_loss=True, alpha=0.25, gamma=2.0):
        """
        初始化损失函数
        
        @param {float} cls_weight - 分类损失权重
        @param {float} loc_weight - 位置损失权重
        @param {float} size_weight - 尺寸损失权重
        @param {bool} use_focal_loss - 是否使用Focal Loss
        @param {float} alpha - Focal Loss的alpha参数
        @param {float} gamma - Focal Loss的gamma参数
        """
        super(TTNetLoss, self).__init__()
        
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.size_weight = size_weight
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        
        # MSE损失用于位置和尺寸回归
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def forward(self, predictions, targets):
        """
        计算损失
        
        @param {dict} predictions - 模型预测结果
        @param {dict} targets - 目标标签
        @returns {dict} - 损失字典
        """
        global_pred = predictions.get('global', {})
        
        # 分类损失
        cls_pred = global_pred.get('cls', None)
        cls_target = targets.get('cls', None)
        cls_loss = 0.0
        
        if cls_pred is not None and cls_target is not None:
            if self.use_focal_loss:
                cls_loss = self._focal_loss(cls_pred, cls_target)
            else:
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_pred, cls_target, reduction='mean'
                )
        
        # 位置损失（x和y坐标）
        loc_pred = global_pred.get('loc', None)
        loc_x_target = targets.get('loc_x', None)
        loc_y_target = targets.get('loc_y', None)
        loc_x_loss = 0.0
        loc_y_loss = 0.0
        
        if loc_pred is not None and loc_x_target is not None and loc_y_target is not None:
            # 确保尺寸匹配
            if loc_pred.shape != loc_x_target.shape:
                # 如果尺寸不匹配，尝试上采样或下采样
                if loc_pred.shape[2:] != loc_x_target.shape[2:]:
                    loc_pred = F.interpolate(
                        loc_pred, 
                        size=loc_x_target.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
            
            # 使用mask只计算有目标的位置
            mask = cls_target > 0.1  # 阈值可调
            if mask.sum() > 0:
                # loc_pred的形状是 [B, 2, H, W]，第0通道是x，第1通道是y
                # loc_x_target和loc_y_target都是 [B, 2, H, W]
                # 我们使用第一个通道来计算损失
                if loc_pred.shape[1] >= 1 and loc_x_target.shape[1] >= 1:
                    # x坐标损失（使用第一个通道）
                    loc_x_pred = loc_pred[:, 0:1, :, :]  # x坐标预测
                    loc_x_loss = self.mse_loss(
                        loc_x_pred * mask.unsqueeze(1),
                        loc_x_target[:, 0:1, :, :] * mask.unsqueeze(1)
                    )
                
                if loc_pred.shape[1] >= 2 and loc_y_target.shape[1] >= 1:
                    # y坐标损失（使用第二个通道）
                    loc_y_pred = loc_pred[:, 1:2, :, :]  # y坐标预测
                    loc_y_loss = self.mse_loss(
                        loc_y_pred * mask.unsqueeze(1),
                        loc_y_target[:, 0:1, :, :] * mask.unsqueeze(1)
                    )
        
        # 尺寸损失
        size_pred = global_pred.get('size', None)
        size_target = targets.get('size', None)
        size_loss = 0.0
        
        if size_pred is not None and size_target is not None:
            mask = cls_target > 0.1
            if mask.sum() > 0:
                size_loss = self.mse_loss(
                    size_pred * mask.unsqueeze(1),
                    size_target * mask.unsqueeze(1)
                )
        
        # 总损失
        total_loss = (self.cls_weight * cls_loss + 
                     self.loc_weight * (loc_x_loss + loc_y_loss) +
                     self.size_weight * size_loss)
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'loc_loss': loc_x_loss + loc_y_loss,
            'size_loss': size_loss
        }
    
    def _focal_loss(self, pred, target):
        """
        Focal Loss实现
        用于处理类别不平衡问题，对小目标检测特别有效
        
        @param {torch.Tensor} pred - 预测值
        @param {torch.Tensor} target - 目标值
        @returns {torch.Tensor} - 损失值
        """
        # 计算BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算p_t
        p_t = torch.exp(-bce)
        
        # Focal Loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        
        return focal_loss.mean()

