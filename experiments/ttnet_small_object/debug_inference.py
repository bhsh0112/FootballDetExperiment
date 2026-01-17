"""
调试推理脚本 - 检查模型输出
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from inference_ttnet import load_model, preprocess_image

# 测试图像
image_path = 'train/images/07-02-2025_18-02-00_jpg.rf.dd9bde48d193d4f1331d3e07f9934eb3.jpg'
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'ttnet.pth')
device = 'cuda'

# 加载模型
print('加载模型...')
model = load_model(model_path, device, input_size=640)

# 预处理图像
print('预处理图像...')
image_tensor, original_image, _ = preprocess_image(image_path, 640)
original_h, original_w = original_image.shape[:2]

# 推理
print('进行推理...')
with torch.no_grad():
    image_tensor = image_tensor.to(device)
    predictions = model(image_tensor)

# 检查输出
global_pred = predictions.get('global', {})
cls_output = global_pred.get('cls', None)
loc_output = global_pred.get('loc', None)
size_output = global_pred.get('size', None)

print('\n=== 模型输出统计 ===')
if cls_output is not None:
    cls_prob = torch.sigmoid(cls_output)
    print(f'分类输出形状: {cls_output.shape}')
    print(f'分类输出范围: [{cls_output.min().item():.4f}, {cls_output.max().item():.4f}]')
    print(f'分类概率范围: [{cls_prob.min().item():.4f}, {cls_prob.max().item():.4f}]')
    print(f'分类概率均值: {cls_prob.mean().item():.4f}')
    print(f'分类概率最大值位置: {torch.unravel_index(cls_prob.argmax(), cls_prob.shape)}')
    
    # 检查不同阈值下的检测数量
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        mask = cls_prob > thresh
        count = mask.sum().item()
        print(f'阈值 {thresh}: {count} 个位置超过阈值')

if loc_output is not None:
    print(f'\n位置输出形状: {loc_output.shape}')
    print(f'位置输出范围: [{loc_output.min().item():.4f}, {loc_output.max().item():.4f}]')
    print(f'位置输出均值: [{loc_output[0, 0].mean().item():.4f}, {loc_output[0, 1].mean().item():.4f}]')

if size_output is not None:
    print(f'\n尺寸输出形状: {size_output.shape}')
    print(f'尺寸输出范围: [{size_output.min().item():.4f}, {size_output.max().item():.4f}]')
    print(f'尺寸输出均值: [{size_output[0, 0].mean().item():.4f}, {size_output[0, 1].mean().item():.4f}]')

# 尝试不同的后处理方式
print('\n=== 尝试后处理 ===')
from inference_ttnet import postprocess_output

for conf_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    boxes = postprocess_output(predictions, conf_thresh, original_size=(original_w, original_h))
    print(f'置信度阈值 {conf_thresh}: 检测到 {len(boxes)} 个目标')
    if len(boxes) > 0:
        print(f'  第一个目标: {boxes[0]}')



