"""
调试脚本：检查模型输出和标签的对应关系
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from inference_ttnet import load_model, preprocess_image, postprocess_output

# 测试图像和标签
image_path = 'train/images/07-02-2025_18-02-00_jpg.rf.dd9bde48d193d4f1331d3e07f9934eb3.jpg'
label_path = 'train/labels/07-02-2025_18-02-00_jpg.rf.dd9bde48d193d4f1331d3e07f9934eb3.txt'
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'ttnet.pth')
device = 'cuda'

print("=" * 60)
print("调试模型输出")
print("=" * 60)

# 1. 读取真实标签
print("\n1. 读取真实标签...")
true_boxes = []
with open(label_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5 and int(parts[0]) == 0:  # Ball类别
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            true_boxes.append({
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
            print(f"  真实目标: x={x_center:.4f}, y={y_center:.4f}, w={width:.4f}, h={height:.4f}")

# 2. 加载模型并推理
print("\n2. 加载模型并推理...")
model = load_model(model_path, device, input_size=640)
image_tensor, original_image, _ = preprocess_image(image_path, 640)
original_h, original_w = original_image.shape[:2]

print(f"  原始图像尺寸: {original_w}x{original_h}")

with torch.no_grad():
    image_tensor = image_tensor.to(device)
    predictions = model(image_tensor)

# 3. 检查模型输出
print("\n3. 检查模型输出...")
global_pred = predictions.get('global', {})
cls_output = global_pred.get('cls', None)
loc_output = global_pred.get('loc', None)
size_output = global_pred.get('size', None)

if cls_output is not None:
    cls_prob = torch.sigmoid(cls_output)
    print(f"  分类输出形状: {cls_output.shape}")
    print(f"  分类概率范围: [{cls_prob.min().item():.4f}, {cls_prob.max().item():.4f}]")
    print(f"  分类概率均值: {cls_prob.mean().item():.4f}")
    
    # 找到最高置信度的位置
    max_idx = cls_prob.argmax()
    max_y, max_x = torch.unravel_index(max_idx, cls_prob.shape[2:])
    max_conf = cls_prob[0, 0, max_y, max_x].item()
    print(f"  最高置信度位置: ({max_x}, {max_y}), 置信度: {max_conf:.4f}")

if loc_output is not None:
    print(f"  位置输出形状: {loc_output.shape}")
    print(f"  位置输出范围: x=[{loc_output[0, 0].min().item():.4f}, {loc_output[0, 0].max().item():.4f}], "
          f"y=[{loc_output[0, 1].min().item():.4f}, {loc_output[0, 1].max().item():.4f}]")
    print(f"  位置输出均值: x={loc_output[0, 0].mean().item():.4f}, y={loc_output[0, 1].mean().item():.4f}")
    
    # 在最高置信度位置检查位置输出
    if cls_output is not None:
        max_y, max_x = torch.unravel_index(cls_prob.argmax(), cls_prob.shape[2:])
        loc_x = loc_output[0, 0, max_y, max_x].item()
        loc_y = loc_output[0, 1, max_y, max_x].item()
        print(f"  最高置信度位置的位置输出: x={loc_x:.4f}, y={loc_y:.4f}")

if size_output is not None:
    print(f"  尺寸输出形状: {size_output.shape}")
    print(f"  尺寸输出范围: w=[{size_output[0, 0].min().item():.4f}, {size_output[0, 0].max().item():.4f}], "
          f"h=[{size_output[0, 1].min().item():.4f}, {size_output[0, 1].max().item():.4f}]")
    print(f"  尺寸输出均值: w={size_output[0, 0].mean().item():.4f}, h={size_output[0, 1].mean().item():.4f}")
    
    # 在最高置信度位置检查尺寸输出
    if cls_output is not None:
        max_y, max_x = torch.unravel_index(cls_prob.argmax(), cls_prob.shape[2:])
        size_w = size_output[0, 0, max_y, max_x].item()
        size_h = size_output[0, 1, max_y, max_x].item()
        print(f"  最高置信度位置的尺寸输出: w={size_w:.4f}, h={size_h:.4f}")

# 4. 测试后处理
print("\n4. 测试后处理...")
for conf_thresh in [0.05, 0.1, 0.2, 0.3]:
    boxes = postprocess_output(predictions, conf_thresh, original_size=(original_w, original_h))
    print(f"  置信度阈值 {conf_thresh}: 检测到 {len(boxes)} 个目标")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box['bbox']
        conf = box['conf']
        # 转换为归一化坐标以便比较
        x_center_norm = ((x1 + x2) / 2) / original_w
        y_center_norm = ((y1 + y2) / 2) / original_h
        width_norm = (x2 - x1) / original_w
        height_norm = (y2 - y1) / original_h
        print(f"    目标{i+1}: conf={conf:.4f}, "
              f"x={x_center_norm:.4f}, y={y_center_norm:.4f}, "
              f"w={width_norm:.4f}, h={height_norm:.4f}")

# 5. 比较真实标签和检测结果
print("\n5. 比较真实标签和检测结果...")
if len(boxes) > 0 and len(true_boxes) > 0:
    print("  真实标签 vs 检测结果:")
    for i, true_box in enumerate(true_boxes):
        print(f"    真实{i+1}: x={true_box['x_center']:.4f}, y={true_box['y_center']:.4f}, "
              f"w={true_box['width']:.4f}, h={true_box['height']:.4f}")
    
    for i, det_box in enumerate(boxes):
        x1, y1, x2, y2 = det_box['bbox']
        x_center_norm = ((x1 + x2) / 2) / original_w
        y_center_norm = ((y1 + y2) / 2) / original_h
        width_norm = (x2 - x1) / original_w
        height_norm = (y2 - y1) / original_h
        print(f"    检测{i+1}: x={x_center_norm:.4f}, y={y_center_norm:.4f}, "
              f"w={width_norm:.4f}, h={height_norm:.4f}, conf={det_box['conf']:.4f}")

print("\n" + "=" * 60)


