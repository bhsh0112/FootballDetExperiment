"""
YOLO格式数据集加载器
适配TTNet小目标检测
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class YOLODataset(Dataset):
    """
    YOLO格式数据集
    专门用于小目标（Ball）检测
    """
    
    def __init__(self, images_dir, labels_dir, input_size=640, 
                 target_class=0, augment=True, transform=None):
        """
        初始化数据集
        
        @param {str} images_dir - 图片目录
        @param {str} labels_dir - 标签目录
        @param {int} input_size - 输入图像尺寸
        @param {int} target_class - 目标类别ID（0表示Ball）
        @param {bool} augment - 是否使用数据增强
        @param {callable} transform - 自定义变换
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        self.target_class = target_class
        self.augment = augment
        self.hflip_prob = 0.5
        
        # 获取所有图片文件
        self.image_files = []
        self.label_files = []
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in os.listdir(images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(images_dir, filename)
                label_path = os.path.join(labels_dir, 
                                         os.path.splitext(filename)[0] + '.txt')
                
                # 只包含有标签文件的图片
                if os.path.exists(label_path):
                    self.image_files.append(image_path)
                    self.label_files.append(label_path)
        
        # 数据增强
        if augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                      saturation=0.3, hue=0.1),
            ])
        else:
            self.transform = None
        
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        @param {int} idx - 索引
        @returns {dict} - 包含图像和标签的字典
        """
        # 读取图像
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # 调整图像尺寸
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # 读取标签
        label_path = self.label_files[idx]
        boxes, classes = self._load_yolo_labels(label_path, original_w, original_h)
        
        # 只保留目标类别的标注
        target_boxes = []
        for box, cls in zip(boxes, classes):
            if cls == self.target_class:
                target_boxes.append(box)

        # 数据增强：水平翻转（必须同步更新 YOLO 的 x_center，否则会导致训练坐标系统性错误/推理偏移）
        if self.augment and self.hflip_prob > 0:
            if np.random.rand() < self.hflip_prob:
                # 翻转图像（RGB，水平翻转）
                image = np.ascontiguousarray(image[:, ::-1, :])
                # 翻转标签：x_center -> 1 - x_center（YOLO归一化坐标）
                flipped_boxes = []
                for (x_center, y_center, width, height) in target_boxes:
                    flipped_boxes.append([1.0 - x_center, y_center, width, height])
                target_boxes = flipped_boxes

        # 转为PIL并做仅影响颜色的增强（不会改变几何坐标）
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        
        # 转换为tensor
        image = self.base_transform(image)
        
        # 生成目标（1D高斯分布表示位置）
        target = self._generate_target(target_boxes, self.input_size)
        
        return {
            'image': image,
            'target': {
                'cls': target['cls'],
                'loc_x': target['loc_x'],
                'loc_y': target['loc_y'],
                'size': target['size']
            }
        }
    
    def _load_yolo_labels(self, label_path, img_w, img_h):
        """
        加载YOLO格式标签
        
        @param {str} label_path - 标签文件路径
        @param {int} img_w - 图像宽度
        @param {int} img_h - 图像高度
        @returns {tuple} - (boxes, classes)
        """
        boxes = []
        classes = []
        
        if not os.path.exists(label_path):
            return boxes, classes
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 转换为像素坐标
                x_center_px = x_center * img_w
                y_center_px = y_center * img_h
                width_px = width * img_w
                height_px = height * img_h
                
                # 转换为边界框格式 [x_center, y_center, width, height] (归一化)
                box = [x_center, y_center, width, height]
                
                boxes.append(box)
                classes.append(cls_id)
        
        return boxes, classes
    
    def _generate_target(self, boxes, input_size, sigma=10.0):
        """
        生成目标（使用1D高斯分布表示位置，类似TTNet）
        
        @param {list} boxes - 边界框列表
        @param {int} input_size - 输入尺寸
        @param {float} sigma - 高斯分布标准差
        @returns {dict} - 目标字典
        """
        # 初始化目标
        cls_target = torch.zeros(1, input_size, input_size)  # 分类目标
        loc_x_target = torch.zeros(2, input_size, input_size)  # x坐标目标
        loc_y_target = torch.zeros(2, input_size, input_size)  # y坐标目标
        size_target = torch.zeros(2, input_size, input_size)  # 尺寸目标
        
        for box in boxes:
            x_center, y_center, width, height = box
            
            # 转换为像素坐标
            x_center_px = int(x_center * input_size)
            y_center_px = int(y_center * input_size)
            
            # 生成1D高斯分布（x方向）
            x_coords = torch.arange(input_size).float()
            gaussian_x = torch.exp(-0.5 * ((x_coords - x_center_px) / sigma) ** 2)
            gaussian_x = gaussian_x.unsqueeze(0).unsqueeze(0).repeat(1, input_size, 1)
            
            # 生成1D高斯分布（y方向）
            y_coords = torch.arange(input_size).float()
            gaussian_y = torch.exp(-0.5 * ((y_coords - y_center_px) / sigma) ** 2)
            gaussian_y = gaussian_y.unsqueeze(0).unsqueeze(2).repeat(1, 1, input_size)
            
            # 组合为2D高斯分布
            gaussian_2d = gaussian_x * gaussian_y
            
            # 更新分类目标
            cls_target = torch.max(cls_target, gaussian_2d)
            
            # 位置目标（归一化的坐标）
            # 使用torch.max而不是torch.maximum（兼容性）
            loc_x_target[0] = torch.max(loc_x_target[0], 
                                       gaussian_2d * x_center)
            loc_x_target[1] = torch.max(loc_x_target[1], 
                                       gaussian_2d * (1 - x_center))
            
            loc_y_target[0] = torch.max(loc_y_target[0], 
                                       gaussian_2d * y_center)
            loc_y_target[1] = torch.max(loc_y_target[1], 
                                       gaussian_2d * (1 - y_center))
            
            # 尺寸目标
            size_target[0] = torch.max(size_target[0], 
                                     gaussian_2d * width)
            size_target[1] = torch.max(size_target[1], 
                                     gaussian_2d * height)
        
        return {
            'cls': cls_target,
            'loc_x': loc_x_target,
            'loc_y': loc_y_target,
            'size': size_target
        }

