# 基于TTNet架构的小目标检测

本项目基于[TTNet](https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch)架构，实现了专门针对足球（Ball）小目标检测的深度学习模型。

## 项目特点

- **两阶段检测**：全局阶段粗定位 + 局部阶段精确定位
- **小目标优化**：专门针对小目标（Ball）检测优化
- **多尺度特征融合**：使用FPN进行多尺度特征融合
- **Focal Loss**：处理类别不平衡问题
- **实时推理**：优化的模型结构支持实时推理

## 项目结构

```
football_detect/
├── experiments/
│   └── ttnet_small_object/          # TTNet 小目标检测尝试
│       ├── ttnet_small_object/      # 模型/数据/损失模块
│       │   ├── models/
│       │   ├── dataset/
│       │   └── losses/
│       ├── train_ttnet.py           # 训练脚本
│       ├── inference_ttnet.py       # 推理脚本
│       ├── weights/                 # 权重文件
│       └── runs/                    # 训练输出
├── data/                            # 数据集（YOLO 格式）
└── requirements.txt                 # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python experiments/ttnet_small_object/train_ttnet.py \
    --data data/data.yaml \
    --input-size 640 \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.001 \
    --use-focal-loss \
    --use-local \
    --save-dir experiments/ttnet_small_object/runs/ttnet
```

主要参数说明：
- `--data`: 数据集配置文件路径
- `--input-size`: 输入图像尺寸（默认640）
- `--batch-size`: 批次大小（根据GPU内存调整）
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--use-focal-loss`: 使用Focal Loss处理类别不平衡
- `--use-local`: 启用局部阶段细化
- `--save-dir`: 模型保存目录

### 2. 推理

```bash
python experiments/ttnet_small_object/inference_ttnet.py \
    --model experiments/ttnet_small_object/weights/ttnet.pth \
    --input path/to/image.jpg \
    --output output.jpg \
    --conf-thresh 0.5 \
    --show
```

参数说明：
- `--model`: 模型权重路径
- `--input`: 输入图像路径
- `--output`: 输出图像路径
- `--conf-thresh`: 置信度阈值
- `--show`: 显示结果

### 3. 查看训练过程

```bash
tensorboard --logdir experiments/ttnet_small_object/runs/ttnet/tensorboard
```

然后在浏览器中打开 `http://localhost:6006`

## 模型架构

### 全局阶段（Global Stage）
- **Backbone**: ResNet-50（预训练）
- **FPN**: 特征金字塔网络，用于多尺度特征融合
- **检测头**: 输出分类、位置和尺寸信息

### 局部阶段（Local Stage，可选）
- **Backbone**: ResNet-18（更轻量）
- **细化头**: 对全局阶段检测到的区域进行精确细化

### 损失函数
- **分类损失**: Focal Loss（处理类别不平衡）
- **位置损失**: MSE Loss（1D高斯分布表示）
- **尺寸损失**: MSE Loss

## 数据集格式

项目使用YOLO格式的数据集：
- 图片目录：`train/images/`, `valid/images/`, `test/images/`
- 标签目录：`train/labels/`, `valid/labels/`, `test/labels/`
- 标签格式：`class_id x_center y_center width height`（归一化坐标）

## 训练技巧

1. **小目标检测优化**：
   - 使用较大的输入尺寸（640或更大）
   - 启用Focal Loss处理类别不平衡
   - 使用数据增强提高模型泛化能力

2. **两阶段训练**：
   - 可以先只训练全局阶段
   - 然后冻结全局阶段，训练局部阶段
   - 最后联合微调

3. **学习率调整**：
   - 初始学习率：0.001
   - 使用ReduceLROnPlateau自动调整
   - 如果验证损失不下降，学习率会自动降低

## 性能优化

- 模型支持GPU加速
- 使用混合精度训练（AMP）可以加速训练
- 局部阶段可以按需启用/禁用

## 参考

- [TTNet论文](https://arxiv.org/pdf/2004.09927.pdf)
- [TTNet实现](https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch)

## 许可证

本项目基于TTNet架构实现，遵循相应的开源许可证。

