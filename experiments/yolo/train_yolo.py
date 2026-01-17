"""
YOLOv11训练脚本
用于训练足球检测模型（Ball, Player, Ref）
"""

from ultralytics import YOLO
import os
import torch

def main():
    """
    主训练函数
    """
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 数据集配置文件路径
    data_yaml = 'data.yaml'
    
    # 检查配置文件是否存在
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"找不到配置文件: {data_yaml}")
    
    # 初始化YOLOv11模型
    # 可以选择不同的模型大小: yolo11n.pt (nano), yolo11s.pt (small), 
    # yolo11m.pt (medium), yolo11l.pt (large), yolo11x.pt (xlarge)
    model = YOLO('yolo11n.pt')  # 从预训练模型开始
    
    # 训练参数
    results = model.train(
        data=data_yaml,           # 数据集配置文件
        epochs=100,               # 训练轮数
        imgsz=640,                # 输入图像尺寸
        batch=16,                 # 批次大小（根据GPU内存调整）
        device=device,            # 训练设备
        workers=8,                # 数据加载线程数
        project=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'detect'),    # 项目保存路径
        name='football_detect',   # 实验名称
        exist_ok=True,            # 允许覆盖已存在的实验
        pretrained=True,          # 使用预训练权重
        optimizer='auto',         # 优化器：auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp
        verbose=True,             # 打印详细信息
        seed=0,                   # 随机种子
        deterministic=True,       # 确定性训练
        single_cls=False,         # 多类别检测
        rect=False,               # 矩形训练
        cos_lr=False,             # 余弦学习率调度
        close_mosaic=10,          # 最后N个epoch关闭mosaic增强
        resume=False,             # 是否从上次检查点恢复
        amp=True,                 # 自动混合精度训练
        fraction=1.0,             # 使用数据集的比例
        profile=False,            # 性能分析
        freeze=None,              # 冻结层数（None表示不冻结）
        # 数据增强参数
        hsv_h=0.015,              # 色调增强
        hsv_s=0.7,                # 饱和度增强
        hsv_v=0.4,                # 明度增强
        degrees=0.0,              # 旋转角度
        translate=0.1,            # 平移
        scale=0.5,                # 缩放
        shear=0.0,                # 剪切
        perspective=0.0,          # 透视变换
        flipud=0.0,               # 上下翻转概率
        fliplr=0.5,               # 左右翻转概率
        mosaic=1.0,               # Mosaic增强概率
        mixup=0.0,                # MixUp增强概率
        copy_paste=0.0,           # Copy-paste增强概率
    )
    
    # 训练完成后进行验证
    print("\n开始验证模型...")
    metrics = model.val()
    print(f"验证结果: {metrics}")
    
    # 导出模型（可选）
    # model.export(format='onnx')  # 导出为ONNX格式
    # model.export(format='torchscript')  # 导出为TorchScript格式
    
    print("\n训练完成！")
    print(f"模型保存在: {model.trainer.best}")
    print(f"训练结果保存在: runs/detect/football_detect")

if __name__ == '__main__':
    main()

