#!/bin/bash
# TTNet小目标检测快速开始脚本

echo "=========================================="
echo "TTNet小目标检测 - 快速开始"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    DEVICE="cuda"
else
    echo "未检测到GPU，将使用CPU"
    DEVICE="cpu"
fi

echo ""
echo "开始训练..."
echo ""

# 训练参数
python "$SCRIPT_DIR/train_ttnet.py" \
    --data "$ROOT_DIR/data/data.yaml" \
    --input-size 640 \
    --target-class 0 \
    --batch-size 8 \
    --epochs 50 \
    --lr 0.001 \
    --device $DEVICE \
    --use-focal-loss \
    --use-local \
    --save-dir "$SCRIPT_DIR/runs/ttnet" \
    --print-freq 10 \
    --save-freq 5

echo ""
echo "训练完成！"
echo "模型保存在: $SCRIPT_DIR/runs/ttnet/best_model.pth"
echo ""
echo "查看训练过程:"
echo "  tensorboard --logdir $SCRIPT_DIR/runs/ttnet/tensorboard"
echo ""
echo "进行推理:"
echo "  python $SCRIPT_DIR/inference_ttnet.py --model $SCRIPT_DIR/runs/ttnet/best_model.pth --input <image_path> --output output.jpg"

