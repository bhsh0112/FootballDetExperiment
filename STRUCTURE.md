# 目录结构说明

本项目已将不同尝试按实验独立归档，顶层仅保留数据集、通用工具与产出目录。

## 实验归档

- `experiments/ttnet_small_object/`：TTNet 小目标检测尝试（训练/推理/权重/输出）
- `experiments/yolo/`：YOLO 系列尝试（推理、训练、权重、第三方 yolov13）
- `experiments/roi_flow_tracker/roi_flow_tracker/`：ROI 光流跟踪尝试
- `experiments/optical_flow_tracker/optical_flow_tracker/`：光流跟踪尝试（v1）
- `experiments/optical_flow_tracker_v2/optical_flow_tracker_v2/`：光流跟踪尝试（v2）
- `experiments/fusion_tracker/fusion_tracker/`：融合跟踪尝试
- `experiments/scheme_a_heatmap_tracker/scheme_a_heatmap_tracker/`：热力图方案尝试
- `experiments/tracknetv4/`：TrackNetV4 适配与数据准备

## 其他目录

- `data/`：YOLO 数据集
- `tools/`：通用工具脚本（如关键帧抽取、数据筛选）
- `output/` / `runs/`：历史产物（未强制迁移，按需可清理或再归档）
- `test_game/` / `keyframes/`：测试视频与关键帧数据
- `misc/cli_args/`：历史命令残留参数文件

