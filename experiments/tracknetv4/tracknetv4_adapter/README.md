# TrackNetV4 足球跟踪接入

本目录提供 TrackNetV4 的推理适配脚本，用于在本项目中直接输出足球轨迹 CSV。

## 依赖准备

1. 克隆 TrackNetV4 仓库到本项目目录（建议放到 `experiments/tracknetv4/third_party/TrackNetV4`）。
2. 按 TrackNetV4 官方说明配置 TensorFlow 环境并下载模型权重。

## 使用方式

```bash
cd /home/buaa/football_detect/experiments/tracknetv4
python -m tracknetv4_adapter.infer_tracknetv4 \
  --tracknetv4_dir /abs/TrackNetV4 \
  --video_path /abs/video.mp4 \
  --model_weights /abs/model_final.keras \
  --output_dir /abs/out \
  --queue_length 5
```

## 输出说明

- TrackNetV4 原生输出：视频与 CSV（由 TrackNetV4 生成）
- 适配输出：`tracknetv4_tracks.csv`（字段：`frame, visibility, x, y`）

## 用 data/ 生成 TrackNetV4 训练标注

将 YOLO 标注数据转换为 TrackNetV4 风格 CSV（`frame,visibility,x,y`），并可选复制/软链接 images。

```bash
cd /home/buaa/football_detect/experiments/tracknetv4
python -m tracknetv4_adapter.prepare_dataset_from_yolo \
  --data_root /home/buaa/football_detect/data \
  --out_dir /home/buaa/football_detect/experiments/tracknetv4/tracknetv4_data \
  --data_yaml /home/buaa/football_detect/data/data.yaml \
  --ball_class_name Ball \
  --pick_policy smallest \
  --image_mode none
```

输出：
- `tracknetv4_data/train/labels.csv`
- `tracknetv4_data/val/labels.csv`
- `tracknetv4_data/test/labels.csv`（若存在）
- `tracknetv4_data/dataset_info.json`

