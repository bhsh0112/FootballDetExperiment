# 方案A：多帧 Heatmap 回归 + 卡尔曼平滑（足球小目标跟踪）

这个文件夹是**独立实现**的方案A：输入连续多帧图像，回归球的**像素级热力图(heatmap)**，再将 heatmap 的峰值转为球心点，并用**常速度卡尔曼滤波**进行轨迹平滑与丢失补点。

## 你会得到什么

- **训练**：用“球心点标注”训练一个多帧 heatmap 网络（默认 5 帧滑窗）
- **推理/跟踪**：对视频或帧目录输出每帧球心点（含置信度），并生成带轨迹的可视化结果
- **数据工具**：从 YOLO bbox 标签自动转换成球心点标注（便于复用你现有 `train/valid/test`）

## 依赖

本仓库根目录 `requirements.txt` 已包含必要依赖（`torch/torchvision/opencv-python/pillow/numpy/pyyaml/tqdm`）。

## 数据格式（推荐：序列帧目录 + 点标注 CSV）

### 1) 帧目录

把某段视频抽帧到一个目录，例如：

```text
<frames_dir>/
  frame_000000.jpg
  frame_000001.jpg
  ...
```

文件名里**最好包含可排序的帧号**（本实现会按文件名排序）。

### 2) 点标注（CSV）

一个 CSV 文件，对应到每一帧的球心点（像素坐标）：

```text
frame,x,y,visible
frame_000000.jpg,123.4,456.7,1
frame_000001.jpg,,,0
...
```

- `visible=1`：该帧可见且提供 `(x,y)`
- `visible=0`：该帧不可见（遮挡/出画/模糊），`x/y` 可以留空

## 快速开始

## 用本仓库 `data/`（YOLO格式）直接训练（推荐起步方式）

> 这套 Roboflow 数据通常是**独立图片**而非连续视频帧，因此建议先用 **window=1** 训练“单帧 heatmap 定位”作为起步模型；后续你有连续抽帧序列时，再把 window 提升到 5 做时序增强与跟踪端到端更稳。

一键转换+训练：

```bash
conda run -n yolo python -m scheme_a_heatmap_tracker.train_from_data_yolo \
  --data_root /home/buaa/football_detect/data \
  --out_dir /home/buaa/football_detect/runs/scheme_a/run1 \
  --window 1 \
  --img_size 640 \
  --heatmap_size 160 \
  --epochs 50
```

## 显存不足（OOM）怎么解决

如果你看到 `CUDA out of memory` 且 `nvidia-smi` 显示显存几乎被占满（例如只剩几百 MiB），通常是**别的进程占用显存**或 batch/模型过大。

建议按优先级尝试：

- **先清显存**：用 `nvidia-smi` 找到占用进程并结束，或重启 Python/终端会话。
- **启用 AMP**（强烈推荐）：加 `--amp`
- **减小 batch**：例如 `--batch_size 2`
- **减小模型**：例如 `--base_ch 16`
- **减小输入尺寸**：例如 `--img_size 320 --heatmap_size 80`

一个更稳妥的低显存命令示例：

```bash
conda run -n yolo python -m scheme_a_heatmap_tracker.train_from_data_yolo \
  --data_root /home/buaa/football_detect/data \
  --out_dir /home/buaa/football_detect/runs/scheme_a/run1 \
  --window 1 \
  --img_size 320 \
  --heatmap_size 80 \
  --base_ch 16 \
  --batch_size 2 \
  --amp \
  --epochs 50
```

### A. 从视频抽帧

```bash
python -m scheme_a_heatmap_tracker.tools.extract_frames \
  --video /abs/path/to/video.mp4 \
  --out_dir /abs/path/to/frames_dir \
  --every 1
```

## 半自动标注（给 keyframes / 视频抽帧做域内微调）

当 `data/` 训练出来的模型在你的机位/分辨率上不工作（域差异大）时，最有效的方式是：
对你自己的视频抽帧做少量“球心点”标注（例如 200~1000 帧），再用 `train.py` 或 `train_from_data_yolo.py` 做微调。

标注工具（OpenCV GUI）：

```bash
conda run -n yolo python -m scheme_a_heatmap_tracker.tools.annotate_points \
  --input /abs/frames_dir \
  --out_csv /abs/points.csv \
  --autosave
```

提示：
- 鼠标左键点击设置球心点
- `n/Space` 下一帧，`p` 上一帧，`v` 标记不可见，`s` 保存

### B. 从 YOLO 标签转换为“球心点 CSV”（复用现有 train/valid/test）

> 假设你已有 YOLO 格式标签：`<split>/labels/*.txt`，图像在 `<split>/images/*`。

```bash
python -m scheme_a_heatmap_tracker.tools.yolo_to_points \
  --images_dir /home/buaa/football_detect/train/images \
  --labels_dir /home/buaa/football_detect/train/labels \
  --out_csv /home/buaa/football_detect/train_ball_points.csv \
  --ball_class_id 0
```

### C. 训练（多帧滑窗）

```bash
python -m scheme_a_heatmap_tracker.train \
  --frames_dir /abs/path/to/frames_dir \
  --points_csv /abs/path/to/points.csv \
  --out_dir /abs/path/to/outputs/run1 \
  --window 5 \
  --img_size 640 \
  --heatmap_size 160 \
  --epochs 50 \
  --batch_size 8
```

### D. 推理 + 跟踪（输出轨迹与点文件）

```bash
python -m scheme_a_heatmap_tracker.infer_track \
  --weights /abs/path/to/outputs/run1/best.pt \
  --input /abs/path/to/video.mp4 \
  --out_dir /abs/path/to/out_vis \
  --window 5 \
  --img_size 640 \
  --heatmap_size 160 \
  --conf_thresh 0.25
```

输出：
- `tracks.csv`：逐帧球心（原图像素坐标）+ 置信度 + 是否为卡尔曼预测
- `tracked.mp4` 或帧结果图：叠加轨迹与当前点

## 关键可调参数（建议从这里开始调）

- `--window`：多帧输入长度（默认 5；高速运动可试 7/9）
- `--heatmap_size`：heatmap 分辨率（建议 `img_size/4`，如 640→160）
- `--sigma`：生成高斯 heatmap 的标准差（小球更小可用 1.0~2.5）
- `--conf_thresh`：低于阈值则判定“本帧无球”，交给卡尔曼预测


