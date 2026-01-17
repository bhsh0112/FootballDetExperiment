# 足球目标跟踪实验总览

本仓库汇总了多种足球检测与跟踪方案的实验实现与结果产出，方便团队快速了解每种尝试的目标、方法、数据与效果。

## 快速导航

- 结构说明：`STRUCTURE.md`
- 实验目录：`experiments/`
- 通用工具：`tools/`
- 数据集：`data/`（已在 `.gitignore` 中默认忽略）

## 实验清单与效果

> 说明：效果指标（如 mAP/Precision/Recall、FPS、轨迹误差等）建议在每个实验条目中补充。

### 1) TTNet 小目标检测

- 位置：`experiments/ttnet_small_object/`
- 目的：面向小目标（足球）检测的单模型尝试
- 训练脚本：`experiments/ttnet_small_object/train_ttnet.py`
- 推理脚本：`experiments/ttnet_small_object/inference_ttnet.py`
- 产出目录：
  - 训练输出：`experiments/ttnet_small_object/runs/`
  - 推理输出：`experiments/ttnet_small_object/output/`
- 当前效果：
  - 数据集：`https://universe.roboflow.com/football-sfolh/football-inlmh`

### 2) YOLO 系列检测

- 位置：`experiments/yolo/`
- 目的：使用 YOLOv11/YOLOv13 等模型进行检测对比
- 训练脚本：
  - YOLOv11：`experiments/yolo/train_yolo.py`
  - YOLOv13（单类）：`experiments/yolo/train_yolov13_ball.py`
- 推理脚本：`experiments/yolo/inference_yolo.py`
- 产出目录：
  - 训练输出：`experiments/yolo/runs/`
  - 推理输出：`experiments/yolo/output/`
- 当前效果：
  - 数据集：`https://universe.roboflow.com/football-sfolh/football-inlmh`

### 3) ROI 光流跟踪

- 位置：`experiments/roi_flow_tracker/roi_flow_tracker/`
- 目的：基于 ROI 的光流跟踪
- 入口脚本：`experiments/roi_flow_tracker/roi_flow_tracker/track_ball_roi_flow.py`
- 产出目录：`experiments/roi_flow_tracker/roi_flow_tracker/output*/`
- 当前效果：
  - 视频/场景：`蒲公英数据`

### 4) 光流跟踪 v1

- 位置：`experiments/optical_flow_tracker/optical_flow_tracker/`
- 目的：传统光流跟踪基线
- 入口脚本：`experiments/optical_flow_tracker/optical_flow_tracker/track_ball_optical_flow.py`
- 产出目录：`experiments/optical_flow_tracker/optical_flow_tracker/output*/`
- 当前效果：
  - 视频/场景：`蒲公英数据`

### 5) 光流跟踪 v2

- 位置：`experiments/optical_flow_tracker_v2/optical_flow_tracker_v2/`
- 目的：光流跟踪改进版（方法/改动待补充）
- 入口脚本：`（目前仍在实验）`
- 产出目录：`（目前仍在实验）`
- 当前效果：
  - 视频/场景：`（待补充）`
  - 指标：`（待补充）`
  - 备注：`（待补充）`

### 6) 融合跟踪

- 位置：`experiments/fusion_tracker/fusion_tracker/`
- 目的：融合多源检测/跟踪信息
- 入口脚本：`experiments/fusion_tracker/fusion_tracker/track_ball_fusion.py`
- 产出目录：`experiments/fusion_tracker/fusion_tracker/output*/`
- 当前效果：
  - 视频/场景：`https://universe.roboflow.com/football-sfolh/football-inlmh`

### 7) 热力图方案（Scheme A）

- 位置：`experiments/scheme_a_heatmap_tracker/scheme_a_heatmap_tracker/`
- 目的：基于热力图回归的跟踪方案
- 入口脚本：`experiments/scheme_a_heatmap_tracker/scheme_a_heatmap_tracker/infer_track.py`
- 产出目录：`experiments/scheme_a_heatmap_tracker/scheme_a_heatmap_tracker/runs*/`
- 当前效果：
  - 数据集：`https://universe.roboflow.com/football-sfolh/football-inlmh`


### 8) TrackNetV4 适配

- 位置：`experiments/tracknetv4/`
- 目的：TrackNetV4 的推理与数据转换适配
- 入口脚本：`experiments/tracknetv4/tracknetv4_adapter/infer_tracknetv4.py`
- 数据转换：`experiments/tracknetv4/tracknetv4_adapter/prepare_dataset_from_yolo.py`
- 产出目录：
  - 数据：`experiments/tracknetv4/tracknetv4_data/`
  - 推理：`（由 TrackNetV4 原仓库生成）`
- 当前效果：
  - 数据集：`https://universe.roboflow.com/football-sfolh/football-inlmh`

## 备注

- 若新增实验，请在 `experiments/` 下创建独立目录，并在本 README 更新实验清单与效果。
- 如需共享大文件（数据/权重/视频），建议使用对象存储或 Git LFS。

