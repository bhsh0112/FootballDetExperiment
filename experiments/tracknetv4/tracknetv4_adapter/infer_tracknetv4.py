"""
TrackNetV4 推理入口（足球轨迹输出）。

示例：
python -m tracknetv4_adapter.infer_tracknetv4 \
  --tracknetv4_dir /abs/TrackNetV4 \
  --video_path /abs/video.mp4 \
  --model_weights /abs/model_final.keras \
  --output_dir /abs/out
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="TrackNetV4 Football Tracking Runner")
    p.add_argument("--tracknetv4_dir", type=str, required=True, help="TrackNetV4 仓库路径")
    p.add_argument("--video_path", type=str, required=True, help="输入视频路径")
    p.add_argument("--model_weights", type=str, required=True, help="TrackNetV4 .keras 权重路径")
    p.add_argument("--output_dir", type=str, required=True, help="输出目录")
    p.add_argument("--queue_length", type=int, default=5, help="轨迹历史长度")
    p.add_argument("--result_csv", type=str, default="", help="转换后CSV输出路径（可选）")
    return p.parse_args()


def ensure_tracknetv4_dir(tracknetv4_dir: str) -> Path:
    """
    检查 TrackNetV4 仓库路径。

    @param {str} tracknetv4_dir - TrackNetV4 仓库路径
    @returns {Path} - 规范化路径
    @throws {FileNotFoundError} - 路径不存在或缺少 predict.py
    """

    base = Path(tracknetv4_dir).expanduser().resolve()
    predict_py = base / "src" / "predict.py"
    if not predict_py.exists():
        raise FileNotFoundError(f"未找到 TrackNetV4 入口：{predict_py}")
    return base


def run_tracknetv4_predict(
    tracknetv4_dir: Path,
    video_path: str,
    model_weights: str,
    output_dir: str,
    queue_length: int,
) -> None:
    """
    执行 TrackNetV4 原生预测脚本。

    @param {Path} tracknetv4_dir - TrackNetV4 仓库路径
    @param {str} video_path - 输入视频
    @param {str} model_weights - 模型权重
    @param {str} output_dir - 输出目录
    @param {int} queue_length - 轨迹历史长度
    @returns {None}
    """

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "src/predict.py",
        "--video_path",
        video_path,
        "--model_weights",
        model_weights,
        "--output_dir",
        output_dir,
        "--queue_length",
        str(queue_length),
    ]
    subprocess.run(cmd, cwd=str(tracknetv4_dir), check=True)


def find_latest_csv(output_dir: str) -> Optional[str]:
    """
    获取输出目录中最新的 CSV 文件。

    @param {str} output_dir - 输出目录
    @returns {Optional[str]} - CSV 路径（若无则返回 None）
    """

    csv_files = list(Path(output_dir).glob("*.csv"))
    if not csv_files:
        return None
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def normalize_header_map(header: list[str]) -> Dict[str, int]:
    """
    将 CSV 头部字段映射为索引。

    @param {list[str]} header - 头部字段
    @returns {Dict[str, int]} - 字段到索引映射
    """

    return {h.strip().lower(): i for i, h in enumerate(header)}


def convert_tracknetv4_csv(input_csv: str, output_csv: str) -> int:
    """
    转换 TrackNetV4 输出 CSV 为统一格式。

    统一格式：frame,visibility,x,y

    @param {str} input_csv - TrackNetV4 CSV
    @param {str} output_csv - 转换后的 CSV
    @returns {int} - 写入行数
    """

    with open(input_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if not header:
            return 0
        mapping = normalize_header_map(header)
        has_named_fields = all(k in mapping for k in ("frame", "visibility", "x", "y"))

        rows = []
        for row in reader:
            if not row:
                continue
            if has_named_fields:
                frame = row[mapping["frame"]]
                visibility = row[mapping["visibility"]]
                x = row[mapping["x"]]
                y = row[mapping["y"]]
            else:
                if len(row) < 4:
                    continue
                frame, visibility, x, y = row[:4]
            rows.append((frame, visibility, x, y))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "visibility", "x", "y"])
        writer.writerows(rows)
    return len(rows)


def maybe_copy_csv(input_csv: str, output_csv: str) -> None:
    """
    当输出路径与输入路径一致时，直接复制原文件。

    @param {str} input_csv - 输入 CSV
    @param {str} output_csv - 输出 CSV
    @returns {None}
    """

    if os.path.abspath(input_csv) == os.path.abspath(output_csv):
        return
    shutil.copyfile(input_csv, output_csv)


def main() -> None:
    """
    主入口。

    @returns {None}
    """

    args = parse_args()
    tracknetv4_dir = ensure_tracknetv4_dir(args.tracknetv4_dir)
    run_tracknetv4_predict(
        tracknetv4_dir=tracknetv4_dir,
        video_path=args.video_path,
        model_weights=args.model_weights,
        output_dir=args.output_dir,
        queue_length=args.queue_length,
    )

    latest_csv = find_latest_csv(args.output_dir)
    if not latest_csv:
        print("未在输出目录找到 TrackNetV4 CSV 结果。")
        return

    output_csv = args.result_csv.strip() or os.path.join(args.output_dir, "tracknetv4_tracks.csv")
    convert_tracknetv4_csv(latest_csv, output_csv)
    print(f"已生成轨迹 CSV：{output_csv}")


if __name__ == "__main__":
    main()

