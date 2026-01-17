"""
半自动球心点标注工具（OpenCV GUI）。

功能：
- 读取帧目录（或视频）并逐帧显示
- 鼠标左键点击设置球心点（记录为原图像素坐标）
- 支持标记“不可见”（visible=0）
- 随时保存为 CSV：frame,x,y,visible

建议用法（帧目录）：
conda run -n yolo python -m scheme_a_heatmap_tracker.tools.annotate_points \
  --input /abs/frames_dir \
  --out_csv /abs/points.csv

快捷键：
- n / Space : 下一帧
- p         : 上一帧
- v         : 切换可见/不可见（不可见会清空 x/y）
- c         : 清空当前帧点（置为不可见）
- s         : 保存 CSV
- q / Esc   : 退出（会提示保存）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore


@dataclass
class Anno:
    """
    单帧标注。

    @param {int} visible - 1可见，0不可见
    @param {Optional[float]} x - 原图坐标 x（像素）
    @param {Optional[float]} y - 原图坐标 y（像素）
    """

    visible: int = 0
    x: Optional[float] = None
    y: Optional[float] = None


def parse_args() -> argparse.Namespace:
    """
    解析参数。

    @returns {argparse.Namespace} - 参数
    """

    p = argparse.ArgumentParser(description="Annotate ball center points (OpenCV GUI)")
    p.add_argument("--input", type=str, required=True, help="输入：帧目录 或 视频文件")
    p.add_argument("--out_csv", type=str, required=True, help="输出 CSV：frame,x,y,visible")
    p.add_argument("--resume_csv", type=str, default=None, help="可选：从已有 CSV 继续标注")
    p.add_argument("--start", type=int, default=0, help="起始帧索引（按排序后的列表）")
    p.add_argument("--max_width", type=int, default=1280, help="显示最大宽度（会等比缩放）")
    p.add_argument("--autosave", action="store_true", help="每次改动后自动保存 CSV")
    return p.parse_args()


def is_video_file(path: str) -> bool:
    """
    判断是否视频文件。

    @param {str} path - 路径
    @returns {bool} - 是否视频
    """

    exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}
    return Path(path).suffix.lower() in exts


def list_frames_from_dir(frames_dir: str) -> List[Path]:
    """
    列出帧目录中的图片，按文件名排序。

    @param {str} frames_dir - 目录
    @returns {List[Path]} - 图片路径列表
    """

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames = [p for p in Path(frames_dir).iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(frames, key=lambda p: p.name)


def load_csv(csv_path: str) -> Dict[str, Anno]:
    """
    读取已有点标注 CSV。

    @param {str} csv_path - CSV 路径
    @returns {Dict[str, Anno]} - frame_name -> Anno
    """

    out: Dict[str, Anno] = {}
    if not csv_path or not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            frame = str(row.get("frame", "")).strip()
            if frame == "":
                continue
            vis = int(str(row.get("visible", "0")).strip() or "0")
            xs = str(row.get("x", "")).strip()
            ys = str(row.get("y", "")).strip()
            x = float(xs) if xs != "" else None
            y = float(ys) if ys != "" else None
            if vis == 0:
                x, y = None, None
            out[frame] = Anno(visible=vis, x=x, y=y)
    return out


def save_csv(out_csv: str, frame_names: List[str], annos: Dict[str, Anno]) -> None:
    """
    保存点标注 CSV。

    @param {str} out_csv - 输出路径
    @param {List[str]} frame_names - 帧名列表（按顺序输出）
    @param {Dict[str,Anno]} annos - 标注字典
    @returns {None}
    """

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "visible"])
        for name in frame_names:
            a = annos.get(name, Anno())
            if int(a.visible) == 1 and a.x is not None and a.y is not None:
                w.writerow([name, f"{a.x:.3f}", f"{a.y:.3f}", 1])
            else:
                w.writerow([name, "", "", 0])


def compute_display_scale(img_w: int, max_width: int) -> float:
    """
    根据 max_width 计算显示缩放比例（<=1）。

    @param {int} img_w - 原图宽
    @param {int} max_width - 最大显示宽
    @returns {float} - scale
    """

    if img_w <= 0:
        return 1.0
    if int(max_width) <= 0 or img_w <= int(max_width):
        return 1.0
    return float(max_width) / float(img_w)


def draw_overlay(img: "cv2.Mat", anno: Anno, idx: int, total: int, name: str) -> "cv2.Mat":
    """
    画标注与提示文本。

    @param {cv2.Mat} img - BGR 图像
    @param {Anno} anno - 当前帧标注
    @param {int} idx - 当前索引
    @param {int} total - 总数
    @param {str} name - 帧名
    @returns {cv2.Mat} - 绘制后的图
    """

    out = img.copy()
    text = f"[{idx+1}/{total}] {name}  visible={anno.visible}"
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if anno.visible == 1 and anno.x is not None and anno.y is not None:
        cv2.circle(out, (int(round(anno.x)), int(round(anno.y))), 6, (0, 255, 0), -1)
        cv2.putText(
            out,
            f"({anno.x:.1f},{anno.y:.1f})",
            (int(round(anno.x)) + 10, int(round(anno.y)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return out


def main() -> None:
    """
    主函数。

    @returns {None}
    """

    args = parse_args()
    inp = args.input
    out_csv = args.out_csv

    if not os.path.exists(inp):
        raise FileNotFoundError(f"输入不存在：{inp}")

    if is_video_file(inp):
        raise NotImplementedError(
            "当前版本先支持“帧目录”标注。请先用 extract_frames.py 抽帧，再用本工具标注。"
        )

    frames = list_frames_from_dir(inp)
    if len(frames) == 0:
        raise FileNotFoundError(f"帧目录为空：{inp}")

    frame_names = [p.name for p in frames]
    annos = load_csv(args.resume_csv) if args.resume_csv else load_csv(out_csv)

    idx = int(max(0, min(int(args.start), len(frames) - 1)))
    win = "annotate_points"
    state: Dict[str, object] = {"scale": 1.0, "idx": idx, "dirty": False}

    def set_dirty(v: bool) -> None:
        state["dirty"] = bool(v)

    def current_name() -> str:
        return frame_names[int(state["idx"])]

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        """
        鼠标回调：左键点击设置球心点（坐标会映射回原图）。

        @param {int} event - OpenCV event
        @param {int} x - 显示坐标 x
        @param {int} y - 显示坐标 y
        @param {int} flags - flags
        @param {object} param - param
        @returns {None}
        """

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        scale = float(state.get("scale", 1.0))
        ox = float(x) / max(scale, 1e-12)
        oy = float(y) / max(scale, 1e-12)
        name = current_name()
        annos[name] = Anno(visible=1, x=ox, y=oy)
        set_dirty(True)
        if bool(args.autosave):
            save_csv(out_csv, frame_names, annos)
            set_dirty(False)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        idx = int(state["idx"])
        fp = frames[idx]
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 无法读取：{fp}")
            state["idx"] = min(idx + 1, len(frames) - 1)
            continue

        name = fp.name
        anno = annos.get(name, Anno())

        disp = draw_overlay(img, anno, idx, len(frames), name)
        scale = compute_display_scale(disp.shape[1], int(args.max_width))
        state["scale"] = scale
        if scale < 1.0:
            disp = cv2.resize(disp, (int(disp.shape[1] * scale), int(disp.shape[0] * scale)))

        cv2.imshow(win, disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q / esc
            if bool(state.get("dirty", False)):
                print("[INFO] 你有未保存的修改，按 s 保存或再次按 q 退出。")
                key2 = cv2.waitKey(0) & 0xFF
                if key2 in (ord("s"),):
                    save_csv(out_csv, frame_names, annos)
                    set_dirty(False)
                    break
                if key2 in (ord("q"), 27):
                    break
            else:
                break
        elif key in (ord("n"), ord(" ")):  # next
            state["idx"] = min(idx + 1, len(frames) - 1)
        elif key == ord("p"):  # prev
            state["idx"] = max(idx - 1, 0)
        elif key == ord("s"):  # save
            save_csv(out_csv, frame_names, annos)
            set_dirty(False)
            print(f"[OK] 已保存：{out_csv}")
        elif key == ord("v"):  # toggle visible
            if anno.visible == 1:
                annos[name] = Anno(visible=0, x=None, y=None)
            else:
                # 变为可见但没点，先保持 None，等点击
                annos[name] = Anno(visible=1, x=anno.x, y=anno.y)
            set_dirty(True)
            if bool(args.autosave):
                save_csv(out_csv, frame_names, annos)
                set_dirty(False)
        elif key == ord("c"):  # clear current
            annos[name] = Anno(visible=0, x=None, y=None)
            set_dirty(True)
            if bool(args.autosave):
                save_csv(out_csv, frame_names, annos)
                set_dirty(False)
        else:
            # 未识别按键：继续等待
            pass

    cv2.destroyAllWindows()
    if bool(state.get("dirty", False)):
        save_csv(out_csv, frame_names, annos)
        print(f"[OK] 已自动保存：{out_csv}")


if __name__ == "__main__":
    main()


