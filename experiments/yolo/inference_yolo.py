"""
YOLO推理脚本
支持图像和视频输入
"""

from __future__ import annotations

import os
import argparse
import cv2
from pathlib import Path
import sys
from typing import Optional


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO目标检测推理')
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径（.pt文件）')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像或视频路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径（图像：.jpg/.png，视频：.mp4/.avi）')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值（0-1）')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值（用于NMS）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备（cuda/cpu）')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--show', action='store_true',
                       help='显示结果（仅对图像有效）')
    parser.add_argument('--save-txt', action='store_true',
                       help='保存检测结果为YOLO格式的txt文件')
    parser.add_argument('--save-conf', action='store_true',
                       help='在保存的txt文件中包含置信度')
    parser.add_argument('--line-width', type=int, default=2,
                       help='边界框线条宽度')
    parser.add_argument('--hide-labels', action='store_true',
                       help='隐藏类别标签')
    parser.add_argument('--hide-conf', action='store_true',
                       help='隐藏置信度')

    parser.add_argument(
        '--yolov13-repo',
        type=str,
        default=None,
        help=(
            "可选：YOLOv13(iMoonLab) 仓库路径，用于加载包含自定义层(如 DSC3k2)的 ultralytics。"
            "例如：experiments/yolo/third_party/yolov13。若不传且本地存在该目录，将自动启用。"
        ),
    )
    
    return parser.parse_args()


def is_video_file(file_path):
    """
    判断文件是否为视频文件
    
    @param {str} file_path - 文件路径
    @returns {bool} - 是否为视频文件
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)


def is_image_file(file_path):
    """
    判断文件是否为图像文件
    
    @param {str} file_path - 文件路径
    @returns {bool} - 是否为图像文件
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def get_output_path(input_path, output_path=None, suffix='_detected'):
    """
    获取输出路径
    
    @param {str} input_path - 输入路径
    @param {str} output_path - 指定的输出路径
    @param {str} suffix - 后缀
    @returns {str} - 输出路径
    """
    if output_path:
        # 如果指定了输出路径，确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return output_path
    
    # 创建默认的output文件夹
    output_dir = str(Path(__file__).resolve().parent / 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_path)
    input_name = input_path.stem
    
    if is_video_file(str(input_path)):
        # 视频文件
        output_path = os.path.join(output_dir, f"{input_name}{suffix}.mp4")
    elif is_image_file(str(input_path)):
        # 图像文件
        output_path = os.path.join(output_dir, f"{input_name}{suffix}.jpg")
    else:
        # 默认
        output_path = os.path.join(output_dir, f"{input_name}{suffix}")
    
    return output_path


def setup_yolov13_ultralytics(yolov13_repo: Optional[str]) -> None:
    """
    优先使用 YOLOv13(iMoonLab) 仓库内置的 ultralytics，实现与 yolov13*.pt 权重的自定义层兼容。

    背景：YOLOv13 权重通常包含自定义模块（例如 `DSC3k2`），官方 pip 的 `ultralytics` 可能缺失该类，
    导致加载 `.pt` 时出现：
      AttributeError: Can't get attribute 'DSC3k2' on ultralytics.nn.modules.block

    策略：
    - 若用户显式指定 `--yolov13-repo` 且目录存在，则把该目录插入到 sys.path 最前，确保 import 优先命中仓库代码。
    - 若未指定，但本地存在 `experiments/yolo/third_party/yolov13`，自动启用。

    @param {Optional[str]} yolov13_repo - YOLOv13 仓库路径
    @returns {None}
    """

    # 自动发现默认路径
    repo = yolov13_repo
    if not repo:
        default_repo = Path(__file__).resolve().parent / "third_party" / "yolov13"
        if default_repo.exists():
            repo = str(default_repo)

    if not repo:
        return

    repo_path = Path(repo).expanduser().resolve()
    if not repo_path.exists():
        print(f"[WARN] --yolov13-repo 指定路径不存在: {repo_path}，将继续使用当前环境的 ultralytics。")
        return

    # 让仓库内 ultralytics 优先于 site-packages
    sys.path.insert(0, str(repo_path))


def load_yolo_model(weights_path: str, yolov13_repo: Optional[str]):
    """
    加载 YOLO 模型（兼容 YOLOv13 fork 的自定义层）。

    @param {str} weights_path - 权重路径（.pt）
    @param {Optional[str]} yolov13_repo - YOLOv13 仓库路径
    @returns {YOLO} - Ultralytics YOLO 模型实例
    """

    setup_yolov13_ultralytics(yolov13_repo)

    # 延迟导入，确保 sys.path 已生效
    from ultralytics import YOLO  # type: ignore
    import ultralytics.nn.modules.block as _block  # type: ignore

    # 预检：如果当前 ultralytics 缺失 DSC3k2，给出明确引导
    if not hasattr(_block, "DSC3k2"):
        raise RuntimeError(
            "当前环境的 ultralytics 缺少 YOLOv13 权重所需的自定义层 `DSC3k2`。\n"
            "你可以选择以下任一方案：\n"
            "1) 使用本工程已 clone 的 YOLOv13 仓库推理（推荐）：\n"
            "   python experiments/yolo/inference_yolo.py --yolov13-repo experiments/yolo/third_party/yolov13 --model <yolov13*.pt> --input <img>\n"
            "2) 在 conda 环境中安装 YOLOv13 仓库的 ultralytics（可选）：\n"
            "   conda run -n yolo pip uninstall -y ultralytics\n"
            "   conda run -n yolo pip install -e experiments/yolo/third_party/yolov13\n"
            "然后重新运行推理。\n"
        )

    return YOLO(weights_path)


def process_image(model, image_path, output_path, args):
    """
    处理单张图像
    
    @param {YOLO} model - YOLO模型
    @param {str} image_path - 图像路径
    @param {str} output_path - 输出路径
    @param {argparse.Namespace} args - 参数
    """
    print(f'处理图像: {image_path}')
    
    # 推理
    results = model.predict(
        source=image_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        line_width=args.line_width,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        project=os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        name=os.path.basename(os.path.dirname(output_path)) if os.path.dirname(output_path) else 'predict',
        exist_ok=True
    )
    
    # 获取保存的路径
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'save_dir'):
            saved_path = os.path.join(result.save_dir, os.path.basename(image_path))
            if os.path.exists(saved_path):
                # 如果输出路径不同，复制文件
                if saved_path != output_path:
                    import shutil
                    shutil.copy2(saved_path, output_path)
                    print(f'结果已保存到: {output_path}')
                else:
                    print(f'结果已保存到: {saved_path}')
        
        # 显示结果
        if args.show:
            # 读取并显示结果图像
            result_img = result.plot()
            cv2.imshow('Detection Result', result_img)
            print('按任意键关闭窗口...')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 打印检测信息
        if result.boxes is not None:
            print(f'检测到 {len(result.boxes)} 个目标:')
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id] if cls_id < len(model.names) else f'class_{cls_id}'
                print(f'  {i+1}. {cls_name}: {conf:.2f}')


def process_video(model, video_path, output_path, args):
    """
    处理视频文件
    
    @param {YOLO} model - YOLO模型
    @param {str} video_path - 视频路径
    @param {str} output_path - 输出路径
    @param {argparse.Namespace} args - 参数
    """
    print(f'处理视频: {video_path}')
    
    # 打开视频获取信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f'视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧')
    print('开始处理视频，请稍候...')
    
    # 推理
    results = model.predict(
        source=video_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        line_width=args.line_width,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        project=os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        name=os.path.basename(os.path.dirname(output_path)) if os.path.dirname(output_path) else 'predict',
        exist_ok=True
    )
    
    # 获取保存的路径
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'save_dir'):
            # 查找保存的视频文件
            save_dir = result.save_dir
            for file in os.listdir(save_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    saved_path = os.path.join(save_dir, file)
                    if os.path.exists(saved_path):
                        # 如果输出路径不同，复制文件
                        if saved_path != output_path:
                            import shutil
                            shutil.copy2(saved_path, output_path)
                            print(f'结果已保存到: {output_path}')
                        else:
                            print(f'结果已保存到: {saved_path}')
                        break
    
    print('视频处理完成！')


def process_directory(model, dir_path, output_dir, args):
    """
    处理目录中的所有图像和视频
    
    @param {YOLO} model - YOLO模型
    @param {str} dir_path - 目录路径
    @param {str} output_dir - 输出目录
    @param {argparse.Namespace} args - 参数
    """
    print(f'处理目录: {dir_path}')
    
    # 获取所有图像和视频文件
    image_files = []
    video_files = []
    
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
        image_files.extend(Path(dir_path).glob(f'*{ext}'))
        image_files.extend(Path(dir_path).glob(f'*{ext.upper()}'))
    
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']:
        video_files.extend(Path(dir_path).glob(f'*{ext}'))
        video_files.extend(Path(dir_path).glob(f'*{ext.upper()}'))
    
    print(f'找到 {len(image_files)} 个图像文件，{len(video_files)} 个视频文件')
    
    # 处理图像
    for img_path in image_files:
        output_path = os.path.join(output_dir, f"{img_path.stem}_detected{img_path.suffix}")
        process_image(model, str(img_path), output_path, args)
    
    # 处理视频
    for vid_path in video_files:
        output_path = os.path.join(output_dir, f"{vid_path.stem}_detected.mp4")
        process_video(model, str(vid_path), output_path, args)


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件/目录不存在: {args.input}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    
    # 加载模型
    print(f'加载模型: {args.model}')
    model = load_yolo_model(args.model, args.yolov13_repo)
    print(f'模型类别: {list(model.names.values())}')
    
    # 确定输出路径
    output_path = get_output_path(args.input, args.output)
    
    # 判断输入类型并处理
    if os.path.isdir(args.input):
        # 处理目录
        output_dir = args.output if args.output else 'output'
        os.makedirs(output_dir, exist_ok=True)
        process_directory(model, args.input, output_dir, args)
    elif is_video_file(args.input):
        # 处理视频
        process_video(model, args.input, output_path, args)
    elif is_image_file(args.input):
        # 处理图像
        process_image(model, args.input, output_path, args)
    else:
        # 尝试作为图像或视频处理
        print(f'警告: 无法识别文件类型，尝试作为图像处理...')
        process_image(model, args.input, output_path, args)


if __name__ == '__main__':
    main()

