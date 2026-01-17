"""
基于TTNet的小目标检测推理脚本
支持图像和视频输入
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from ttnet_small_object.models import TTNetSmallObject


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='TTNet小目标检测推理')
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像或视频路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径（图像：.jpg/.png，视频：.mp4/.avi）')
    parser.add_argument('--input-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='置信度阈值（建议0.1-0.3）')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                       help='NMS阈值')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备')
    parser.add_argument('--show', action='store_true',
                       help='显示结果')
    
    return parser.parse_args()


def load_model(model_path, device, input_size=640):
    """
    加载模型
    
    @param {str} model_path - 模型路径
    @param {str} device - 设备
    @param {int} input_size - 输入尺寸
    @returns {nn.Module} - 模型
    """
    # 加载检查点（weights_only=False 以支持加载包含 argparse.Namespace 的检查点）
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # 兼容旧版本的 PyTorch（< 2.6）
        checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    use_local = False
    if 'args' in checkpoint:
        if hasattr(checkpoint['args'], 'use_local'):
            use_local = checkpoint['args'].use_local
        elif isinstance(checkpoint['args'], dict):
            use_local = checkpoint['args'].get('use_local', False)
    
    # 创建模型
    model = TTNetSmallObject(
        num_classes=1,
        input_size=input_size,
        use_local=use_local
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, input_size=640):
    """
    预处理图像
    
    @param {str} image_path - 图像路径
    @param {int} input_size - 输入尺寸
    @returns {tuple} - (tensor, original_image, scale)
    """
    # 读取图像
    image = cv2.imread(image_path)
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 记录原始尺寸
    original_h, original_w = image.shape[:2]
    
    # 调整尺寸
    image_resized = cv2.resize(image, (input_size, input_size))
    
    # 转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(Image.fromarray(image_resized)).unsqueeze(0)
    
    # 计算缩放比例
    scale_x = original_w / input_size
    scale_y = original_h / input_size
    
    return image_tensor, original_image, (scale_x, scale_y)


def postprocess_output(
    predictions,
    conf_thresh=0.25,
    original_size=None,
    nms_thresh=0.5,
    max_candidates=200,
    max_detections=20,
):
    """
    后处理模型输出
    
    @param {dict} predictions - 模型预测结果
    @param {float} conf_thresh - 置信度阈值（降低默认值以提高检测率）
    @param {tuple} original_size - 原始图像尺寸 (width, height)
    @param {float} nms_thresh - NMS阈值
    @param {int} max_candidates - 最大候选点数量（按置信度从高到低截断，避免产生大量伪检）
    @param {int} max_detections - NMS后最多保留的检测数量
    @returns {list} - 检测框列表
    """
    global_pred = predictions.get('global', {})
    
    cls_output = global_pred.get('cls', None)
    loc_output = global_pred.get('loc', None)
    size_output = global_pred.get('size', None)
    
    if cls_output is None:
        return []
    
    # 应用sigmoid得到概率
    cls_prob = torch.sigmoid(cls_output)
    
    # 获取输入尺寸
    input_h, input_w = cls_prob.shape[-2:]
    
    # 计算缩放比例
    if original_size:
        scale_x = original_size[0] / input_w
        scale_y = original_size[1] / input_h
    else:
        scale_x = scale_y = 1.0
        original_size = (input_w, input_h)
    
    # 找到置信度高的位置（使用更低的阈值）
    # 首先找到局部最大值，避免检测到太多重复的点
    kernel_size = 5
    max_pool = torch.nn.functional.max_pool2d(
        cls_prob, kernel_size=kernel_size, stride=1, 
        padding=kernel_size // 2
    )
    
    # 找到局部最大值且超过阈值的位置
    mask = (cls_prob == max_pool) & (cls_prob > conf_thresh)
    
    boxes = []
    
    if mask.sum() > 0:
        # 获取高置信度位置的坐标
        y_coords, x_coords = torch.where(mask[0, 0])

        # 按置信度排序并截断候选点（TTNet类热力图很容易产生大量局部峰值）
        if y_coords.numel() > 0 and max_candidates is not None and max_candidates > 0:
            confs = cls_prob[0, 0, y_coords, x_coords]
            topk = min(int(max_candidates), int(confs.numel()))
            _, idx = torch.topk(confs, k=topk, largest=True, sorted=True)
            y_coords = y_coords[idx]
            x_coords = x_coords[idx]
        
        # 收集所有候选框
        candidates = []
        for y, x in zip(y_coords, x_coords):
            # 获取当前位置的置信度（作为权重）
            conf = cls_prob[0, 0, y, x].item()
            
            # 位置解码策略：
            # 训练时cls通常学习成“以目标中心为峰值的热力图”，所以峰值位置 (x,y) 本身就是最稳定的中心估计。
            # loc/size 分支在当前实现中并不一定与 cls_prob 共享相同的“高斯权重”语义，
            # 直接做 loc/conf 容易把中心点推到离谱的位置（你遇到的偏移问题基本就是这个）。
            #
            # 因此这里优先使用网格峰值坐标作为中心；如果 loc_output 解码出来更接近峰值，再用它作微调。
            grid_x_norm = (x.item() + 0.5) / float(input_w)
            grid_y_norm = (y.item() + 0.5) / float(input_h)

            x_center_norm = grid_x_norm
            y_center_norm = grid_y_norm

            if loc_output is not None and loc_output.shape[1] >= 2:
                loc_x = loc_output[0, 0, y, x].item()
                loc_y = loc_output[0, 1, y, x].item()

                # 两种常见语义候选：
                # 1) 直接预测归一化中心 (0~1)
                cand_raw_x = max(0.0, min(1.0, loc_x))
                cand_raw_y = max(0.0, min(1.0, loc_y))
                # 2) 预测的是(某种权重)*中心，需要除以 cls_prob 才能恢复
                if conf > 1e-6:
                    cand_div_x = max(0.0, min(1.0, loc_x / conf))
                    cand_div_y = max(0.0, min(1.0, loc_y / conf))
                else:
                    cand_div_x, cand_div_y = cand_raw_x, cand_raw_y

                # 选择与网格峰值更一致的那个（越一致越不容易出现“离谱偏移”）
                dist_raw = abs(cand_raw_x - grid_x_norm) + abs(cand_raw_y - grid_y_norm)
                dist_div = abs(cand_div_x - grid_x_norm) + abs(cand_div_y - grid_y_norm)
                best_x, best_y = (cand_raw_x, cand_raw_y) if dist_raw <= dist_div else (cand_div_x, cand_div_y)
                best_dist = min(dist_raw, dist_div)

                # 关键兜底：如果loc解码与峰值差得太远，说明语义不一致（或模型没学好），强制回退到峰值网格坐标
                # 经验阈值：L1距离>0.1（约等于横纵各偏移~0.05）就不可信
                if best_dist <= 0.1:
                    x_center_norm, y_center_norm = best_x, best_y

            # 转换为像素坐标
            x_center = x_center_norm * original_size[0]
            y_center = y_center_norm * original_size[1]
            
            if size_output is not None and size_output.shape[1] >= 2:
                size_w = size_output[0, 0, y, x].item()
                size_h = size_output[0, 1, y, x].item()

                # 尺寸同样提供 raw / raw/conf 两种候选；更保守地取“更大且合理”的那个，避免尺寸被压得太小
                cand_raw_w = max(0.0, size_w)
                cand_raw_h = max(0.0, size_h)
                if conf > 1e-6:
                    cand_div_w = max(0.0, size_w / conf)
                    cand_div_h = max(0.0, size_h / conf)
                else:
                    cand_div_w, cand_div_h = cand_raw_w, cand_raw_h

                width_norm = max(cand_raw_w, cand_div_w)
                height_norm = max(cand_raw_h, cand_div_h)
                if width_norm <= 0 or height_norm <= 0:
                    width_norm = height_norm = 0.02
                
                # 限制在合理范围内
                width_norm = max(0.01, min(0.5, width_norm))  # 至少1%，最多50%
                height_norm = max(0.01, min(0.5, height_norm))
                
                # 转换为像素尺寸
                width = width_norm * original_size[0]
                height = height_norm * original_size[1]
            else:
                # 使用默认尺寸（小目标的典型尺寸）
                width = height = min(original_size) * 0.05  # 图像尺寸的5%
            
            # 确保尺寸合理
            width = max(10, min(width, original_size[0] * 0.2))  # 至少10像素，最多20%图像宽度
            height = max(10, min(height, original_size[1] * 0.2))  # 至少10像素，最多20%图像高度
            
            # 转换为左上角和右下角坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, original_size[0] - 1))
            y1 = max(0, min(y1, original_size[1] - 1))
            x2 = max(0, min(x2, original_size[0] - 1))
            y2 = max(0, min(y2, original_size[1] - 1))
            
            # 确保框有效
            if x2 > x1 and y2 > y1:
                conf = cls_prob[0, 0, y, x].item()
                candidates.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': 0  # Ball
                })
        
        # 非极大值抑制（NMS）
        if len(candidates) > 0:
            boxes = nms(candidates, nms_thresh)
            if max_detections is not None and max_detections > 0:
                boxes = boxes[: int(max_detections)]
        else:
            boxes = []
    
    return boxes


def nms(boxes, iou_thresh=0.5):
    """
    非极大值抑制
    
    @param {list} boxes - 检测框列表
    @param {float} iou_thresh - IoU阈值
    @returns {list} - 抑制后的检测框列表
    """
    if len(boxes) == 0:
        return []
    
    # 按置信度排序
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while boxes:
        # 保留置信度最高的框
        current = boxes.pop(0)
        keep.append(current)
        
        # 移除与当前框IoU过高的框
        boxes = [box for box in boxes if iou(current['bbox'], box['bbox']) < iou_thresh]
    
    return keep


def iou(box1, box2):
    """
    计算两个框的IoU
    
    @param {list} box1 - 框1 [x1, y1, x2, y2]
    @param {list} box2 - 框2 [x1, y1, x2, y2]
    @returns {float} - IoU值
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def draw_detections(image, boxes):
    """
    在图像上绘制检测结果
    
    @param {numpy.ndarray} image - 图像
    @param {list} boxes - 检测框列表
    @returns {numpy.ndarray} - 绘制后的图像
    """
    result_image = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        conf = box['conf']
        
        # 绘制边界框
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"Ball: {conf:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image


def is_video_file(file_path):
    """
    判断文件是否为视频文件
    
    @param {str} file_path - 文件路径
    @returns {bool} - 是否为视频文件
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)


def process_video(model, video_path, output_path, device, input_size, conf_thresh, nms_thresh=0.5, show=False):
    """
    处理视频文件
    
    @param {nn.Module} model - 模型
    @param {str} video_path - 视频路径
    @param {str} output_path - 输出路径
    @param {str} device - 设备
    @param {int} input_size - 输入尺寸
    @param {float} conf_thresh - 置信度阈值
    @param {float} nms_thresh - NMS阈值
    @param {bool} show - 是否显示处理过程
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧')
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 处理每一帧
    frame_count = 0
    with tqdm(total=total_frames, desc='处理视频') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_h, original_w = frame_rgb.shape[:2]
            
            # 调整尺寸
            frame_resized = cv2.resize(frame_rgb, (input_size, input_size))
            
            # 转换为tensor
            frame_tensor = transform(Image.fromarray(frame_resized)).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                predictions = model(frame_tensor)
            
            # 后处理
            boxes = postprocess_output(
                predictions, 
                conf_thresh, 
                original_size=(original_w, original_h),
                nms_thresh=nms_thresh
            )
            
            # 绘制检测结果
            result_frame = draw_detections(frame, boxes)
            
            # 写入视频
            out.write(result_frame)
            
            # 显示（如果启用）
            if show:
                cv2.imshow('Video Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            pbar.update(1)
    
    # 释放资源
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f'处理完成: {frame_count} 帧，结果保存到 {output_path}')


def process_image(model, image_path, output_path, device, input_size, conf_thresh, nms_thresh=0.5, show=False):
    """
    处理图像文件
    
    @param {nn.Module} model - 模型
    @param {str} image_path - 图像路径
    @param {str} output_path - 输出路径
    @param {str} device - 设备
    @param {int} input_size - 输入尺寸
    @param {float} conf_thresh - 置信度阈值
    @param {float} nms_thresh - NMS阈值
    @param {bool} show - 是否显示结果
    """
    # 预处理图像
    image_tensor, original_image, _ = preprocess_image(image_path, input_size)
    original_h, original_w = original_image.shape[:2]
    
    # 推理
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)
    
    # 后处理
    boxes = postprocess_output(
        predictions, 
        conf_thresh, 
        original_size=(original_w, original_h),
        nms_thresh=nms_thresh
    )
    
    print(f'检测到 {len(boxes)} 个目标')
    
    # 绘制结果
    result_image = draw_detections(original_image, boxes)
    
    # 保存结果
    cv2.imwrite(output_path, result_image)
    print(f'结果已保存到: {output_path}')
    
    # 显示结果
    if show:
        cv2.imshow('Detection Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    # 确定输出路径
    if args.output is None:
        # 创建output文件夹
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取输入文件名（不含路径）
        input_basename = os.path.basename(args.input)
        input_name, input_ext = os.path.splitext(input_basename)
        
        if is_video_file(args.input):
            args.output = os.path.join(output_dir, f"{input_name}_detected.mp4")
        else:
            args.output = os.path.join(output_dir, f"{input_name}_detected.jpg")
    else:
        # 如果指定了输出路径，确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f'加载模型: {args.model}')
    model = load_model(args.model, device, args.input_size)
    
    # 获取NMS阈值
    nms_thresh = getattr(args, 'nms_thresh', 0.5)
    
    # 判断输入类型并处理
    if is_video_file(args.input):
        print(f'处理视频: {args.input}')
        process_video(
            model, 
            args.input, 
            args.output, 
            device, 
            args.input_size, 
            args.conf_thresh,
            nms_thresh,
            args.show
        )
    else:
        print(f'处理图像: {args.input}')
        process_image(
            model, 
            args.input, 
            args.output, 
            device, 
            args.input_size, 
            args.conf_thresh,
            nms_thresh,
            args.show
        )


if __name__ == '__main__':
    main()

