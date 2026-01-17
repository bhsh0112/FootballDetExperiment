"""
数据筛选工具
筛选数据集中包含Ball（类别0）的数据，删除不包含Ball的数据
"""

import os
import glob
from pathlib import Path

# Ball的类别ID（根据data.yaml，Ball是第一个类别，ID为0）
BALL_CLASS_ID = 0

def check_contains_ball(label_file_path):
    """
    检查标签文件是否包含Ball（类别0）
    
    @param {str} label_file_path - 标签文件路径
    @returns {bool} - 如果包含Ball返回True，否则返回False
    """
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                # YOLO格式：class_id x_center y_center width height
                parts = line.split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if class_id == BALL_CLASS_ID:
                        return True
        return False
    except Exception as e:
        print(f"读取标签文件 {label_file_path} 时出错: {e}")
        return False

def get_image_path_from_label(label_path, image_extensions=['.jpg', '.jpeg', '.png']):
    """
    根据标签文件路径获取对应的图片文件路径
    
    @param {str} label_path - 标签文件路径
    @param {list} image_extensions - 图片文件扩展名列表
    @returns {str|None} - 图片文件路径，如果找不到返回None
    """
    label_dir = os.path.dirname(label_path)
    # 将labels目录替换为images目录
    image_dir = label_dir.replace('/labels', '/images').replace('\\labels', '\\images')
    
    # 获取标签文件名（不含扩展名）
    label_basename = os.path.splitext(os.path.basename(label_path))[0]
    
    # 尝试不同的图片扩展名
    for ext in image_extensions:
        image_path = os.path.join(image_dir, label_basename + ext)
        if os.path.exists(image_path):
            return image_path
    
    return None

def preview_dataset(split_name, base_dir='.'):
    """
    预览指定数据集分割的统计信息（不执行删除）
    
    @param {str} split_name - 数据集分割名称（train/valid/test）
    @param {str} base_dir - 项目根目录
    @returns {dict} - 包含统计信息的字典
    """
    labels_dir = os.path.join(base_dir, split_name, 'labels')
    images_dir = os.path.join(base_dir, split_name, 'images')
    
    if not os.path.exists(labels_dir):
        print(f"警告: 标签目录不存在: {labels_dir}")
        return None
    
    if not os.path.exists(images_dir):
        print(f"警告: 图片目录不存在: {images_dir}")
        return None
    
    # 获取所有标签文件
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    total_files = len(label_files)
    
    if total_files == 0:
        print(f"{split_name}: 没有找到标签文件")
        return None
    
    kept_count = 0
    to_delete_count = 0
    
    for label_file in label_files:
        if check_contains_ball(label_file):
            kept_count += 1
        else:
            to_delete_count += 1
    
    stats = {
        'split': split_name,
        'total': total_files,
        'kept': kept_count,
        'to_delete': to_delete_count,
        'keep_rate': kept_count / total_files * 100 if total_files > 0 else 0
    }
    
    return stats

def filter_dataset(split_name, base_dir='.', dry_run=False):
    """
    筛选指定数据集分割（train/valid/test）中包含Ball的数据
    
    @param {str} split_name - 数据集分割名称（train/valid/test）
    @param {str} base_dir - 项目根目录
    @param {bool} dry_run - 如果为True，只预览不执行删除
    """
    labels_dir = os.path.join(base_dir, split_name, 'labels')
    images_dir = os.path.join(base_dir, split_name, 'images')
    
    if not os.path.exists(labels_dir):
        print(f"警告: 标签目录不存在: {labels_dir}")
        return
    
    if not os.path.exists(images_dir):
        print(f"警告: 图片目录不存在: {images_dir}")
        return
    
    # 获取所有标签文件
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    total_files = len(label_files)
    
    if total_files == 0:
        print(f"{split_name}: 没有找到标签文件")
        return
    
    print(f"\n处理 {split_name} 数据集...")
    print(f"总共找到 {total_files} 个标签文件")
    
    kept_count = 0
    deleted_count = 0
    
    for label_file in label_files:
        # 检查是否包含Ball
        if check_contains_ball(label_file):
            # 保留该文件
            kept_count += 1
        else:
            if dry_run:
                # 预览模式，只计数不删除
                deleted_count += 1
            else:
                # 删除标签文件和对应的图片文件
                try:
                    # 删除标签文件
                    os.remove(label_file)
                    
                    # 删除对应的图片文件
                    image_path = get_image_path_from_label(label_file)
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                        deleted_count += 1
                    else:
                        print(f"警告: 找不到对应的图片文件: {label_file}")
                        deleted_count += 1
                except Exception as e:
                    print(f"删除文件时出错 {label_file}: {e}")
    
    mode_text = "（预览模式）" if dry_run else ""
    print(f"{split_name} 数据集处理完成{mode_text}:")
    print(f"  - 保留: {kept_count} 个文件（包含Ball）")
    print(f"  - {'将删除' if dry_run else '已删除'}: {deleted_count} 个文件（不包含Ball）")
    print(f"  - 保留率: {kept_count/total_files*100:.2f}%")

def main():
    """
    主函数：筛选所有数据集分割中包含Ball的数据
    """
    print("=" * 60)
    print("数据筛选工具 - 筛选包含Ball的数据")
    print("=" * 60)
    print(f"Ball类别ID: {BALL_CLASS_ID}")
    
    # 处理所有数据集分割
    splits = ['train', 'valid', 'test']
    
    # 先预览统计信息
    print("\n【预览模式】正在统计数据集信息...")
    print("-" * 60)
    
    total_stats = {'total': 0, 'kept': 0, 'to_delete': 0}
    
    for split in splits:
        stats = preview_dataset(split)
        if stats:
            print(f"\n{split.upper()} 数据集:")
            print(f"  - 总文件数: {stats['total']}")
            print(f"  - 包含Ball: {stats['kept']} 个")
            print(f"  - 不包含Ball: {stats['to_delete']} 个（将被删除）")
            print(f"  - 保留率: {stats['keep_rate']:.2f}%")
            
            total_stats['total'] += stats['total']
            total_stats['kept'] += stats['kept']
            total_stats['to_delete'] += stats['to_delete']
    
    print("\n" + "-" * 60)
    print("总计:")
    print(f"  - 总文件数: {total_stats['total']}")
    print(f"  - 将保留: {total_stats['kept']} 个文件")
    print(f"  - 将删除: {total_stats['to_delete']} 个文件")
    if total_stats['total'] > 0:
        print(f"  - 保留率: {total_stats['kept']/total_stats['total']*100:.2f}%")
    print("=" * 60)
    
    print("\n⚠️  警告: 此操作将永久删除不包含Ball的图片和标签文件！")
    
    # 确认操作
    confirm = input("\n是否继续执行删除操作？(yes/no): ").strip().lower()
    if confirm not in ['yes', 'y', '是']:
        print("操作已取消")
        return
    
    # 执行删除操作
    print("\n【执行模式】开始删除不包含Ball的数据...")
    print("-" * 60)
    
    for split in splits:
        filter_dataset(split, dry_run=False)
    
    print("\n" + "=" * 60)
    print("所有数据集处理完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

