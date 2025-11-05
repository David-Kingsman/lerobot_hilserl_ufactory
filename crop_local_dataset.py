#!/usr/bin/env python3
"""
本地数据集裁剪工具 - 不需要Hugging Face连接
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 添加lerobot到Python路径
sys.path.insert(0, '/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src')

from lerobot.rl.crop_dataset_roi import get_image_from_lerobot_dataset, crop_and_resize_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def load_local_dataset(root_path):
    """加载本地数据集，绕过Hugging Face检查"""
    try:
        # 尝试使用一个假的repo_id，但强制使用本地数据
        dataset = LeRobotDataset(repo_id="local/dataset", root=root_path)
        return dataset
    except Exception as e:
        print(f"无法加载数据集: {e}")
        print("请确保数据集目录结构正确，包含meta/episodes/目录")
        return None

def main():
    parser = argparse.ArgumentParser(description="裁剪本地LeRobot数据集的ROI区域")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="本地数据集的根目录路径"
    )
    parser.add_argument(
        "--crop-params-path",
        type=str,
        default=None,
        help="包含ROI参数的JSON文件路径"
    )
    parser.add_argument(
        "--new-root",
        type=str,
        default=None,
        help="新数据集的保存路径"
    )
    
    args = parser.parse_args()
    
    print(f"加载本地数据集: {args.root}")
    
    # 检查数据集目录是否存在
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"错误: 数据集目录不存在: {root_path}")
        return
    
    # 检查必要的目录结构
    meta_dir = root_path / "meta"
    episodes_dir = meta_dir / "episodes"
    data_dir = root_path / "data"
    
    if not episodes_dir.exists():
        print(f"错误: episodes目录不存在: {episodes_dir}")
        return
        
    if not data_dir.exists():
        print(f"错误: data目录不存在: {data_dir}")
        return
    
    print("数据集目录结构检查通过")
    
    # 如果没有指定裁剪参数，就启动交互式ROI选择
    if args.crop_params_path is None:
        print("启动交互式ROI选择...")
        
        # 这里需要实现交互式选择逻辑
        # 暂时使用默认参数
        crop_params = {
            "observation.images.realsense": [0, 0, 480, 640],
            "observation.images.webcam_1": [0, 0, 480, 640]
        }
        print("使用默认裁剪参数（无裁剪）")
    else:
        with open(args.crop_params_path, 'r') as f:
            crop_params = json.load(f)
        print(f"从文件加载裁剪参数: {args.crop_params_path}")
    
    print("裁剪参数:", crop_params)
    
    # 设置输出路径
    if args.new_root is None:
        new_root = str(root_path) + "_cropped"
    else:
        new_root = args.new_root
    
    print(f"输出路径: {new_root}")
    
    print("开始处理数据集...")
    print("注意: 由于绕过了Hugging Face检查，某些功能可能受限")
    print("建议先备份原始数据集")

if __name__ == "__main__":
    main()
