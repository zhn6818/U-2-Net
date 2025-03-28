#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
from PIL import Image
import shutil
from tqdm import tqdm
import argparse

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def slide_window_crop(image, mask, window_size=768, stride=None):
    """
    使用滑动窗口裁剪图像和掩码
    
    参数:
        image: PIL图像对象，输入图像
        mask: PIL图像对象，输入掩码
        window_size: 窗口大小，默认为768
        stride: 滑动步长，如果为None则自动计算为在横竖方向各滑动2次
    
    返回:
        crops: 裁剪后的图像列表
        mask_crops: 裁剪后的掩码列表
    """
    width, height = image.size
    
    # 如果stride为None，则计算stride使得横竖方向各滑动2次
    if stride is None:
        stride_x = (width - window_size) // 1
        stride_y = (height - window_size) // 1
    else:
        stride_x = stride
        stride_y = stride
    
    crops = []
    mask_crops = []
    
    # 左上角
    crops.append(image.crop((0, 0, window_size, window_size)))
    mask_crops.append(mask.crop((0, 0, window_size, window_size)))
    
    # 右上角
    crops.append(image.crop((stride_x, 0, stride_x + window_size, window_size)))
    mask_crops.append(mask.crop((stride_x, 0, stride_x + window_size, window_size)))
    
    # 左下角
    crops.append(image.crop((0, stride_y, window_size, stride_y + window_size)))
    mask_crops.append(mask.crop((0, stride_y, window_size, stride_y + window_size)))
    
    # 右下角
    crops.append(image.crop((stride_x, stride_y, stride_x + window_size, stride_y + window_size)))
    mask_crops.append(mask.crop((stride_x, stride_y, stride_x + window_size, stride_y + window_size)))
    
    return crops, mask_crops

def augment_dataset(src_dir, dst_dir, window_size=768):
    """
    扩充数据集，通过滑窗方式从原始图像中裁剪出较小的图像
    
    参数:
        src_dir: 源数据集目录
        dst_dir: 目标数据集目录
        window_size: 裁剪窗口大小
    """
    # 确保目标目录存在
    img_dst_dir = os.path.join(dst_dir, 'im_aug')
    mask_dst_dir = os.path.join(dst_dir, 'gt_aug')
    ensure_dir(img_dst_dir)
    ensure_dir(mask_dst_dir)
    
    # 获取所有图像路径
    img_paths = glob.glob(os.path.join(src_dir, 'im_aug', '*.jpg'))
    
    # 处理每个图像和对应的掩码
    for img_path in tqdm(img_paths, desc="处理图像"):
        # 获取文件名（不包含扩展名）
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(src_dir, 'gt_aug', f"{base_name}.png")
        
        # 检查掩码是否存在
        if not os.path.exists(mask_path):
            print(f"警告：找不到掩码 {mask_path}")
            continue
        
        # 读取图像和掩码
        try:
            image = Image.open(img_path)
            mask = Image.open(mask_path)
            
            # 确保图像和掩码尺寸一致
            if image.size != mask.size:
                print(f"警告：图像和掩码尺寸不一致 {img_path}")
                continue
            
            # 检查图像尺寸是否足够大
            if image.size[0] < window_size or image.size[1] < window_size:
                print(f"警告：图像尺寸小于滑窗尺寸 {img_path}")
                continue
            
            # 裁剪图像和掩码
            crops, mask_crops = slide_window_crop(image, mask, window_size)
            
            # 保存裁剪后的图像和掩码
            for i, (crop, mask_crop) in enumerate(zip(crops, mask_crops)):
                crop_name = f"{base_name}_crop{i}.jpg"
                mask_crop_name = f"{base_name}_crop{i}.png"
                
                crop.save(os.path.join(img_dst_dir, crop_name))
                mask_crop.save(os.path.join(mask_dst_dir, mask_crop_name))
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='通过滑窗方式扩充数据集')
    parser.add_argument('--src_dir', type=str, default='U2net_data/train_data',
                        help='源数据集目录')
    parser.add_argument('--dst_dir', type=str, default='U2net_data768/train_data',
                        help='目标数据集目录')
    parser.add_argument('--window_size', type=int, default=768,
                        help='裁剪窗口大小')
    
    args = parser.parse_args()
    
    print(f"从 {args.src_dir} 创建扩充数据集到 {args.dst_dir}, 窗口大小: {args.window_size}")
    
    augment_dataset(args.src_dir, args.dst_dir, args.window_size)
    
    print("数据集扩充完成!")

if __name__ == '__main__':
    main() 