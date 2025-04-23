#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件配对脚本
此脚本用于扫描一个包含两个子文件夹的目录，
找出两个子文件夹中名称对应的文件，并将配对结果保存到txt文件中。
确保jpg文件在前面，png文件在后面，并使用全路径。
"""

import os
import argparse
from pathlib import Path


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='将两个文件夹中对应的文件名配对并保存到txt中')
    parser.add_argument('input_dir', type=str, help='包含两个子文件夹的输入目录路径')
    parser.add_argument('--output', type=str, default='file_pairs.txt', help='输出的txt文件路径 (默认: file_pairs.txt)')
    parser.add_argument('--extensions', type=str, default='jpg,png', 
                        help='指定两个子文件夹中文件的扩展名，格式为"jpg,png"，表示第一个文件夹找jpg文件，第二个文件夹找png文件 (默认: jpg,png)')
    return parser.parse_args()


def find_subfolders(input_dir):
    """查找输入目录下的所有子文件夹"""
    subfolders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    return subfolders


def get_filename_without_extension(file_path):
    """获取不带扩展名的文件名"""
    return os.path.splitext(os.path.basename(file_path))[0]


def pair_files(input_dir, subfolders, extensions=None):
    """
    配对两个子文件夹中的文件，确保jpg在前面，png在后面
    
    Args:
        input_dir: 输入目录路径
        subfolders: 子文件夹列表
        extensions: 可选的两个子文件夹中文件的扩展名，格式为[ext1, ext2]，默认为['.jpg', '.png']
    
    Returns:
        pairs: 配对结果的列表，每个元素为(jpg文件全路径, png文件全路径)的元组
    """
    if len(subfolders) != 2:
        raise ValueError(f"需要恰好两个子文件夹，但找到了 {len(subfolders)} 个")
    
    # 设置默认扩展名为jpg和png
    if extensions is None:
        extensions = ['.jpg', '.png']
    
    # 获取每个文件夹的完整路径
    folder1 = os.path.join(input_dir, subfolders[0])
    folder2 = os.path.join(input_dir, subfolders[1])
    
    # 获取所有文件
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    
    # 用于存储jpg和png文件的字典
    jpg_files = {}
    png_files = {}
    
    # 处理第一个文件夹中的文件
    for file in files1:
        file_path = os.path.join(folder1, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            base_name = get_filename_without_extension(file)
            
            if ext == '.jpg':
                jpg_files[base_name] = os.path.abspath(file_path)
            elif ext == '.png':
                png_files[base_name] = os.path.abspath(file_path)
    
    # 处理第二个文件夹中的文件
    for file in files2:
        file_path = os.path.join(folder2, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            base_name = get_filename_without_extension(file)
            
            if ext == '.jpg':
                jpg_files[base_name] = os.path.abspath(file_path)
            elif ext == '.png':
                png_files[base_name] = os.path.abspath(file_path)
    
    # 找到共同的基础文件名
    common_base_names = set(jpg_files.keys()) & set(png_files.keys())
    
    # 创建配对，确保jpg在前面，png在后面
    pairs = []
    for base_name in sorted(common_base_names):
        pair = (jpg_files[base_name], png_files[base_name])
        pairs.append(pair)
    
    return pairs


def save_pairs_to_txt(pairs, output_path):
    """将配对结果保存到txt文件，使用全路径"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for jpg_path, png_path in pairs:
            f.write(f"{jpg_path} {png_path}\n")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 确保输入目录存在
    if not os.path.isdir(args.input_dir):
        print(f"错误：输入目录 '{args.input_dir}' 不存在")
        return 1
    
    try:
        # 查找子文件夹
        subfolders = find_subfolders(args.input_dir)
        if len(subfolders) != 2:
            print(f"错误：需要恰好两个子文件夹，但找到了 {len(subfolders)} 个")
            return 1
        
        print(f"找到两个子文件夹: {subfolders[0]} 和 {subfolders[1]}")
        
        # 处理扩展名参数
        extensions = None
        if args.extensions:
            ext_parts = args.extensions.split(',')
            if len(ext_parts) == 2:
                extensions = [f".{ext.strip().lower()}" for ext in ext_parts]
                print(f"使用指定的文件扩展名: {extensions[0]} 和 {extensions[1]}")
        
        # 配对文件，确保jpg在前，png在后
        pairs = pair_files(args.input_dir, subfolders, extensions)
        
        # 保存配对结果到txt文件
        save_pairs_to_txt(pairs, args.output)
        
        print(f"成功找到 {len(pairs)} 对匹配的文件")
        print(f"配对结果已保存到: {args.output}")
        print(f"注意: 文件对格式为 'jpg全路径 png全路径'")
        
    except Exception as e:
        print(f"错误：{str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 