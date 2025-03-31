import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
from skimage import io, transform, color, filters
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import scipy.ndimage as ndimage

from model import U2NET
from model import U2NETP
from data_loader import MultiChannelToTensorLab, RescaleT

# --------- 参数解析 ---------
def get_args():
    parser = argparse.ArgumentParser(description='U2NET多通道分割模型推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--input_dir', type=str, default='test_images', help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='test_results', help='输出结果目录')
    parser.add_argument('--gt_dir', type=str, default='/Volumes/data1/JH/projects/ttprocess/train/masks', 
                        help='真实标签(Ground Truth)目录')
    parser.add_argument('--model_name', type=str, default='u2net', choices=['u2net', 'u2netp'], help='模型名称')
    parser.add_argument('--num_classes', type=int, default=2, help='分割类别数量')
    parser.add_argument('--input_size', type=int, default=512, help='输入图像大小')
    parser.add_argument('--flag', type=int, default=0, choices=[0, 1, 2], 
                       help='颜色空间标志，0: RGB, 1: Lab, 2: RGB+Lab')
    parser.add_argument('--boundary_thickness', type=int, default=3, 
                       help='边界线条粗细')
    parser.add_argument('--boundary_only', action='store_true',
                       help='是否只输出边界图')
    parser.add_argument('--mode', type=int, default=1, choices=[1, 2], 
                       help='模式选择: 1=完整处理所有图像, 2=只处理前景标签可视化')
    parser.add_argument('--single_image', type=str, default=None, 
                       help='指定单张图像路径(如果提供，则只处理这张图像)')
    parser.add_argument('--background_channel', type=int, default=0, 
                       help='指定背景对应的通道索引，默认为0')
    return parser.parse_args()

# --------- 图像预处理 ---------
def transform_image(image, target_size=512, flag=0, num_classes=2):
    """使用MultiChannelToTensorLab对输入图像进行预处理"""
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.concatenate([image, image, image], axis=2)
    
    # 创建一个假标签（推理时不需要真实标签）
    dummy_label = np.zeros((image.shape[0], image.shape[1], num_classes))
    
    # 准备样本字典
    sample = {
        'imidx': np.array([0]),
        'image': image,
        'label': dummy_label
    }
    
    # 首先使用RescaleT调整图像大小
    rescale = RescaleT(target_size)
    sample = rescale(sample)
    
    # 然后使用MultiChannelToTensorLab处理图像
    to_tensor = MultiChannelToTensorLab(flag=flag, num_channels=num_classes)
    tensor_sample = to_tensor(sample)
    
    # 提取处理后的图像张量并添加批次维度
    input_tensor = tensor_sample['image'].unsqueeze(0)
    
    return input_tensor

# --------- 模型加载 ---------
def load_model(model_path, model_name='u2net', num_classes=2, device='cpu'):
    """加载多通道分割模型"""
    # 创建模型
    if model_name == 'u2net':
        net = U2NET(3, num_classes)
    elif model_name == 'u2netp':
        net = U2NETP(3, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # 加载权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    return net

# --------- 图像推理 ---------
def predict(net, image, device='cpu'):
    """使用模型进行推理"""
    # 将图像移动到指定设备
    inputs = image.to(device)
    
    # 进行推理
    with torch.no_grad():
        d0, d1, d2, d3, d4, d5, d6 = net(inputs)
    
    # 取d0作为最终预测结果 (已经经过sigmoid)
    pred = d0.squeeze(0)
    pred = pred.cpu().numpy()
    
    return pred

# --------- 查找真实标签 ---------
def find_gt_mask(filename, gt_dir, num_classes):
    """
    在真实标签目录中查找对应的标签文件
    
    Args:
        filename: 输入图像的文件名（不含扩展名）
        gt_dir: 真实标签目录
        num_classes: 类别数量
    
    Returns:
        gt_masks: 多通道真实标签，如果找不到则返回None
    """
    # 检查不同可能的扩展名
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        gt_path = os.path.join(gt_dir, filename + ext)
        if os.path.exists(gt_path):
            print(f"找到对应GT: {gt_path}")
            gt_image = io.imread(gt_path)
            
            # 处理GT格式
            if len(gt_image.shape) == 2:  # 单通道标签
                if num_classes > 1:
                    # 多类别分割任务，将单通道转为多通道
                    gt_masks = []
                    for c in range(num_classes):
                        mask = (gt_image == c).astype(float)
                        gt_masks.append(mask)
                    return gt_masks
                else:
                    # 二分类任务
                    gt_mask = (gt_image > 0).astype(float)
                    return [gt_mask]
            elif len(gt_image.shape) == 3 and gt_image.shape[2] == num_classes:
                # 已经是正确格式的多通道标签
                gt_masks = []
                for c in range(num_classes):
                    gt_masks.append(gt_image[:, :, c])
                return gt_masks
            elif len(gt_image.shape) == 3 and gt_image.shape[2] == 3:
                # RGB格式标签，需要转换
                print(f"警告: GT标签是RGB格式，尝试转换为多通道格式")
                gt_masks = []
                for c in range(num_classes):
                    # 简单的颜色区分方法，可能需要根据实际情况调整
                    if c == 0:  # 假设红色通道表示第一类
                        mask = (gt_image[:, :, 0] > 128).astype(float)
                    elif c == 1:  # 假设绿色通道表示第二类
                        mask = (gt_image[:, :, 1] > 128).astype(float)
                    elif c == 2:  # 假设蓝色通道表示第三类
                        mask = (gt_image[:, :, 2] > 128).astype(float)
                    else:
                        mask = np.zeros(gt_image.shape[:2])
                    gt_masks.append(mask)
                return gt_masks
    
    # 如果找不到匹配的GT，寻找可能的同名不同扩展名文件
    all_gt_files = glob.glob(os.path.join(gt_dir, "*"))
    for gt_file in all_gt_files:
        gt_filename = os.path.splitext(os.path.basename(gt_file))[0]
        if filename.lower() in gt_filename.lower() or gt_filename.lower() in filename.lower():
            print(f"找到相似文件名的GT: {gt_file}")
            # 处理GT文件...与上面相同
            gt_image = io.imread(gt_file)
            
            # 处理GT格式
            if len(gt_image.shape) == 2:  # 单通道标签
                if num_classes > 1:
                    # 多类别分割任务，将单通道转为多通道
                    gt_masks = []
                    for c in range(num_classes):
                        mask = (gt_image == c+1).astype(float)
                        gt_masks.append(mask)
                    return gt_masks
                else:
                    # 二分类任务
                    gt_mask = (gt_image > 0).astype(float)
                    return [gt_mask]
            elif len(gt_image.shape) == 3 and gt_image.shape[2] == num_classes:
                # 已经是正确格式的多通道标签
                gt_masks = []
                for c in range(num_classes):
                    gt_masks.append(gt_image[:, :, c])
                return gt_masks
            elif len(gt_image.shape) == 3 and gt_image.shape[2] == 3:
                # RGB格式标签，需要转换
                print(f"警告: GT标签是RGB格式，尝试转换为多通道格式")
                gt_masks = []
                for c in range(num_classes):
                    # 简单的颜色区分方法，可能需要根据实际情况调整
                    if c == 0:  # 假设红色通道表示第一类
                        mask = (gt_image[:, :, 0] > 128).astype(float)
                    elif c == 1:  # 假设绿色通道表示第二类
                        mask = (gt_image[:, :, 1] > 128).astype(float)
                    elif c == 2:  # 假设蓝色通道表示第三类
                        mask = (gt_image[:, :, 2] > 128).astype(float)
                    else:
                        mask = np.zeros(gt_image.shape[:2])
                    gt_masks.append(mask)
                return gt_masks
    
    print(f"找不到对应的GT: {filename}")
    return None

# --------- 可视化结果 ---------
def visualize_results(original_image, prediction, output_path, num_classes, gt_masks=None):
    """可视化和保存预测结果，可选择性地与真实标签对比"""
    if gt_masks is not None:
        # 如果有GT，创建两行显示
        fig, axes = plt.subplots(2, num_classes + 1, figsize=(4 * (num_classes + 1), 8))
        
        # 第一行显示原始图像和预测结果
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        for c in range(num_classes):
            mask = prediction[c]
            axes[0, c+1].imshow(mask, cmap='jet')
            axes[0, c+1].set_title(f"Pred Class {c+1}")
            axes[0, c+1].axis('off')
            
        # 第二行显示真实标签
        axes[1, 0].imshow(original_image)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis('off')
        
        for c in range(num_classes):
            if c < len(gt_masks):
                gt_mask = gt_masks[c]
                axes[1, c+1].imshow(gt_mask, cmap='jet')
                axes[1, c+1].set_title(f"GT Class {c+1}")
                axes[1, c+1].axis('off')
            else:
                axes[1, c+1].axis('off')
    else:
        # 无GT，只显示一行
        fig, axes = plt.subplots(1, num_classes + 1, figsize=(4 * (num_classes + 1), 4))
        
        # 显示原始图像
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 显示每个通道的预测结果
        for c in range(num_classes):
            mask = prediction[c]
            axes[c+1].imshow(mask, cmap='jet')
            axes[c+1].set_title(f"Class {c+1}")
            axes[c+1].axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# --------- 保存预测掩码 ---------
def save_masks(prediction, output_prefix, num_classes, original_image=None, gt_masks=None):
    """
    将分割结果以彩色叠加在原图上进行可视化，可选择性地添加真实标签对比
    
    Args:
        prediction: 模型预测的多通道结果
        output_prefix: 输出文件路径前缀
        num_classes: 分割类别数量
        original_image: 原始图像
        gt_masks: 真实标签掩码列表
    """
    # 定义不同类别的颜色映射 (BGR格式) - 鲜艳且容易区分的颜色
    # 颜色示例: 红、绿、蓝、黄、洋红、青、橙、紫、粉、棕等
    color_map = [
        (255, 0, 0),     # 红色 - 类别1
        (0, 255, 0),     # 绿色 - 类别2
        (0, 0, 255),     # 蓝色 - 类别3
        (255, 255, 0),   # 黄色 - 类别4
        (255, 0, 255),   # 洋红 - 类别5
        (0, 255, 255),   # 青色 - 类别6
        (255, 165, 0),   # 橙色 - 类别7
        (128, 0, 128),   # 紫色 - 类别8
        (255, 192, 203), # 粉色 - 类别9
        (165, 42, 42)    # 棕色 - 类别10
    ]
    
    # 确保颜色足够
    if num_classes > len(color_map):
        # 如果类别超过预定义颜色，随机生成更多颜色
        for i in range(num_classes - len(color_map)):
            color_map.append((np.random.randint(0, 255), 
                             np.random.randint(0, 255), 
                             np.random.randint(0, 255)))
    
    # 创建一个彩色掩码图像
    if original_image is None:
        # 如果没有原图，创建一个空白画布
        h, w = prediction[0].shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # 使用原图作为背景
        colored_mask = original_image.copy()
        # 如果原图是灰度图，转换为RGB
        if len(colored_mask.shape) == 2:
            colored_mask = np.stack([colored_mask] * 3, axis=-1)
    
    # 创建一个透明度掩码
    alpha = 0.5  # 透明度参数
    
    # 对每个类别的预测结果进行处理
    for c in range(num_classes):
        # 获取类别预测掩码并二值化
        mask = prediction[c]
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 将二值掩码应用到原图上对应颜色
        for i in range(3):  # RGB三个通道
            channel = colored_mask[:, :, i]
            # 对掩码区域应用颜色混合: 原图*(1-alpha) + 掩码颜色*alpha
            channel[binary_mask == 1] = channel[binary_mask == 1] * (1 - alpha) + color_map[c][i] * alpha
    
    # 保存彩色掩码图像
    colored_mask_image = Image.fromarray(colored_mask.astype(np.uint8))
    colored_mask_image.save(f"{output_prefix}_segmentation.png")
    
    # 额外保存一个只有分割结果的彩色图像（不含原图）
    # 创建纯色彩分割图像
    pure_segmentation = np.zeros((prediction[0].shape[0], prediction[0].shape[1], 3), dtype=np.uint8)
    for c in range(num_classes):
        mask = prediction[c]
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        for i in range(3):
            channel = pure_segmentation[:, :, i]
            channel[binary_mask == 1] = color_map[c][i]
    
    # 保存纯分割结果图像
    pure_seg_image = Image.fromarray(pure_segmentation)
    pure_seg_image.save(f"{output_prefix}_pure_segmentation.png")
    
    # 如果有真实标签，同样进行可视化
    if gt_masks is not None:
        # 创建一个彩色真实标签图像
        if original_image is None:
            h, w = gt_masks[0].shape
            colored_gt_mask = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            colored_gt_mask = original_image.copy()
            if len(colored_gt_mask.shape) == 2:
                colored_gt_mask = np.stack([colored_gt_mask] * 3, axis=-1)
        
        # 对每个类别的真实标签进行处理
        for c in range(min(num_classes, len(gt_masks))):
            # 获取类别真实标签并二值化
            gt_mask = gt_masks[c]
            gt_binary_mask = (gt_mask > 0.5).astype(np.uint8)
            
            # 将二值掩码应用到原图上对应颜色
            for i in range(3):  # RGB三个通道
                channel = colored_gt_mask[:, :, i]
                channel[gt_binary_mask == 1] = channel[gt_binary_mask == 1] * (1 - alpha) + color_map[c][i] * alpha
        
        # 保存彩色真实标签图像
        colored_gt_image = Image.fromarray(colored_gt_mask.astype(np.uint8))
        colored_gt_image.save(f"{output_prefix}_gt_segmentation.png")
        
        # 创建纯色彩GT分割图像
        pure_gt_segmentation = np.zeros((gt_masks[0].shape[0], gt_masks[0].shape[1], 3), dtype=np.uint8)
        for c in range(min(num_classes, len(gt_masks))):
            gt_mask = gt_masks[c]
            gt_binary_mask = (gt_mask > 0.5).astype(np.uint8)
            
            for i in range(3):
                channel = pure_gt_segmentation[:, :, i]
                channel[gt_binary_mask == 1] = color_map[c][i]
        
        # 保存纯GT分割结果图像
        pure_gt_image = Image.fromarray(pure_gt_segmentation)
        pure_gt_image.save(f"{output_prefix}_pure_gt_segmentation.png")
        
        # 创建预测vs真实对比图
        comparison = np.zeros(((original_image.shape[0] if original_image is not None else prediction[0].shape[0]) * 2, 
                              (original_image.shape[1] if original_image is not None else prediction[0].shape[1]), 3), 
                              dtype=np.uint8)
        
        # 上半部分显示预测结果
        comparison[:comparison.shape[0]//2, :, :] = colored_mask_image
        # 下半部分显示真实标签
        comparison[comparison.shape[0]//2:, :, :] = colored_gt_image
        
        # 保存对比图
        comparison_image = Image.fromarray(comparison)
        comparison_image.save(f"{output_prefix}_pred_vs_gt.png")

# --------- 边界提取与可视化 ---------
def extract_boundaries(mask, thickness=3):
    """
    从分割掩码中提取边界
    
    Args:
        mask: 输入掩码 [H, W]
        thickness: 边界线条的粗细
    
    Returns:
        boundary: 边界图 [H, W]
    """
    # 二值化掩码
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # 使用形态学操作提取更粗的边界
    if thickness <= 1:
        # 如果需要细边界，使用Sobel算子
        grad_x = ndimage.sobel(binary_mask, axis=0)
        grad_y = ndimage.sobel(binary_mask, axis=1)
        boundary = np.sqrt(grad_x**2 + grad_y**2)
        boundary = (boundary > 0).astype(np.float32)
    else:
        # 使用膨胀和腐蚀提取更粗的边界
        # 先进行高斯平滑来减少噪点对边界的影响
        smoothed = ndimage.gaussian_filter(binary_mask.astype(float), sigma=1.0)
        smoothed_binary = (smoothed > 0.5).astype(np.uint8)
        
        # 使用形态学操作提取更粗的边界
        eroded = ndimage.binary_erosion(smoothed_binary, iterations=thickness//2)
        dilated = ndimage.binary_dilation(smoothed_binary, iterations=thickness//2)
        boundary = dilated.astype(np.float32) - eroded.astype(np.float32)
        
        # 确保边界是连续的
        if thickness > 3:
            # 对于较粗的边界，进行额外的膨胀操作
            boundary = ndimage.binary_dilation(boundary, iterations=1)
    
    # 确保边界的值在[0,1]范围内
    boundary = (boundary > 0).astype(np.float32)
    
    return boundary

def visualize_boundaries(prediction, output_prefix, num_classes, original_image=None, gt_masks=None, thickness=1, boundary_only=False):
    """
    可视化分割结果的边界，并与真实标签边界进行对比
    
    Args:
        prediction: 模型预测的多通道结果
        output_prefix: 输出文件路径前缀
        num_classes: 分割类别数量
        original_image: 原始图像
        gt_masks: 真实标签掩码列表
        thickness: 边界线条的粗细
        boundary_only: 是否只显示边界而不显示完整掩码
    """
    # 定义边界颜色映射 (RGB格式)
    boundary_colors = [
        [255, 0, 0],     # 红色 - 类别1
        [0, 255, 0],     # 绿色 - 类别2
        [0, 0, 255],     # 蓝色 - 类别3
        [255, 255, 0],   # 黄色 - 类别4
        [255, 0, 255],   # 洋红 - 类别5
    ]
    
    # 确保颜色足够
    if num_classes > len(boundary_colors):
        for i in range(num_classes - len(boundary_colors)):
            boundary_colors.append([np.random.randint(0, 255), 
                                  np.random.randint(0, 255), 
                                  np.random.randint(0, 255)])
    
    # 创建画布
    h, w = prediction[0].shape
    # 设置每个类别的单独显示
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    
    # 处理单类别的情况
    if num_classes == 1:
        axes = np.array([axes])
    
    # 为每个类别创建边界可视化
    for c in range(num_classes):
        # 提取当前类别的预测掩码和边界
        pred_mask = prediction[c]
        pred_boundary = extract_boundaries(pred_mask, thickness)
        
        # 提取当前类别的GT掩码和边界(如果有)
        if gt_masks is not None and c < len(gt_masks):
            gt_mask = gt_masks[c]
            gt_boundary = extract_boundaries(gt_mask, thickness)
        else:
            gt_mask = None
            gt_boundary = None
        
        # 创建边界可视化图像
        if original_image is not None:
            # 如果原图是灰度图，转换为RGB
            if len(original_image.shape) == 2:
                bg_image = np.stack([original_image] * 3, axis=-1)
            else:
                bg_image = original_image.copy()
        else:
            bg_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 创建预测边界可视化
        pred_vis = bg_image.copy()
        for i in range(3):
            pred_vis[:, :, i] = np.where(pred_boundary > 0, 
                                         boundary_colors[c][i], 
                                         pred_vis[:, :, i])
        
        # 创建GT边界可视化 (如果有GT)
        if gt_boundary is not None:
            gt_vis = bg_image.copy()
            for i in range(3):
                gt_vis[:, :, i] = np.where(gt_boundary > 0, 
                                          boundary_colors[c][i], 
                                          gt_vis[:, :, i])
            
            # 创建边界对比图 (预测边界为红色，GT边界为绿色)
            compare_vis = bg_image.copy()
            compare_vis[:, :, 0] = np.where(pred_boundary > 0, 255, compare_vis[:, :, 0])  # 红色
            compare_vis[:, :, 1] = np.where(gt_boundary > 0, 255, compare_vis[:, :, 1])    # 绿色
        else:
            gt_vis = np.zeros_like(pred_vis)
            compare_vis = pred_vis
        
        # 显示原始掩码、边界和对比图
        axes[c, 0].imshow(pred_vis)
        axes[c, 0].set_title(f'Class {c+1} Predicted Boundary')
        axes[c, 0].axis('off')
        
        if gt_boundary is not None:
            axes[c, 1].imshow(gt_vis)
            axes[c, 1].set_title(f'Class {c+1} GT Boundary')
        else:
            axes[c, 1].imshow(np.zeros_like(pred_vis))
            axes[c, 1].set_title(f'Class {c+1} No GT')
        axes[c, 1].axis('off')
        
        axes[c, 2].imshow(compare_vis)
        axes[c, 2].set_title(f'Class {c+1} Boundary Comparison (Red:Pred, Green:GT)')
        axes[c, 2].axis('off')
    
    # 保存边界可视化图像
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_boundaries.png", bbox_inches='tight')
    plt.close()
    
    # 额外创建所有类别边界的组合图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 组合所有预测边界
    all_pred_vis = original_image.copy() if original_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
    all_gt_vis = original_image.copy() if original_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
    all_compare_vis = original_image.copy() if original_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
    
    # 确保是RGB格式
    if len(all_pred_vis.shape) == 2:
        all_pred_vis = np.stack([all_pred_vis] * 3, axis=-1)
        all_gt_vis = np.stack([all_gt_vis] * 3, axis=-1)
        all_compare_vis = np.stack([all_compare_vis] * 3, axis=-1)
    
    # 为每个类别添加边界，使用不同颜色
    for c in range(num_classes):
        pred_mask = prediction[c]
        pred_boundary = extract_boundaries(pred_mask, thickness)
        
        # 在组合图中添加预测边界
        for i in range(3):
            all_pred_vis[:, :, i] = np.where(pred_boundary > 0, 
                                            boundary_colors[c][i], 
                                            all_pred_vis[:, :, i])
        
        # 在对比图中添加预测边界
        all_compare_vis[:, :, i] = np.where(pred_boundary > 0, 
                                           boundary_colors[c][i], 
                                           all_compare_vis[:, :, i])
        
        # 如果有GT，添加GT边界
        if gt_masks is not None and c < len(gt_masks):
            gt_mask = gt_masks[c]
            gt_boundary = extract_boundaries(gt_mask, thickness)
            
            # 在组合图中添加GT边界
            for i in range(3):
                all_gt_vis[:, :, i] = np.where(gt_boundary > 0, 
                                             boundary_colors[c][i], 
                                             all_gt_vis[:, :, i])
    
    # 显示所有边界的组合图
    axes[0].imshow(all_pred_vis)
    axes[0].set_title('All Predicted Boundaries')
    axes[0].axis('off')
    
    axes[1].imshow(all_gt_vis)
    axes[1].set_title('All GT Boundaries')
    axes[1].axis('off')
    
    axes[2].imshow(all_compare_vis)
    axes[2].set_title('Combined Boundaries')
    axes[2].axis('off')
    
    # 保存组合边界图
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_all_boundaries.png", bbox_inches='tight')
    plt.close()
    
    # 单独保存类别1和类别2的边界对比图(如果至少有两个类别)
    if num_classes >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 提取类别1的边界
        pred_mask1 = prediction[0]
        pred_boundary1 = extract_boundaries(pred_mask1, thickness)
        
        # 提取类别2的边界
        pred_mask2 = prediction[1]
        pred_boundary2 = extract_boundaries(pred_mask2, thickness)
        
        # 创建类别1和类别2的边界对比图
        class_compare_vis = original_image.copy() if original_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
        if len(class_compare_vis.shape) == 2:
            class_compare_vis = np.stack([class_compare_vis] * 3, axis=-1)
        
        # 类别1边界为红色
        class_compare_vis[:, :, 0] = np.where(pred_boundary1 > 0, 255, class_compare_vis[:, :, 0])
        
        # 类别2边界为绿色
        class_compare_vis[:, :, 1] = np.where(pred_boundary2 > 0, 255, class_compare_vis[:, :, 1])
        
        # 显示类别1和类别2的边界
        class1_vis = original_image.copy() if original_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
        if len(class1_vis.shape) == 2:
            class1_vis = np.stack([class1_vis] * 3, axis=-1)
        class1_vis[:, :, 0] = np.where(pred_boundary1 > 0, 255, class1_vis[:, :, 0])
        
        class2_vis = original_image.copy() if original_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
        if len(class2_vis.shape) == 2:
            class2_vis = np.stack([class2_vis] * 3, axis=-1)
        class2_vis[:, :, 1] = np.where(pred_boundary2 > 0, 255, class2_vis[:, :, 1])
        
        # 显示类别1、类别2和对比图
        axes[0].imshow(class1_vis)
        axes[0].set_title('Class 1 Boundary (Red)')
        axes[0].axis('off')
        
        axes[1].imshow(class2_vis)
        axes[1].set_title('Class 2 Boundary (Green)')
        axes[1].axis('off')
        
        axes[2].imshow(class_compare_vis)
        axes[2].set_title('Class 1 and Class 2 Boundary Comparison')
        axes[2].axis('off')
        
        # 保存类别对比图
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_class1_class2_boundaries.png", bbox_inches='tight')
        plt.close()
        
        # 如果有GT，创建预测与GT的细粒度对比图
        if gt_masks is not None and len(gt_masks) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # 提取GT边界
            gt_mask1 = gt_masks[0]
            gt_boundary1 = extract_boundaries(gt_mask1, thickness)
            
            gt_mask2 = gt_masks[1]
            gt_boundary2 = extract_boundaries(gt_mask2, thickness)
            
            # 创建预测vs GT的对比图
            class1_compare = original_image.copy() if original_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
            if len(class1_compare.shape) == 2:
                class1_compare = np.stack([class1_compare] * 3, axis=-1)
            
            class2_compare = class1_compare.copy()
            
            # 类别1: 预测边界为红色，GT边界为绿色
            class1_compare[:, :, 0] = np.where(pred_boundary1 > 0, 255, class1_compare[:, :, 0])
            class1_compare[:, :, 1] = np.where(gt_boundary1 > 0, 255, class1_compare[:, :, 1])
            
            # 类别2: 预测边界为红色，GT边界为绿色
            class2_compare[:, :, 0] = np.where(pred_boundary2 > 0, 255, class2_compare[:, :, 0])
            class2_compare[:, :, 1] = np.where(gt_boundary2 > 0, 255, class2_compare[:, :, 1])
            
            # 显示类别1和类别2的预测vs GT对比
            axes[0, 0].imshow(pred_boundary1, cmap='gray')
            axes[0, 0].set_title('Class 1 Predicted Boundary')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gt_boundary1, cmap='gray')
            axes[0, 1].set_title('Class 1 GT Boundary')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(pred_boundary2, cmap='gray')
            axes[1, 0].set_title('Class 2 Predicted Boundary')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(gt_boundary2, cmap='gray')
            axes[1, 1].set_title('Class 2 GT Boundary')
            axes[1, 1].axis('off')
            
            # 保存细粒度对比图
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_detailed_boundaries.png", bbox_inches='tight')
            plt.close()
            
            # 创建类别1和类别2的预测与GT直接对比图
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(class1_compare)
            axes[0].set_title('Class 1 Boundary Comparison (Red:Pred, Green:GT)')
            axes[0].axis('off')
            
            axes[1].imshow(class2_compare)
            axes[1].set_title('Class 2 Boundary Comparison (Red:Pred, Green:GT)')
            axes[1].axis('off')
            
            # 保存直接对比图
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_direct_boundary_comparison.png", bbox_inches='tight')
            plt.close()

# 添加新的函数: 可视化前景标签
def visualize_foreground_labels(image, predictions, output_path, num_classes, background_channel=0):
    """将非背景标签(除通道0外)在原图上以不同颜色可视化"""
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    # 创建标签掩码和纯标签图像
    h, w = predictions[0].shape
    pure_labels = np.zeros((h, w, 3), dtype=np.uint8)
    overlay = image_rgb.copy()
    
    # 颜色列表
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 青色
        (255, 165, 0),  # 橙色
        (128, 0, 128),  # 紫色
        (255, 192, 203),# 粉色
        (165, 42, 42)   # 棕色
    ]
    
    # 为每个前景类别应用颜色
    foreground_mask = np.zeros((h, w), dtype=bool)
    
    for c in range(num_classes):
        if c == background_channel:
            continue  # 跳过背景通道
            
        mask = predictions[c] > 0.5
        if not np.any(mask):
            continue  # 跳过没有像素的类别
            
        foreground_mask |= mask  # 更新前景掩码
        color = colors[min(c, len(colors)-1)]  # 获取颜色
        
        # 应用颜色到覆盖图
        for i in range(3):
            overlay[:, :, i][mask] = overlay[:, :, i][mask] * 0.3 + color[i] * 0.7
            
        # 应用颜色到纯标签图
        for i in range(3):
            pure_labels[:, :, i][mask] = color[i]
    
    # 保存纯标签图像
    Image.fromarray(pure_labels).save(os.path.splitext(output_path)[0] + "_pure_labels.png")
    
    # 保存覆盖图
    Image.fromarray(overlay).save(output_path)

# --------- 主函数 ---------
def main():
    # 解析命令行参数
    args = get_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    net = load_model(
        args.model_path, 
        args.model_name, 
        args.num_classes, 
        device
    )
    print(f"Model loaded from {args.model_path}")
    
    # 根据模式选择处理方式
    if args.mode == 1:
        # 模式1: 完整处理所有图像
        # 获取输入图像列表
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            found_paths = glob.glob(os.path.join(args.input_dir, f'*{ext}'))
            # 确保找到的路径是文件而不是目录
            for path in found_paths:
                if os.path.isfile(path):
                    image_paths.append(path)
        
        print(f"Found {len(image_paths)} images")
        image_paths.sort()
        
        if len(image_paths) == 0:
            raise ValueError(f"No valid image files found in {args.input_dir}")
            
        # 处理每个图像
        for i, image_path in enumerate(image_paths):
            try:
                # 获取文件名
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # 读取图像
                original_image = io.imread(image_path)
                
                # 预处理图像 - 使用与训练阶段相同的MultiChannelToTensorLab
                input_image = transform_image(
                    original_image, 
                    args.input_size, 
                    args.flag, 
                    args.num_classes
                )
                
                # 推理
                prediction = predict(net, input_image, device)
                
                # 将预测结果调整为原始图像大小
                resized_pred = []
                for c in range(args.num_classes):
                    channel_pred = prediction[c]
                    resized_channel = transform.resize(
                        channel_pred, 
                        (original_image.shape[0], original_image.shape[1]), 
                        mode='constant'
                    )
                    resized_pred.append(resized_channel)
                
                # 加载真实标签（如果存在）
                gt_masks = find_gt_mask(name_without_ext, args.gt_dir, args.num_classes)
                
                # 保存标准可视化结果
                if not args.boundary_only:
                    output_path = os.path.join(args.output_dir, f"{name_without_ext}_visual.png")
                    visualize_results(original_image, resized_pred, output_path, args.num_classes, gt_masks)
                    
                    # 保存彩色分割结果
                    output_prefix = os.path.join(args.output_dir, name_without_ext)
                    save_masks(resized_pred, output_prefix, args.num_classes, original_image, gt_masks)
                
                # 保存边界可视化结果
                output_prefix = os.path.join(args.output_dir, name_without_ext)
                visualize_boundaries(
                    resized_pred, 
                    output_prefix, 
                    args.num_classes, 
                    original_image, 
                    gt_masks, 
                    args.boundary_thickness
                )
                
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    elif args.mode == 2:
        # 模式2: 处理前景标签可视化
        # 获取图像路径
        image_paths = []
        if args.single_image is not None:
            if not os.path.exists(args.single_image):
                raise ValueError(f"Input image not found: {args.single_image}")
            image_paths = [args.single_image]
        else:
            # 从输入目录获取所有图像
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                found_paths = glob.glob(os.path.join(args.input_dir, f'*{ext}'))
                for path in found_paths:
                    if os.path.isfile(path):
                        image_paths.append(path)
            
            if len(image_paths) == 0:
                raise ValueError(f"No valid image files found in {args.input_dir}")
        
        print(f"Found {len(image_paths)} images to process")
        
        # 处理每个图像
        for image_path in image_paths:
            try:
                # 获取文件名
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # 读取并处理图像
                print(f"Processing: {filename}")
                original_image = io.imread(image_path)
                input_tensor = transform_image(original_image, args.input_size, args.flag, args.num_classes)
                prediction = predict(net, input_tensor, device)
                
                # 调整预测结果为原始图像大小
                resized_pred = []
                for c in range(args.num_classes):
                    resized_channel = transform.resize(
                        prediction[c], 
                        (original_image.shape[0], original_image.shape[1]), 
                        mode='constant'
                    )
                    resized_pred.append(resized_channel)
                
                # 可视化前景标签
                output_path = os.path.join(args.output_dir, f"{name_without_ext}_foreground_labels.png")
                visualize_foreground_labels(original_image, resized_pred, output_path, args.num_classes)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        print(f"前景标签可视化完成! 共处理 {len(image_paths)} 张图像。")
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print("推理完成! 结果保存在:", args.output_dir)

if __name__ == "__main__":
    main() 