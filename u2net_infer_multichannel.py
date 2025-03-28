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
from skimage import io, transform, color
from PIL import Image
import matplotlib.pyplot as plt
import argparse

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
    
    # 获取输入图像列表
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f'*{ext}')))
    
    print(f"Found {len(image_paths)} images")
    image_paths.sort()
    # 处理每个图像
    for i, image_path in enumerate(image_paths):
        if i < 90:
            continue
            
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
        
        # 保存可视化结果
        output_path = os.path.join(args.output_dir, f"{name_without_ext}_visual.png")
        visualize_results(original_image, resized_pred, output_path, args.num_classes, gt_masks)
        
        # 保存彩色分割结果
        output_prefix = os.path.join(args.output_dir, name_without_ext)
        save_masks(resized_pred, output_prefix, args.num_classes, original_image, gt_masks)
        
        print(f"Processed {filename}")
    
    print("推理完成! 结果保存在:", args.output_dir)

if __name__ == "__main__":
    main() 