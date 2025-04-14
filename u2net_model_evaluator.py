import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io, transform
import argparse
from PIL import Image
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from pathlib import Path

from model import U2NET
from model import U2NETP
from data_loader import MultiChannelToTensorLab, RescaleT

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='U2Net多模型评估工具')
    parser.add_argument('--input_image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--models_dir', type=str, required=True, help='模型所在目录')
    parser.add_argument('--gt_mask', type=str, required=True, help='真实标签图像路径')
    parser.add_argument('--output_dir', type=str, default='model_evaluation_results', help='评估结果输出目录')
    parser.add_argument('--num_classes', type=int, default=8, help='分割类别数量')
    parser.add_argument('--input_size', type=int, default=512, help='输入图像大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='分割二值化阈值')
    parser.add_argument('--flag', type=int, default=0, choices=[0, 1, 2], help='颜色空间标志,0:RGB, 1:Lab, 2:RGB+Lab')
    parser.add_argument('--class_names', type=str, nargs='+', help='类别名称列表，顺序与模型输出通道对应')
    parser.add_argument('--device', type=str, default='auto', help='设备选择 (auto, cpu, cuda)')
    return parser.parse_args()

def get_device(device_arg):
    """获取设备"""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg

def transform_image(image, target_size=512, flag=0, num_classes=8):
    """对输入图像进行预处理，用于模型推理"""
    # 确保值在0-1之间
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建一个假标签（推理时不需要真实标签）
    dummy_label = np.zeros((image.shape[0], image.shape[1], num_classes))
    
    # 准备样本字典
    sample = {
        'imidx': np.array([0]),
        'image': image.copy(),
        'label': dummy_label
    }
    
    # 首先使用RescaleT调整图像大小
    if image.shape[0] != target_size or image.shape[1] != target_size:
        rescale = RescaleT(target_size)
        sample = rescale(sample)
    
    # 然后使用MultiChannelToTensorLab处理图像
    to_tensor = MultiChannelToTensorLab(flag=flag, num_channels=num_classes)
    tensor_sample = to_tensor(sample)
    
    # 提取处理后的图像张量并添加批次维度
    input_tensor = tensor_sample['image'].unsqueeze(0)
    
    return input_tensor

def load_model(model_path, model_name='u2net', num_classes=8, device='cpu'):
    """加载多通道分割模型"""
    # 创建模型
    print(f"Loading model from {model_path}")
    if model_name.lower() == 'u2net':
        net = U2NET(3, num_classes)
    elif model_name.lower() == 'u2netp':
        net = U2NETP(3, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # 加载权重
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        return net
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict(net, image_tensor, device='cpu'):
    """使用模型进行推理"""
    # 将图像移动到指定设备
    inputs = image_tensor.to(device)
    
    # 进行推理
    with torch.no_grad():
        try:
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    # 取d0作为最终预测结果 (已经经过sigmoid)
    pred = d0.squeeze(0)
    pred = pred.cpu().numpy()
    
    return pred

def find_models(models_dir, model_name=None):
    """查找指定目录下的所有模型权重文件"""
    model_files = []
    
    # 设置常见的模型文件扩展名
    model_exts = ['.pth', '.pt']
    
    # 搜索目录和子目录中的所有模型文件
    for ext in model_exts:
        if model_name:
            # 如果指定了模型名称，只查找相关模型
            pattern = os.path.join(models_dir, f"**/*{model_name}*{ext}")
            model_files.extend(glob.glob(pattern, recursive=True))
        else:
            # 否则查找所有模型文件
            pattern = os.path.join(models_dir, f"**/*{ext}")
            model_files.extend(glob.glob(pattern, recursive=True))
    
    return model_files

def load_image(image_path, target_size=None):
    """加载并预处理图像"""
    try:
        # 读取图像
        image = io.imread(image_path)
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:  # 灰度图
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA图
            image = image[:, :, :3]
        
        # 如果指定了目标尺寸，调整图像大小
        if target_size is not None:
            image = transform.resize(image, (target_size, target_size), mode='constant')
            # 确保值范围在0-1之间
            if image.max() > 1.0:
                image = image / 255.0
        
        return image
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None

def load_mask(mask_path, num_classes=8, target_size=None):
    """加载并预处理掩码"""
    try:
        # 读取掩码
        mask = io.imread(mask_path)
        
        # 调整掩码尺寸(如果需要)
        if target_size is not None:
            mask = transform.resize(mask, (target_size, target_size), mode='constant', order=0, preserve_range=True)
        
        # 掩码格式处理
        if len(mask.shape) == 2:
            # 单通道掩码，需要转换为多通道
            masks = []
            for c in range(num_classes):
                # 创建当前类别的掩码
                class_mask = np.zeros_like(mask)
                class_mask[mask == c] = 1
                masks.append(class_mask)
            return masks
        elif len(mask.shape) == 3 and mask.shape[2] == num_classes:
            # 已经是多通道格式，直接分离
            return [mask[:, :, c] for c in range(num_classes)]
        elif len(mask.shape) == 3 and mask.shape[2] == 3:
            # RGB掩码，需要转换
            print(f"GT mask is RGB format, trying to convert to multi-channel.")
            # 假设是按类别索引编码的
            multi_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes))
            # 尝试从RGB值推断类别
            for c in range(num_classes):
                if c == 0:  # 背景通常为黑色
                    class_mask = (mask.sum(axis=2) == 0).astype(float)
                else:
                    # 根据颜色通道区分类别
                    # 这里是简化处理，实际应用可能需要更复杂的颜色映射
                    dominant_channel = (c - 1) % 3
                    threshold = 128
                    class_mask = (mask[:, :, dominant_channel] > threshold).astype(float)
                    # 确保与其他通道不冲突
                    for other_ch in range(3):
                        if other_ch != dominant_channel:
                            class_mask = class_mask & (mask[:, :, other_ch] <= threshold)
                multi_mask[:, :, c] = class_mask
            
            return [multi_mask[:, :, c] for c in range(num_classes)]
        
        print(f"Warning: Unsupported mask format: {mask.shape}")
        return None
    except Exception as e:
        print(f"Failed to load mask {mask_path}: {e}")
        return None

def calculate_metrics(pred, gt, threshold=0.5):
    """
    计算分割评估指标：每个类别的像素点准确率(TP/GT像素数)
    
    Args:
        pred: 模型预测的分割掩码 [C, H, W]
        gt: 真实标签 [C, H, W] 或 list of [H, W]
        threshold: 二值化阈值
    
    Returns:
        metrics: 包含各指标的字典
    """
    num_classes = pred.shape[0]
    
    # 确保gt是列表格式
    if not isinstance(gt, list):
        gt_list = [gt[c] for c in range(num_classes)]
    else:
        gt_list = gt
    
    # 准备结果字典
    metrics = {
        'accuracy_per_class': [],
        'class_exists': [],  # 标记每个类别是否在GT中存在
    }
    
    # 对每个类别计算指标
    for c in range(num_classes):
        if c < len(gt_list):
            # 二值化预测和真实标签
            pred_binary = (pred[c] > threshold).astype(np.float32)
            gt_binary = (gt_list[c] > threshold).astype(np.float32)
            
            # 检查该类别是否在GT中存在（有非零像素）
            gt_area = gt_binary.sum()
            
            if gt_area > 0:
                # 该类别存在，计算准确率 = TP / (GT像素数)
                # TP是预测为1且真实也为1的像素数
                tp = np.logical_and(pred_binary > 0, gt_binary > 0).sum()
                
                # 准确率 = TP / GT像素总数
                accuracy = tp / gt_area
                
                metrics['accuracy_per_class'].append(accuracy)
                metrics['class_exists'].append(True)
            else:
                # 该类别在GT中不存在，标记为null
                metrics['accuracy_per_class'].append(None)
                metrics['class_exists'].append(False)
        else:
            # 如果没有对应的GT类别，设为null
            metrics['accuracy_per_class'].append(None)
            metrics['class_exists'].append(False)
    
    # 计算存在的类别的平均准确率
    valid_accuracies = [acc for acc, exists in zip(metrics['accuracy_per_class'], metrics['class_exists']) 
                        if exists and acc is not None]
    
    if valid_accuracies:
        metrics['mean_accuracy'] = np.mean(valid_accuracies)
    else:
        metrics['mean_accuracy'] = None
    
    return metrics

def save_model_predictions(prediction, output_path, gt_masks=None, class_names=None, threshold=0.5, original_image=None):
    """
    保存模型预测的分割结果图片和GT的比较
    
    Args:
        prediction: 模型预测结果 [C, H, W]
        output_path: 输出目录
        gt_masks: 真实标签掩码列表
        class_names: 类别名称列表
        threshold: 二值化阈值
        original_image: 原始图像，用于参考
    """
    num_classes = prediction.shape[0]
    
    # 获取图像尺寸
    h, w = prediction[0].shape
    
    # 设置类别名称
    if class_names is None or len(class_names) < num_classes:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # 处理每个类别
    for c in range(num_classes):
        # 创建二值掩码
        pred_binary = (prediction[c] > threshold).astype(np.uint8) * 255
        pred_pixels = (pred_binary > 0).sum()
        
        # 获取GT掩码（如果有）
        gt_binary = None
        gt_pixels = 0
        tp = 0
        
        # 只处理GT中存在的类别
        if gt_masks is not None and c < len(gt_masks):
            gt_binary = (gt_masks[c] > threshold).astype(np.uint8) * 255
            gt_pixels = (gt_binary > 0).sum()
            
            # 如果GT中没有该类别的像素，跳过
            if gt_pixels == 0:
                print(f"Skipping class {class_names[c]} as it has no GT pixels")
                continue
                
            # 计算TP (真阳性)
            tp = np.logical_and(pred_binary > 0, gt_binary > 0).sum()
            
            # 计算差异区域
            diff_mask = np.zeros_like(gt_binary)
            # 标记假阳性 (FP) - 预测有但实际没有
            diff_mask[np.logical_and(pred_binary > 0, gt_binary == 0)] = 127
            # 标记假阴性 (FN) - 预测没有但实际有
            diff_mask[np.logical_and(pred_binary == 0, gt_binary > 0)] = 255
            
            # 直接创建三通道RGB图像 (灰度值转RGB)
            comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
            
            # 将黑白掩码复制到RGB的三个通道
            for i in range(3):
                comparison[:, :w, i] = pred_binary
                comparison[:, w:2*w, i] = gt_binary
                comparison[:, 2*w:, i] = diff_mask
            
            # 添加文本信息
            # 创建PIL图像用于添加文本
            from PIL import ImageDraw, ImageFont, Image
            
            # 直接使用RGB图像
            comparison_pil = Image.fromarray(comparison)
            draw = ImageDraw.Draw(comparison_pil)
            
            # 尝试加载一个常见字体，并设置较大的字号
            font_size = 36  # 增大字体大小
            try:
                font = ImageFont.truetype("Arial", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    # 如果无法加载指定字体，使用默认字体
                    font = ImageFont.load_default()
            
            # 设置蓝色文字
            text_color = (0, 0, 255)  # RGB蓝色
            stroke_color = (255, 255, 255)  # 白色描边
            
            # 添加预测部分文本信息
            draw.text((10, 10), f"PREDICTION: {class_names[c]}", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            draw.text((10, 10 + font_size + 10), f"Pred pixels: {pred_pixels}", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            
            # 添加GT部分文本信息
            draw.text((w + 10, 10), f"GROUND TRUTH: {class_names[c]}", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            draw.text((w + 10, 10 + font_size + 10), f"GT pixels: {gt_pixels}", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            
            # 添加差异部分文本信息
            draw.text((2*w + 10, 10), f"DIFFERENCE", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            draw.text((2*w + 10, 10 + font_size + 10), f"TP: {tp} px", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            
            # 计算FP和FN
            fp = pred_pixels - tp
            fn = gt_pixels - tp
            draw.text((2*w + 10, 10 + (font_size + 10) * 2), f"FP: {fp} px", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            draw.text((2*w + 10, 10 + (font_size + 10) * 3), f"FN: {fn} px", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            
            # 如果GT有像素，则计算并显示准确率
            if gt_pixels > 0:
                accuracy = tp / gt_pixels
                draw.text((2*w + 10, 10 + (font_size + 10) * 4), f"Accuracy: {accuracy:.4f}", fill=text_color, font=font, stroke_width=2, stroke_fill=stroke_color)
            
            # 保存比较图像
            comparison_pil.save(os.path.join(output_path, f"compare_{class_names[c]}.png"))

def get_class_names(args_class_names, num_classes):
    """
    获取类别名称，如果命令行未提供则使用默认名称
    """
    if args_class_names and len(args_class_names) >= num_classes:
        return args_class_names[:num_classes]
    
    # 默认类别名称
    default_names = ['BG', 'A', 'B', 'C', 'D', 'DS', 'TIB', 'TID', 'Class9', 'Class10']
    return default_names[:num_classes]

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载输入图像
    original_image = load_image(args.input_image, target_size=args.input_size)
    if original_image is None:
        print(f"Failed to load input image: {args.input_image}")
        return
    
    # 加载真实标签掩码
    gt_masks = load_mask(args.gt_mask, args.num_classes, target_size=args.input_size)
    if gt_masks is None:
        print(f"Failed to load ground truth mask: {args.gt_mask}")
        return
    
    # 准备输入张量
    input_tensor = transform_image(original_image, args.input_size, args.flag, args.num_classes)
    
    # 查找模型
    model_files = find_models(args.models_dir)
    if not model_files:
        print(f"No model files found in directory: {args.models_dir}")
        return
    
    print(f"Found {len(model_files)} model files")
    
    # 获取类别名称
    class_names = get_class_names(args.class_names, args.num_classes)
    
    # 对每个模型进行评估
    for model_path in tqdm(model_files, desc="Evaluating models"):
        # 提取模型名称和类型
        model_filename = os.path.basename(model_path)
        model_name_without_ext = os.path.splitext(model_filename)[0]
        
        if 'u2netp' in model_filename.lower():
            model_name = 'u2netp'
        else:
            model_name = 'u2net'
        
        # 加载模型
        net = load_model(model_path, model_name, args.num_classes, device)
        if net is None:
            continue
        
        # 进行推理
        prediction = predict(net, input_tensor, device)
        if prediction is None:
            continue
        
        # 计算评估指标
        metrics = calculate_metrics(prediction, gt_masks, args.threshold)
        
        # 创建带有准确率的文件夹名
        folder_name = model_name_without_ext
        
        # 添加除BG外所有类别的准确率到文件夹名
        for c in range(1, args.num_classes):  # 从1开始，跳过BG类别
            if c < len(class_names) and metrics['class_exists'][c] and metrics['accuracy_per_class'][c] is not None:
                acc = metrics['accuracy_per_class'][c]
                folder_name += f"_{class_names[c]}_{acc:.4f}"
        
        # 为每个模型创建单独的输出目录并保存预测结果
        model_result_dir = os.path.join(args.output_dir, folder_name)
        os.makedirs(model_result_dir, exist_ok=True)
        
        # 保存该模型的预测结果图片，包含GT对比和像素统计
        save_model_predictions(
            prediction=prediction, 
            output_path=model_result_dir, 
            gt_masks=gt_masks, 
            class_names=class_names, 
            threshold=args.threshold,
            original_image=original_image
        )
        
        # 释放模型内存
        del net
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"Evaluation completed. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 