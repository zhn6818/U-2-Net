import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import argparse
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from model import U2NET
from model import U2NETP
from data_loader import MultiChannelToTensorLab, RescaleT

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化原图和标注图')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='标注掩码目录')
    parser.add_argument('--num_classes', type=int, default=8, help='分割类别数量(包括背景)')
    parser.add_argument('--input_size', type=int, default=512, help='输入图像大小')
    parser.add_argument('--alpha', type=float, default=0.7, help='标注覆盖透明度')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径,如果提供则进行推理')
    parser.add_argument('--model_name', type=str, default='u2net', choices=['u2net', 'u2netp'], help='模型名称')
    parser.add_argument('--flag', type=int, default=0, choices=[0, 1, 2], help='颜色空间标志,0:RGB, 1:Lab, 2:RGB+Lab')
    parser.add_argument('--device', type=str, default='auto', help='设备选择 (auto, cpu, cuda)')
    return parser.parse_args()

def get_device(device_arg):
    """获取设备"""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg

def get_color_map():
    """获取类别对应的颜色映射"""
    # 设置类别对应颜色: A类红色，B类黄色，C类蓝色，D类绿色，其他类随机颜色
    color_map = {
        "BG": [0, 0, 0],       # 背景为黑色
        "A": [255, 0, 0],      # 红色
        "B": [255, 255, 0],    # 黄色
        "C": [0, 0, 255],      # 蓝色
        "D": [0, 255, 0],      # 绿色
        "DS": [255, 0, 255],   # 紫色
        "TIB": [0, 255, 255],  # 青色
        "TID": [255, 165, 0]   # 橙色
    }
    return color_map

def get_class_name_by_index(idx):
    """根据索引获取类别名称"""
    class_names = ['BG', 'A', 'B', 'C', 'D', 'DS', 'TIB', 'TID']
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return f"Unknown Class {idx}"

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
        
        # 掩码格式处理: 保证格式为[height, width], 值为类别索引
        if len(mask.shape) == 3:
            # 如果掩码是RGB格式，转换为单通道(简单取第一个通道，实际应根据数据格式调整)
            if mask.shape[2] == 3:
                # 假设RGB格式存储了类别索引
                mask = mask[:,:,0]
        
        # 确保掩码是整数类型
        mask = mask.astype(np.int32)
        
        return mask
    except Exception as e:
        print(f"Failed to load mask {mask_path}: {e}")
        return None

def transform_image_for_model(image, target_size=512, flag=0, num_classes=8):
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
    
    # 首先使用RescaleT调整图像大小（如果尺寸已经对应，这一步可以跳过）
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
    print(f"Loading model {model_name} from {model_path} with {num_classes} classes")
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

def predict(net, image_tensor, device='cpu'):
    """使用模型进行推理"""
    # 将图像移动到指定设备
    inputs = image_tensor.to(device)
    
    # 进行推理
    with torch.no_grad():
        d0, d1, d2, d3, d4, d5, d6 = net(inputs)
    
    # 取d0作为最终预测结果 (已经经过sigmoid)
    pred = d0.squeeze(0)
    pred = pred.cpu().numpy()
    
    return pred

def create_colored_mask(mask, num_classes=8):
    """创建彩色掩码图像"""
    # 获取颜色映射
    color_map = get_color_map()
    class_names = list(color_map.keys())
    
    # 创建空白彩色掩码
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个类别赋予不同颜色
    for i in range(num_classes):
        if i < len(class_names):
            # 获取类别名称
            class_name = class_names[i]
            # 获取该类别对应的颜色
            color = color_map[class_name]
            # 为当前类别的像素赋值颜色
            colored_mask[mask == i] = color
    
    return colored_mask

def create_colored_pred_mask(pred, num_classes=8, threshold=0.5):
    """从预测概率创建彩色掩码图像"""
    # 获取颜色映射
    color_map = get_color_map()
    class_names = list(color_map.keys())
    
    # 预测结果形状应该是 [num_classes, height, width]
    h, w = pred.shape[1], pred.shape[2]
    
    # 创建空白彩色掩码
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 创建分类结果，取每个位置概率最大的类别
    segmentation = np.argmax(pred, axis=0)
    
    # 为每个类别赋予不同颜色
    for i in range(num_classes):
        if i < len(class_names):
            # 获取类别名称
            class_name = class_names[i]
            # 获取该类别对应的颜色
            color = color_map[class_name]
            # 为当前类别的像素赋值颜色
            colored_mask[segmentation == i] = color
    
    return colored_mask

def visualize_and_save(image, mask, output_path, num_classes=8, alpha=0.7, pred=None):
    """可视化图像和掩码，并保存结果"""
    # 创建彩色掩码
    colored_mask = create_colored_mask(mask, num_classes)
    
    # 确保图像是0-255范围的uint8类型
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # 创建图形
    if pred is not None:
        # 包含预测结果，显示三列
        plt.figure(figsize=(18, 6))
        
        # 创建彩色预测掩码
        colored_pred = create_colored_pred_mask(pred, num_classes)
        
        # 显示原图
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示真实标注图
        plt.subplot(1, 3, 2)
        plt.imshow(colored_mask)
        plt.title('Ground Truth')
        plt.axis('off')
        
        # 显示预测标注图
        plt.subplot(1, 3, 3)
        plt.imshow(colored_pred)
        plt.title('Model Prediction')
        plt.axis('off')
    else:
        # 不包含预测结果，显示两列
        plt.figure(figsize=(12, 6))
        
        # 显示原图
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示标注图
        plt.subplot(1, 2, 2)
        plt.imshow(colored_mask)
        plt.title('Ground Truth')
        plt.axis('off')
    
    # 添加图例
    color_map = get_color_map()
    patches = []
    for i, (class_name, color) in enumerate(color_map.items()):
        if class_name == 'BG':  # 背景可以选择不在图例中显示
            continue
        # 转换颜色为0-1范围
        normalized_color = [c/255 for c in color]
        patch = plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=normalized_color, markersize=10,
                          label=class_name)
        patches.append(patch)
    
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def process_images(args):
    """处理所有图像"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有输入图像
    image_files = glob.glob(os.path.join(args.input_dir, '*.*'))
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # 如果提供了模型路径，加载模型
    model = None
    device = 'cpu'
    if args.model_path:
        device = get_device(args.device)
        print(f"Using device: {device}")
        model = load_model(args.model_path, args.model_name, args.num_classes, device)
        print(f"Model loaded successfully")
    
    # 处理每张图像
    for image_path in image_files:
        # 获取文件名(不含扩展名)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 查找对应的掩码文件
        mask_files = glob.glob(os.path.join(args.gt_dir, f"{filename}.*"))
        if not mask_files:
            print(f"Warning: No matching mask found for {filename}")
            continue
        
        mask_path = mask_files[0]
        
        # 加载图像和掩码
        image = load_image(image_path, args.input_size)
        mask = load_mask(mask_path, args.num_classes, args.input_size)
        
        if image is None or mask is None:
            print(f"Skipping {filename}: Failed to load image or mask")
            continue
        
        # 输出路径
        output_path = os.path.join(args.output_dir, f"{filename}_visualization.png")
        
        # 如果有模型，进行预测
        pred = None
        if model:
            # 预处理图像
            image_tensor = transform_image_for_model(image, args.input_size, args.flag, args.num_classes)
            # 进行推理
            pred = predict(model, image_tensor, device)
        
        # 可视化并保存
        visualize_and_save(image, mask, output_path, args.num_classes, args.alpha, pred)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 处理所有图像
    process_images(args)
    
    print("All images processed successfully!")

if __name__ == "__main__":
    main()
