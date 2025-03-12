import os
import shutil
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import cv2

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map

pred_size = 512


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

transform = transforms.Compose([
        RescaleT(pred_size),
        ToTensorLab(flag=0)
    ])

def get_device():
    """
    获取可用的设备类型：CUDA、MPS 或 CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def read_image(image_path):
    """
    读取图片并进行基础处理
    Args:
        image_path: 图片路径
    Returns:
        image: 处理后的图片数组
    """
    # Read image using skimage
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    return image

def inference_single_image(image, net, device, transform):
    """
    对单张图片进行推理
    Args:
        image: 图片数组
        net: 模型
        device: 设备(cuda/cpu/mps)
        transform: 图像预处理转换
    Returns:
        pred: 模型预测结果
    """
    # Create empty label with correct dimensions
    label = np.zeros(image.shape[0:2])
    label = label[:,:,np.newaxis]  # Add channel dimension to match image
    
    # Apply same transforms as in original dataloader
    sample = {'image': image, 'label': label, 'imidx': np.array([0])}
    sample = transform(sample)
    
    # Prepare input
    inputs_test = sample['image']
    inputs_test = inputs_test.unsqueeze(0)  # Add batch dimension
    inputs_test = inputs_test.type(torch.FloatTensor)
    inputs_test = Variable(inputs_test).to(device)
    
    # Forward pass
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7 = net(inputs_test)
    
    # Normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    
    # Clean up
    del d1,d2,d3,d4,d5,d6,d7
    
    return pred

def process_prediction(pred, original_image):
    """
    处理模型预测结果，转换为可保存的图像格式
    Args:
        pred: 模型预测的张量结果
        original_image: 原始图像数组，用于获取尺寸
    Returns:
        pred_image: 处理后的PIL图像对象
    """
    # Convert prediction to numpy array
    pred_np = pred.squeeze().cpu().data.numpy()
    
    # Convert to PIL image
    pred_image = Image.fromarray((pred_np * 255).astype(np.uint8)).convert('RGB')
    
    # Resize to original image size
    pred_image = pred_image.resize(
        (original_image.shape[1], original_image.shape[0]), 
        resample=Image.BILINEAR
    )
    
    return pred_image

def Get_net():
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location=device))
    net.to(device)
    net.eval()
    return device, net

def sliding_window_inference(image, net, device, transform):
    """
    使用滑动窗口进行推理
    Args:
        image: 原始图像数组
        net: 模型
        device: 设备
        transform: 图像预处理转换
    Returns:
        predictions: 包含预测结果的列表
    """
    height, width = image.shape[:2]
    window_height = height // 2
    window_width = width // 2
    step_height = height // 8
    step_width = width // 8
    
    predictions = []
    count = 0
    
    for y in range(0, height - window_height + 1, step_height):
        if y + window_height > height:
            y = height - window_height
        for x in range(0, width - window_width + 1, step_width):
            if x + window_width > width:
                x = width - window_width
                
            # 提取窗口图像
            window = image[y:y+window_height, x:x+window_width]
            
            # 对窗口进行推理
            pred = inference_single_image(window, net, device, transform)
            
            # 处理预测结果
            pred_image = process_prediction(pred, window)
            predictions.append((count, pred_image, (x, y, window_width, window_height)))
            count += 1
            
            # 移除9张图的限制，让它处理所有的滑动窗口
            # if count >= 9:  # 确保只生成9张图
            #     break
        # if count >= 9:
        #     break
    
    return predictions

def merge_predictions(predictions, original_image_shape, edge_fade=32):
    """
    合并多个预测窗口的结果
    Args:
        predictions: 预测结果列表，每个元素包含(index, image, (x, y, w, h))
        original_image_shape: 原始图像的形状 (height, width)
        edge_fade: 边缘淡化的像素数
    Returns:
        merged_image: 合并后的PIL图像
    """
    print(f"开始合并图像，原始图像尺寸: {original_image_shape}")
    
    # 创建与原图相同大小的空白图像（单通道）
    merged_array = np.zeros((original_image_shape[0], original_image_shape[1]), dtype=np.float32)
    count_array = np.zeros_like(merged_array)

    for idx, pred_image, (x, y, w, h) in predictions:
        print(f"处理第{idx}个窗口，位置: x={x}, y={y}, w={w}, h={h}")
        
        # 将PIL图像转换为numpy数组（确保是单通道）
        window_array = np.array(pred_image.convert('L'), dtype=np.float32) / 255.0
        if len(window_array.shape) > 2:
            window_array = window_array[:,:,0]  # 只取第一个通道
        
        # 创建边缘淡化的mask
        mask = np.ones((h, w), dtype=np.float32)
        mask[:edge_fade, :] *= np.linspace(0, 1, edge_fade)[:, np.newaxis]
        mask[-edge_fade:, :] *= np.linspace(1, 0, edge_fade)[:, np.newaxis]
        mask[:, :edge_fade] *= np.linspace(0, 1, edge_fade)[np.newaxis, :]
        mask[:, -edge_fade:] *= np.linspace(1, 0, edge_fade)[np.newaxis, :]
        
        # 应用mask
        window_array *= mask
        
        try:
            # 将处理后的窗口叠加到对应位置
            merged_array[y:y+h, x:x+w] = np.maximum(merged_array[y:y+h, x:x+w], window_array)
            count_array[y:y+h, x:x+w] += mask
        except Exception as e:
            print(f"错误发生在窗口合并过程：{e}")
            print(f"窗口数组形状: {window_array.shape}")
            print(f"目标区域形状: {merged_array[y:y+h, x:x+w].shape}")
            raise

    # 处理重叠区域
    count_array[count_array == 0] = 1
    merged_array = (merged_array * 255).astype(np.uint8)
    
    print(f"合并完成，最终图像形状: {merged_array.shape}")
    
    # 转换为PIL图像（确保是L模式的灰度图）
    merged_image = Image.fromarray(merged_array, mode='L')
    return merged_image

def main():
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results_sliding_window' + os.sep)
    merged_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results_merged' + os.sep)
    
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(f"找到{len(img_name_list)}张待处理图像")
    
    # --------- 2. model define ---------
    device, net = Get_net()
    
    # Create output directories
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    print(f"输出目录创建完成: {merged_dir}")
    
    img_name_list.sort()
    
    # --------- 3. inference for each image ---------
    for image_path in img_name_list:
        print(f"\n开始处理图像: {image_path}")
        
        # Read image
        image = read_image(image_path)
        print(f"原始图像尺寸: {image.shape}")
        
        # 使用滑动窗口进行推理
        predictions = sliding_window_inference(image, net, device, transform)
        print(f"生成{len(predictions)}个预测窗口")
        
        # 保存所有预测结果
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        try:
            # 合并预测结果并保存
            merged_image = merge_predictions(predictions, image.shape)
            merged_output_path = os.path.join(merged_dir, f"{base_filename}_merged.png")
            print(f"保存合并结果到: {merged_output_path}")
            merged_image.save(merged_output_path)
            print(f"图像{base_filename}处理完成")
        except Exception as e:
            print(f"处理图像{base_filename}时发生错误: {e}")
            continue
        
    print("\n所有图像处理完成，合并后的图像保存在: " + merged_dir)

if __name__ == "__main__":
    main()
    
    