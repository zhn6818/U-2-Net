import os
import glob
import torch
import numpy as np
from PIL import Image
from skimage import io
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import U2NET
from model import U2NETP
from model import U2NET_GRAIN
from model import U2NETP_GRAIN
from data_loader import RescaleT
from data_loader import ToTensorLab

pred_size = 512

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def get_device():
    """获取可用的设备类型：CUDA、MPS 或 CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def read_image(image_path):
    """读取图片并进行基础处理"""
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    return image

def inference_folder(image_dir, model_path, output_dir, model_type='u2net'):
    """
    对文件夹中的所有图片进行推理
    Args:
        image_dir: 输入图片文件夹路径
        model_path: 模型权重文件路径
        output_dir: 输出目录
        model_type: 使用的模型类型，'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")

    # 加载模型
    if model_type == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif model_type == 'u2netp':
        print("...load U2NETP---4.7 MB")
        net = U2NETP(3,1)
    elif model_type == 'u2net_grain':
        print("...load U2NET_GRAIN---optimized for grain boundary segmentation")
        net = U2NET_GRAIN(3,1)
    elif model_type == 'u2netp_grain':
        print("...load U2NETP_GRAIN---lightweight grain boundary segmentation")
        net = U2NETP_GRAIN(3,1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 加载模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # 图像预处理
    transform = transforms.Compose([
        RescaleT(pred_size),
        ToTensorLab(flag=0)
    ])

    # 获取所有图片文件
    img_name_list = glob.glob(os.path.join(image_dir, '*.*'))
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_name_list = [f for f in img_name_list if os.path.splitext(f)[1].lower() in supported_formats]
    
    print(f"Found {len(img_name_list)} images in {image_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图片
    for image_path in img_name_list:
        print(f"Processing {image_path}")
        try:
            # 读取和处理图像
            image = read_image(image_path)
            label = np.zeros(image.shape[0:2])
            label = label[:,:,np.newaxis]  # Add channel dimension
            
            # 准备输入数据
            sample = {'imidx': np.array([0]), 'image': image, 'label': label}
            sample = transform(sample)
            inputs_test = sample['image']
            inputs_test = inputs_test.unsqueeze(0)
            inputs_test = inputs_test.type(torch.FloatTensor)
            inputs_test = Variable(inputs_test).to(device)

            # 推理 - 根据模型类型处理不同的输出
            with torch.no_grad():
                outputs = net(inputs_test)
                
                if model_type in ['u2net', 'u2netp']:
                    # 原始U2NET模型有7个输出
                    d1, d2, d3, d4, d5, d6, d7 = outputs
                    pred = d1[:,0,:,:]
                    # 清理内存
                    del d1, d2, d3, d4, d5, d6, d7
                else:
                    # U2NET_GRAIN模型有5个输出
                    d1, d2, d3, d4, d5 = outputs
                    pred = d1[:,0,:,:]
                    # 清理内存
                    del d1, d2, d3, d4, d5
                
                pred = normPRED(pred)

                # 处理预测结果
                predict = pred.squeeze()
                predict_np = predict.cpu().data.numpy()
                im = Image.fromarray(predict_np*255).convert('RGB')
                
                # 调整回原始图像大小
                image = io.imread(image_path)
                imo = im.resize((image.shape[1],image.shape[0]), resample=Image.BILINEAR)

                # 保存结果
                img_name = os.path.splitext(os.path.basename(image_path))[0]
                imo.save(os.path.join(output_dir, f"{img_name}.png"))

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print(f"Processing completed. Results saved in {output_dir}")

def inference_single_image(image_path, model_path, output_path, model_type='u2net'):
    """
    对单张图片进行推理
    Args:
        image_path: 输入图片路径
        model_path: 模型权重文件路径
        output_path: 输出图片路径
        model_type: 使用的模型类型
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")

    # 加载模型
    if model_type == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif model_type == 'u2netp':
        print("...load U2NETP---4.7 MB")
        net = U2NETP(3,1)
    elif model_type == 'u2net_grain':
        print("...load U2NET_GRAIN---optimized for grain boundary segmentation")
        net = U2NET_GRAIN(3,1)
    elif model_type == 'u2netp_grain':
        print("...load U2NETP_GRAIN---lightweight grain boundary segmentation")
        net = U2NETP_GRAIN(3,1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 加载模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # 图像预处理
    transform = transforms.Compose([
        RescaleT(pred_size),
        ToTensorLab(flag=0)
    ])

    print(f"Processing {image_path}")
    
    # 读取和处理图像
    image = read_image(image_path)
    label = np.zeros(image.shape[0:2])
    label = label[:,:,np.newaxis]
    
    # 准备输入数据
    sample = {'imidx': np.array([0]), 'image': image, 'label': label}
    sample = transform(sample)
    inputs_test = sample['image']
    inputs_test = inputs_test.unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)
    inputs_test = Variable(inputs_test).to(device)

    # 推理
    with torch.no_grad():
        outputs = net(inputs_test)
        
        if model_type in ['u2net', 'u2netp']:
            # 原始U2NET模型有7个输出
            d1, d2, d3, d4, d5, d6, d7 = outputs
            pred = d1[:,0,:,:]
            del d1, d2, d3, d4, d5, d6, d7
        else:
            # U2NET_GRAIN模型有5个输出
            d1, d2, d3, d4, d5 = outputs
            pred = d1[:,0,:,:]
            del d1, d2, d3, d4, d5
        
        pred = normPRED(pred)

        # 处理预测结果
        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy()
        im = Image.fromarray(predict_np*255).convert('RGB')
        
        # 调整回原始图像大小
        original_image = io.imread(image_path)
        imo = im.resize((original_image.shape[1], original_image.shape[0]), resample=Image.BILINEAR)

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imo.save(output_path)
        print(f"Result saved to {output_path}")

if __name__ == "__main__":
    # 示例使用
    model_type = 'u2net_grain'  # 可选: 'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'
    image_dir = '/Volumes/data1/JH/projects/JLD_imgprocess/dataset/img/'
    model_path = 'saved_models/u2net_grain/u2net_grain_best_acc_0.9053_epoch_32.pth'
    output_dir = 'test_results/'
    
    # 批量推理
    inference_folder(
        image_dir=image_dir,
        model_path=model_path,
        output_dir=output_dir,
        model_type=model_type
    )
    
    # 或者单张图片推理
    # inference_single_image(
    #     image_path='path/to/single/image.jpg',
    #     model_path=model_path,
    #     output_path='output/result.png',
    #     model_type=model_type
    # ) 