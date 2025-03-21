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
        model_type: 使用的模型类型，'u2net' 或 'u2netp'
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")

    # 加载模型
    if model_type == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif model_type == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    
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

            # 推理
            with torch.no_grad():
                d1,d2,d3,d4,d5,d6,d7 = net(inputs_test)
                pred = d1[:,0,:,:]
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

                del d1,d2,d3,d4,d5,d6,d7

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print(f"Processing completed. Results saved in {output_dir}")

if __name__ == "__main__":
    # 示例使用
    model_type = 'u2net'
    image_dir = 'test_images/'
    model_path = 'daizhuang_saved_models_512/u2net/u2net_best_acc_0.9340_epoch_91.pth'
    output_dir = 'test_results/'
    
    inference_folder(
        image_dir=image_dir,
        model_path=model_path,
        output_dir=output_dir,
        model_type=model_type
    ) 