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

pred_size = 1024

def normPRED(d):
    """归一化预测结果"""
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

def determine_model_type(model_folder_path):
    """根据文件夹路径确定模型类型"""
    folder_name = os.path.basename(model_folder_path.rstrip('/'))
    
    if 'u2net_grain' in folder_name.lower():
        return 'u2net_grain'
    elif 'u2netp_grain' in folder_name.lower():
        return 'u2netp_grain'
    elif 'u2netp' in folder_name.lower():
        return 'u2netp'
    elif 'u2net' in folder_name.lower():
        return 'u2net'
    else:
        # 默认返回u2net
        print(f"Warning: Cannot determine model type from folder name '{folder_name}', defaulting to 'u2net'")
        return 'u2net'

def load_model(model_type, model_path, device):
    """加载指定类型的模型"""
    if model_type == 'u2net':
        print("Loading U2NET model (173.6 MB)")
        net = U2NET(3, 1)
    elif model_type == 'u2netp':
        print("Loading U2NETP model (4.7 MB)")
        net = U2NETP(3, 1)
    elif model_type == 'u2net_grain':
        print("Loading U2NET_GRAIN model (optimized for grain boundary segmentation)")
        net = U2NET_GRAIN(3, 1)
    elif model_type == 'u2netp_grain':
        print("Loading U2NETP_GRAIN model (lightweight grain boundary segmentation)")
        net = U2NETP_GRAIN(3, 1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 加载模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    return net

def inference_with_model(image_path, net, model_type, device):
    """使用指定模型对图片进行推理"""
    # 图像预处理
    transform = transforms.Compose([
        RescaleT(pred_size),
        ToTensorLab(flag=0)
    ])
    
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
            # U2NET_GRAIN模型有6个输出 (d0-d5)
            d1, d2, d3, d4, d5, d6 = outputs
            pred = d1[:,0,:,:]
            del d1, d2, d3, d4, d5, d6
        
        pred = normPRED(pred)

        # 处理预测结果
        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy()
        im = Image.fromarray(predict_np*255).convert('RGB')
        
        # 调整回原始图像大小
        original_image = io.imread(image_path)
        imo = im.resize((original_image.shape[1], original_image.shape[0]), resample=Image.BILINEAR)
        
        return imo

def compare_models_in_folder(model_folder_path, test_image_path, output_dir, model_type=None):
    """
    对指定文件夹中的所有模型进行推理对比
    Args:
        model_folder_path: 模型文件夹路径 (如 'saved_models/u2net_grain' 或 'saved_models/u2net')
        test_image_path: 测试图片路径
        output_dir: 输出目录
        model_type: 模型类型 ('u2net', 'u2netp', 'u2net_grain', 'u2netp_grain')，如果为None则自动推断
    """
    # 检查输入参数
    if not os.path.exists(model_folder_path):
        raise ValueError(f"Model folder not found: {model_folder_path}")
    
    if not os.path.exists(test_image_path):
        raise ValueError(f"Test image not found: {test_image_path}")
    
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 确定模型类型
    if model_type is None:
        model_type = determine_model_type(model_folder_path)
        print(f"Model type auto-determined: {model_type}")
    else:
        print(f"Model type specified: {model_type}")
        # 验证模型类型是否有效
        valid_types = ['u2net', 'u2netp', 'u2net_grain', 'u2netp_grain']
        if model_type not in valid_types:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_types}")
    
    # 获取所有模型文件
    model_files = glob.glob(os.path.join(model_folder_path, '*.pth'))
    model_files.sort()  # 按文件名排序
    
    if not model_files:
        raise ValueError(f"No .pth model files found in {model_folder_path}")
    
    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 对每个模型进行推理
    successful_inferences = 0
    for model_path in model_files:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(output_dir, f"{model_name}.png")
        
        print(f"\nProcessing model: {model_name}")
        
        try:
            # 加载模型
            net = load_model(model_type, model_path, device)
            
            # 推理
            result_image = inference_with_model(test_image_path, net, model_type, device)
            
            # 保存结果
            result_image.save(output_path)
            print(f"✓ Result saved: {output_path}")
            
            successful_inferences += 1
            
            # 清理内存
            del net
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"✗ Error processing {model_name}: {str(e)}")
            continue
    
    print(f"\n=== Processing Summary ===")
    print(f"Total models: {len(model_files)}")
    print(f"Successful inferences: {successful_inferences}")
    print(f"Failed inferences: {len(model_files) - successful_inferences}")
    print(f"Results saved in: {output_dir}")

def main():
    """主函数 - 示例用法"""
    
    # 配置参数
    model_folder_path = "saved_models/u2net_grain_co"  # 或者 "saved_models/u2net"
    model_type = "u2net_grain"  # 明确指定模型类型：'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'
    test_image_path = "dataset/img/230226143240.jpg"  # 使用项目中的测试图片
    output_dir = "model_comparison_results"
    
    # 检查测试图片是否存在
    if not os.path.exists(test_image_path):
        print(f"Warning: Test image not found: {test_image_path}")
        print("Please update the test_image_path variable with a valid image path.")
        
        # 尝试查找项目中的示例图片 
        possible_test_dirs = [
            "U2net_data/train_data/im_aug",
            "test_data/test_images",
            "test_data",
            "U2net_data/test_images",
            "examples"
        ]
        
        for test_dir in possible_test_dirs:
            if os.path.exists(test_dir):
                images = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                        glob.glob(os.path.join(test_dir, "*.png")) + \
                        glob.glob(os.path.join(test_dir, "*.jpeg"))
                if images:
                    test_image_path = images[0]
                    print(f"Using found test image: {test_image_path}")
                    break
        else:
            print("No test images found. Please provide a valid test image path.")
            return
    
    try:
        # 执行模型对比
        compare_models_in_folder(
            model_folder_path=model_folder_path,
            test_image_path=test_image_path,
            output_dir=output_dir,
            model_type=model_type  # 明确传入模型类型
        )
        
        # 可以轻松对比不同类型的模型
        print("\n" + "="*50)
        print("如果要对比u2net模型，可以修改参数如下：")
        print("model_folder_path = 'saved_models/u2net'")
        print("model_type = 'u2net'")
        print("output_dir = 'u2net_comparison_results'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 