import os
import torch
import torch.nn.functional as F
import numpy as np
import glob
import cv2
from skimage import io, transform
from torchvision import transforms

from data_loader import RescaleT
from data_loader import ToTensorLab

from u2net_classify import U2NetWithClassifier

# 定义类别名称
CLASS_NAMES = ['M', 'B', 'P']

def load_model(model_path, n_classes=3, device='cpu'):
    """加载训练好的模型"""
    # 创建模型实例
    model = U2NetWithClassifier(n_classes=n_classes, freeze_backbone=False)
    
    # 加载模型权重
    if os.path.exists(model_path):
        try:
            # 尝试直接加载模型
            print(f"尝试加载模型: {model_path} 到设备: {device}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"成功加载模型: {model_path} 到设备: {device}")
        except Exception as e:
            # 如果出错，尝试多种加载策略
            print(f"直接加载到 {device} 失败: {e}")
            print("尝试替代加载策略...")
            
            try:
                # 1. 尝试通过CPU加载
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"成功通过CPU中转加载模型")
            except Exception as e2:
                print(f"通过CPU加载失败: {e2}")
                
                try:
                    # 2. 尝试部分加载
                    print("尝试部分加载模型权重...")
                    state_dict = torch.load(model_path, map_location=device)
                    
                    # 检查是否包含分类器权重
                    if any("classifier" in k for k in state_dict.keys()):
                        print("模型包含分类器权重")
                        # 尝试部分加载
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        # 只加载U2NET部分
                        print("只加载U2NET主干部分")
                        u2net_state_dict = {k.replace("u2net.", ""): v for k in state_dict.keys() 
                                          if k.startswith("u2net")}
                        
                        if len(u2net_state_dict) == 0:
                            # 没有u2net前缀，可能是直接的u2net模型
                            u2net_state_dict = {"u2net." + k: v for k, v in state_dict.items()}
                        
                        model.load_state_dict(u2net_state_dict, strict=False)
                        print("成功加载U2NET部分权重")
                except Exception as e3:
                    print(f"所有加载策略均失败: {e3}")
                    print("使用随机初始化的模型，预测结果可能不准确")
    else:
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 将模型移动到指定设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path, target_size=512):
    """预处理输入图像"""
    # 读取图像
    image = io.imread(image_path)
    
    # 确保图像是三通道的
    if len(image.shape) < 3:
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)
    
    # 注意: 不进行归一化，因为ToTensorLab会处理这一步
    # 创建标签占位符
    label = np.zeros(image.shape[0:2])
    label = label[:, :, np.newaxis]  # 添加通道维度
    
    # 创建完整的样本字典
    sample = {
        'imidx': np.array([0]),
        'image': image,
        'label': label
    }
    
    # 应用转换
    transform_list = transforms.Compose([
        RescaleT(target_size),
        ToTensorLab(flag=0)  # ToTensorLab会处理归一化
    ])
    
    sample = transform_list(sample)
    
    # 获取处理后的图像，增加batch维度
    img_tensor = sample['image'].unsqueeze(0)
    
    return img_tensor

def predict(model, image_tensor, device='cpu'):
    """使用模型预测分割图和类别"""
    image_tensor = image_tensor.to(device)
    
    # 关闭梯度计算，提高推理速度
    with torch.no_grad():
        seg_outputs, cls_logits = model(image_tensor)
        
        # 获取分割预测结果
        seg_pred = seg_outputs[0]  # 使用d0作为最终分割输出
        
        # 标准化分割预测结果
        ma = torch.max(seg_pred)
        mi = torch.min(seg_pred)
        seg_pred = (seg_pred - mi) / (ma - mi)
        
        # 获取类别预测结果
        _, cls_pred = torch.max(cls_logits, 1)
        cls_prob = F.softmax(cls_logits, dim=1)
    
    # 处理分割输出 - 确保先移动到CPU
    seg_pred = seg_pred.squeeze().cpu().numpy()
    
    # 转换类别预测 - 确保先移动到CPU
    class_idx = cls_pred.cpu().item()
    class_name = CLASS_NAMES[class_idx]
    class_probs = cls_prob.squeeze().cpu().numpy()
    
    return {
        'segmentation': seg_pred,
        'class_id': class_idx,
        'class_name': class_name,
        'class_probabilities': class_probs
    }

def visualize_prediction(image_path, pred_result, output_path=None):
    """可视化预测结果"""
    try:
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            # 创建一个空白图像
            image = np.zeros((300, 300, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取分割掩码
        seg_mask = pred_result['segmentation']
        seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
        
        # 调整掩码大小以匹配原始图像
        seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]))
        
        # 创建彩色掩码以叠加
        color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        color_mask[seg_mask > 128] = [0, 255, 0]  # 绿色掩码
        
        # 叠加掩码和原始图像
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1, color_mask, alpha, 0)
        
        # 添加类别信息
        class_name = pred_result['class_name']
        class_prob = pred_result['class_probabilities'][pred_result['class_id']]
        
        # 准备可视化图像
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"class: {class_name}, cofidence: {class_prob:.2f}"
        cv2.putText(overlay, text, (50, 100), font, 3.0, (255, 0, 0), 5)
        
        # 保存结果
        if output_path:
            try:
                cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"结果已保存到: {output_path}")
            except Exception as e:
                print(f"保存结果时出错: {e}")
        
        return overlay
    except Exception as e:
        print(f"可视化预测结果时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个空图像
        return np.zeros((300, 300, 3), dtype=np.uint8)

def test_single_image(model_path, image_path, output_path=None):
    """测试单张图像"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
    print(f"使用设备: {device}")
    print("注意：CUDA训练的模型将被自动映射到当前设备")
    
    # 加载模型
    model = load_model(model_path, n_classes=len(CLASS_NAMES), device=device)
    if model is None:
        return
    
    # 预处理图像
    img_tensor = preprocess_image(image_path)
    
    # 进行预测
    pred_result = predict(model, img_tensor, device)
    
    # 打印预测结果
    print(f"预测类别: {pred_result['class_name']}")
    print(f"类别ID: {pred_result['class_id']}")
    print(f"类别概率: {pred_result['class_probabilities']}")
    
    # 可视化并保存结果
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + "_result.png"
    
    visualize_prediction(image_path, pred_result, output_path)

def test_batch(model_path, image_dir, output_dir, ext='.jpg'):
    """批量测试图像"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图像文件
    image_paths = glob.glob(os.path.join(image_dir, f'*{ext}'))
    
    if not image_paths:
        print(f"未在 {image_dir} 中找到 {ext} 格式的图像")
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
    print(f"使用设备: {device}")
    print("注意：CUDA训练的模型将被自动映射到当前设备")
    
    # 加载模型
    model = load_model(model_path, n_classes=len(CLASS_NAMES), device=device)
    if model is None:
        return
    
    # 处理每张图像
    for img_path in image_paths:
        print(f"处理图像: {img_path}")
        
        # 获取输出路径
        img_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + "_result.png")
        
        # 预处理图像
        img_tensor = preprocess_image(img_path)
        
        # 进行预测
        pred_result = predict(model, img_tensor, device)
        
        # 可视化并保存结果
        visualize_prediction(img_path, pred_result, out_path)
    
    print(f"所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    # 参数设置
    # 注意：无论模型是在哪个设备上训练的，代码都会自动将其映射到当前可用设备(CUDA/MPS/CPU)
    model_path = "daizhuang_saved_models_classify/u2net_cls/u2net_cls_best_clsacc_0.8136_epoch_5.pth"  # 修改为您的模型路径
    
    # 确保测试目录存在
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
        print("创建了test_images目录，请在此放入测试图像")
    
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"警告：模型文件 {model_path} 不存在，请检查路径")
    
    # 测试单张图像 - 取消注释下面一行并指定图像路径进行单张测试
    # test_single_image(model_path, "test_images/your_image.jpg")
    
    # 批量测试图像
    test_batch(model_path, "test_images", "test_results", ext='.jpg') 