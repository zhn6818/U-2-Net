import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from data_loader import RandomScratch, SalObjDataset, RescaleT, ToTensorLab
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 设置随机种子，确保结果可重复
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 确保figures文件夹存在
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Created directory: {figures_dir}")

def test_random_scratch_single_image():
    """测试单张图像的随机划痕效果"""
    # 查找测试图像
    image_dirs = ['./U2net_data/train_data/im_aug', './U2net_data768/train_data/im_aug']
    test_image_path = None
    
    for dir_path in image_dirs:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        test_image_path = os.path.join(root, file)
                        break
                if test_image_path:
                    break
        if test_image_path:
            break
    
    if not test_image_path:
        print("未找到测试图像，请确保项目目录中有测试图像。")
        return
    
    print(f"使用测试图像: {test_image_path}")
    
    # 加载测试图像
    image = io.imread(test_image_path)
    
    # 确保图像是float类型，值在[0,1]范围内
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # 创建样本字典
    sample = {
        'imidx': np.array([0]),
        'image': image,
        'label': np.zeros((image.shape[0], image.shape[1], 1))  # 创建一个空标签
    }
    
    # 创建不同配置的RandomScratch实例
    # 测试1: 单条极细划痕
    scratch_thin = RandomScratch(
        line_width_range=(1, 1),  # 固定为1像素宽度
        color_range=(0, 20),
        apply_prob=1.0,  # 为了测试目的设置为100%应用
        num_lines=1
    )
    
    # 测试2: 单条细划痕
    scratch_thick = RandomScratch(
        line_width_range=(1, 2),  # 修改为更细的线条
        color_range=(0, 20),
        apply_prob=1.0,
        num_lines=1
    )
    
    # 测试3: 多条划痕，灰色
    scratch_multi = RandomScratch(
        line_width_range=(1, 2),  # 修改为更细的线条
        color_range=(20, 50),
        apply_prob=1.0,
        num_lines=3
    )
    
    # 测试4: 多条划痕，深黑色
    scratch_multi_dark = RandomScratch(
        line_width_range=(1, 2),  # 修改为更细的线条
        color_range=(0, 10),
        apply_prob=1.0,
        num_lines=5
    )
    
    # 应用不同的划痕增强
    scratched_samples = []
    scratch_configs = [scratch_thin, scratch_thick, scratch_multi, scratch_multi_dark]
    
    for i, scratch_config in enumerate(scratch_configs):
        # 重置随机种子以获得不同的增强效果
        random.seed(i + 100)
        scratched_sample = scratch_config(sample)
        scratched_samples.append(scratched_sample['image'])
    
    # 绘制原始图像和增强后的图像
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(sample['image'])
    plt.axis('off')
    
    titles = [
        "Single 1px Scratch",
        "Single Thin Scratch",
        "Multiple Gray Scratches",
        "Multiple Dark Scratches"
    ]
    
    for i, scratched_image in enumerate(scratched_samples):
        plt.subplot(2, 3, i + 2)
        plt.title(titles[i])
        plt.imshow(scratched_image)
        plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(figures_dir, "random_scratch_demo.png")
    plt.savefig(output_path, dpi=200)  # 提高分辨率为200dpi
    plt.close()
    
    print(f"单图像划痕测试完成! 结果已保存为 {output_path}")

def test_dataloader_with_scratch():
    """使用train.txt文件加载数据，并测试划痕数据增强的效果"""
    # 初始化train.txt路径
    train_txt_path = "train.txt"
    
    # 检查文件是否存在
    if not os.path.exists(train_txt_path):
        print(f"训练文件 {train_txt_path} 不存在，无法测试数据加载器!")
        return
        
    tra_img_name_list = []
    tra_lbl_name_list = []

    # 读取train.txt文件，每行按空格分割为图像路径和标签路径
    with open(train_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保行不为空
                parts = line.split()
                if len(parts) == 2:  # 确保每行有两部分
                    img_path, lbl_path = parts
                    tra_img_name_list.append(img_path)
                    tra_lbl_name_list.append(lbl_path)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    if len(tra_img_name_list) == 0:
        print("未找到训练图像，请确保train.txt格式正确!")
        return

    # 创建不使用数据增强的标准数据集
    normal_dataset = SalObjDataset(
        img_name_list=tra_img_name_list[:5],  # 只使用前5张图片测试
        lbl_name_list=tra_lbl_name_list[:5],
        transform=transforms.Compose([
            RescaleT(512), 
            ToTensorLab(flag=0)
        ])
    )
    
    # 创建使用划痕增强的数据集
    scratch_dataset = SalObjDataset(
        img_name_list=tra_img_name_list[:5],  # 只使用前5张图片测试
        lbl_name_list=tra_lbl_name_list[:5],
        transform=transforms.Compose([
            RescaleT(512), 
            RandomScratch(
                line_width_range=(1, 2),  # 修改为1像素宽的线条
                color_range=(0, 30),
                apply_prob=1.0,  # 为了测试目的设置为100%应用
                num_lines=2
            ),
            ToTensorLab(flag=0)
        ])
    )
    
    # 创建数据加载器
    batch_size = 3
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    scratch_loader = DataLoader(scratch_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # 获取一个批次的数据
    normal_batch = next(iter(normal_loader))
    scratch_batch = next(iter(scratch_loader))
    
    # 可视化对比结果
    plt.figure(figsize=(15, 10))
    
    # 显示不使用数据增强的原始图像
    for i in range(batch_size):
        plt.subplot(2, batch_size, i+1)
        img = normal_batch['image'][i].numpy()
        img = img.transpose((1, 2, 0))  # 转换通道顺序从[C,H,W]到[H,W,C]
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Original #{i+1}")
        plt.axis('off')
    
    # 显示使用划痕增强的图像
    for i in range(batch_size):
        plt.subplot(2, batch_size, batch_size+i+1)
        img = scratch_batch['image'][i].numpy()
        img = img.transpose((1, 2, 0))  # 转换通道顺序从[C,H,W]到[H,W,C]
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Scratched #{i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(figures_dir, "dataloader_scratch_comparison.png")
    plt.savefig(output_path, dpi=200)  # 提高分辨率为200dpi
    plt.close()
    
    print(f"数据加载器划痕增强对比测试完成! 结果已保存为 {output_path}")

def main():
    """主函数，运行所有测试"""
    print("=== 测试1: 单张图像的随机划痕效果 ===")
    test_random_scratch_single_image()
    
    print("\n=== 测试2: 数据加载器划痕增强效果对比 ===")
    test_dataloader_with_scratch()

if __name__ == "__main__":
    main() 