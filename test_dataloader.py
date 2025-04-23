import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from data_loader import ColorJitter, RandomMaxFilter, SalObjDataset, RescaleT, ToTensorLab
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import ndimage

# 设置随机种子，确保结果可重复
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 确保figures文件夹存在
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Created directory: {figures_dir}")

def test_color_jitter_single_image():
    """测试单张图像的颜色抖动效果"""
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
    
    # 创建ColorJitter实例
    color_jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.6, hue=0.2)
    
    # 应用颜色抖动增强
    jittered_samples = []
    for i in range(4):  # 生成4个不同的增强示例
        # 重置随机种子以获得不同的增强效果
        random.seed(i + 100)
        jittered_sample = color_jitter(sample)
        jittered_samples.append(jittered_sample['image'])
    
    # 绘制原始图像和增强后的图像
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(sample['image'])
    plt.axis('off')
    
    for i, jittered_image in enumerate(jittered_samples):
        plt.subplot(2, 3, i + 2)
        plt.title(f"Augmented Sample #{i+1}")
        plt.imshow(jittered_image)
        plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(figures_dir, "color_jitter_demo.png")
    plt.savefig(output_path, dpi=200)  # 提高分辨率为200dpi
    plt.close()
    
    print(f"单图像测试完成! 结果已保存为 {output_path}")

def test_dataloader():
    """使用train.txt文件加载数据，并测试数据增强的效果"""
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
    
    # 创建使用颜色抖动的数据集
    color_jitter_dataset = SalObjDataset(
        img_name_list=tra_img_name_list[:5],  # 只使用前5张图片测试
        lbl_name_list=tra_lbl_name_list[:5],
        transform=transforms.Compose([
            RescaleT(512), 
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.6, hue=0.2),
            ToTensorLab(flag=0)
        ])
    )
    
    # 创建使用RandomMaxFilter的数据集
    max_filter_dataset = SalObjDataset(
        img_name_list=tra_img_name_list[:5],  # 只使用前5张图片测试
        lbl_name_list=tra_lbl_name_list[:5],
        transform=transforms.Compose([
            RescaleT(512), 
            RandomMaxFilter(
                num_regions=10,            # 处理区域数量
                kernel_size_range=(5, 15), # 滤波核大小范围
                threshold=0.2,             # 标签像素阈值
                apply_prob=1.0,            # 确保应用
            ),
            ToTensorLab(flag=0)
        ])
    )
    
    # 创建数据加载器
    batch_size = 2
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    color_jitter_loader = DataLoader(color_jitter_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    max_filter_loader = DataLoader(max_filter_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # 获取一个批次的数据
    normal_batch = next(iter(normal_loader))
    color_jitter_batch = next(iter(color_jitter_loader))
    max_filter_batch = next(iter(max_filter_loader))
    
    # 可视化对比结果 - 创建一个3行2列的图表，展示所有增强效果
    plt.figure(figsize=(15, 12))
    
    # 显示不使用数据增强的原始图像
    for i in range(batch_size):
        plt.subplot(3, batch_size, i+1)
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
    
    # 显示使用颜色抖动的图像
    for i in range(batch_size):
        plt.subplot(3, batch_size, batch_size+i+1)
        img = color_jitter_batch['image'][i].numpy()
        img = img.transpose((1, 2, 0))  # 转换通道顺序从[C,H,W]到[H,W,C]
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Color Jittered #{i+1}")
        plt.axis('off')
    
    # 显示使用RandomMaxFilter的图像
    for i in range(batch_size):
        plt.subplot(3, batch_size, 2*batch_size+i+1)
        img = max_filter_batch['image'][i].numpy()
        img = img.transpose((1, 2, 0))  # 转换通道顺序从[C,H,W]到[H,W,C]
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Max Filtered #{i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(figures_dir, "dataloader_augmentation_comparison.png")
    plt.savefig(output_path, dpi=200)  # 提高分辨率为200dpi
    plt.close()
    
    print(f"数据加载器增强效果对比测试完成! 结果已保存为 {output_path}")
    
    # 可选：创建单独的颜色抖动对比图
    plt.figure(figsize=(15, 8))
    
    # 显示不使用颜色抖动的图像
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
    
    # 显示使用颜色抖动的图像
    for i in range(batch_size):
        plt.subplot(2, batch_size, batch_size+i+1)
        img = color_jitter_batch['image'][i].numpy()
        img = img.transpose((1, 2, 0))  # 转换通道顺序从[C,H,W]到[H,W,C]
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Color Jittered #{i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    color_jitter_output_path = os.path.join(figures_dir, "dataloader_color_jitter_demo.png")
    plt.savefig(color_jitter_output_path, dpi=200)  # 提高分辨率为200dpi
    plt.close()

def test_random_max_filter():
    """测试RandomMaxFilter数据增强效果"""
    # 初始化train.txt路径
    train_txt_path = "train.txt"
    
    # 检查文件是否存在
    if not os.path.exists(train_txt_path):
        print(f"训练文件 {train_txt_path} 不存在，无法测试RandomMaxFilter!")
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

    if len(tra_img_name_list) == 0:
        print("未找到训练图像，请确保train.txt格式正确!")
        return

    # 从列表中找到第一个存在的图像和标签
    for i in range(min(10, len(tra_img_name_list))):
        if os.path.exists(tra_img_name_list[i]) and os.path.exists(tra_lbl_name_list[i]):
            # 加载图像和标签
            image = io.imread(tra_img_name_list[i])
            label = io.imread(tra_lbl_name_list[i])
            
            # 确保图像是float类型，值在[0,1]范围内
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
                
            # 处理标签，确保是3D数组
            if len(label.shape) == 2:
                label = label[:, :, np.newaxis]
            if label.dtype == np.uint8:
                label = label.astype(np.float32) / 255.0
                
            # 创建样本字典
            sample = {
                'imidx': np.array([i]),
                'image': image,
                'label': label
            }
            
            # 创建RandomMaxFilter实例 - 使用较大的参数以便观察效果
            max_filter = RandomMaxFilter(
                num_regions=20,             # 增加区域数量
                kernel_size_range=(7, 21),  # 增加滤波核大小
                threshold=0.1,              # 降低阈值以包含更多区域
                apply_prob=1.0,             # 确保应用
                need_regions=True           # 为了可视化，需要返回区域信息
            )
            
            # 应用大值滤波
            filtered_sample = max_filter(sample)
            
            # 提取滤波区域信息
            filtered_regions = filtered_sample['filtered_regions']
            
            # 可视化对比结果
            plt.figure(figsize=(15, 10))
            
            # 显示原始图像
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(sample['image'])
            plt.axis('off')
            
            # 显示标签
            plt.subplot(2, 2, 2)
            plt.title("Label (Mask Areas)")
            plt.imshow(sample['label'][:, :, 0], cmap='gray')
            plt.axis('off')
            
            # 显示滤波后的图像
            plt.subplot(2, 2, 3)
            plt.title("After Max Filter")
            plt.imshow(filtered_sample['image'])
            plt.axis('off')
            
            # 显示带有滤波区域标记的原始图像
            plt.subplot(2, 2, 4)
            plt.title(f"Filtered Regions (Total: {len(filtered_regions)})")
            plt.imshow(sample['image'])
            
            # 绘制滤波区域的边界框
            import matplotlib.patches as patches
            from matplotlib.colors import hsv_to_rgb
            
            ax = plt.gca()
            for idx, region in enumerate(filtered_regions):
                y_min, y_max, x_min, x_max, kernel_size = region
                
                # 使用HSV色彩空间生成不同颜色
                hue = (idx * 0.1) % 1.0  # 色调在0-1之间循环
                color = hsv_to_rgb((hue, 0.8, 0.9))  # 较高的饱和度和亮度
                
                # 创建矩形边框
                rect = patches.Rectangle(
                    (x_min, y_min), 
                    x_max - x_min, 
                    y_max - y_min, 
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none',
                    label=f'Region {idx+1}, Kernel: {kernel_size}'
                )
                ax.add_patch(rect)
                
                # 添加标签文本
                plt.text(
                    x_min, 
                    y_min - 5, 
                    f'{idx+1}',
                    color=color, 
                    fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
            
            plt.axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(figures_dir, "random_max_filter_demo.png")
            plt.savefig(output_path, dpi=200)  # 提高分辨率为200dpi
            plt.close()
            
            print(f"随机大值滤波测试完成! 结果已保存为 {output_path}")
            
            # 额外创建一个只展示滤波区域的大图
            plt.figure(figsize=(10, 10))
            plt.title(f"Max Filter Regions (Total: {len(filtered_regions)})")
            plt.imshow(sample['image'])
            
            ax = plt.gca()
            for idx, region in enumerate(filtered_regions):
                y_min, y_max, x_min, x_max, kernel_size = region
                
                # 使用HSV色彩空间生成不同颜色
                hue = (idx * 0.1) % 1.0
                color = hsv_to_rgb((hue, 0.8, 0.9))
                
                # 创建矩形边框
                rect = patches.Rectangle(
                    (x_min, y_min), 
                    x_max - x_min, 
                    y_max - y_min, 
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加标签文本
                plt.text(
                    x_min, 
                    y_min - 5, 
                    f'Region {idx+1}, Kernel: {kernel_size}',
                    color=color, 
                    fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
            
            plt.axis('off')
            detailed_output_path = os.path.join(figures_dir, "max_filter_regions_detail.png")
            plt.savefig(detailed_output_path, dpi=200)  # 提高分辨率为200dpi
            plt.close()
            
            print(f"大值滤波区域详细图已保存为 {detailed_output_path}")
            break
    else:
        print("未找到可用的图像和标签对进行测试!")

def main():
    """主函数，运行所有测试"""
    print("=== 测试1: 单张图像的颜色抖动 ===")
    test_color_jitter_single_image()
    
    print("\n=== 测试2: 随机大值滤波 ===")
    test_random_max_filter()
    
    print("\n=== 测试3: 数据加载器增强效果对比 ===")
    test_dataloader()

if __name__ == "__main__":
    main() 