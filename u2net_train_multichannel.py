import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
# 导入新增的多通道数据集和转换类
from data_loader import MultiChannelSalObjDataset
from data_loader import MultiChannelToTensorLab

from model import U2NET
from model import U2NETP

# --------- 配置参数 ---------
class Config:
    def __init__(self):
        # 基本配置
        self.model_name = 'u2net'  # 'u2netp'
        
        # 数据路径
        self.data_dir = "/Volumes/data1/JH/projects/ttprocess/train_/"
        self.tra_image_dir = os.path.join('images' + os.sep)
        self.tra_label_dir = os.path.join('masks' + os.sep)
        self.image_ext = '.jpg'
        self.label_ext = '.png'
        
        # 模型保存路径
        self.model_dir = os.path.join(os.getcwd(), 'tuotan3_saved_models_multichannel', self.model_name + os.sep)
        
        # 预训练模型 - 如果使用预训练的单通道模型，这里设置路径
        self.pretrained_model_path = "tuotan3_saved_models_multichannel/u2net/u2net_best_accuracy_0.9380_epoch_3.pth"
        self.start_epoch = 0  # 从哪个epoch开始训练
        
        # 训练参数
        self.epoch_num = 100000
        self.batch_size_train = 8
        self.batch_size_val = 1
        self.save_freq = 2000  # 保存模型的频率
        
        # 分割通道数量
        self.num_classes = 3  # 分割的类别数（通道数）
        
        # 边界损失参数
        self.boundary_weight = 0.5  # 边界损失的权重
        
        # 优化器参数
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-08
        self.weight_decay = 0

# --------- 1. 定义损失函数 ---------
class LossFunctions:
    def __init__(self):
        self.bce_loss = nn.BCELoss(size_average=True)
        
    @staticmethod
    def visualize_boundaries(pred, target, threshold=0.5, save_path=None):
        """
        可视化预测和目标的边界
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 目标掩码 [B, C, H, W]
            threshold: 二值化阈值
            save_path: 保存路径，如果为None则不保存
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 取第一个样本进行可视化
        pred = pred[0].detach().cpu().numpy()  # [C, H, W]
        target = target[0].detach().cpu().numpy()  # [C, H, W]
        
        num_channels = min(pred.shape[0], 3)  # 最多显示3个通道
        fig, axes = plt.subplots(num_channels, 3, figsize=(12, 4*num_channels))
        
        # 如果只有一个通道，确保axes是二维的
        if num_channels == 1:
            axes = axes.reshape(1, -1)
        
        titles = ['预测掩码', '真实掩码', '边界对比']
        
        for c in range(num_channels):
            # 获取当前通道
            pred_c = pred[c]  # [H, W]
            target_c = target[c]  # [H, W]
            
            # 二值化预测
            pred_binary = (pred_c > threshold).astype(np.float32)
            
            # 提取边界 (使用numpy操作而不是torch)
            from scipy import ndimage
            pred_boundary = ndimage.sobel(pred_binary, axis=0)**2 + ndimage.sobel(pred_binary, axis=1)**2
            pred_boundary = np.sqrt(pred_boundary)
            
            target_boundary = ndimage.sobel(target_c, axis=0)**2 + ndimage.sobel(target_c, axis=1)**2
            target_boundary = np.sqrt(target_boundary)
            
            # 显示预测掩码
            axes[c, 0].imshow(pred_binary, cmap='gray')
            axes[c, 0].set_title(f'通道 {c} {titles[0]}')
            axes[c, 0].axis('off')
            
            # 显示真实掩码
            axes[c, 1].imshow(target_c, cmap='gray')
            axes[c, 1].set_title(f'通道 {c} {titles[1]}')
            axes[c, 1].axis('off')
            
            # 显示边界对比 (红色：预测边界，绿色：真实边界)
            # 创建RGB图像
            boundary_vis = np.zeros((pred_boundary.shape[0], pred_boundary.shape[1], 3))
            boundary_vis[:, :, 0] = pred_boundary / max(pred_boundary.max(), 1e-8)  # 红色
            boundary_vis[:, :, 1] = target_boundary / max(target_boundary.max(), 1e-8)  # 绿色
            
            axes[c, 2].imshow(boundary_vis)
            axes[c, 2].set_title(f'通道 {c} {titles[2]}')
            axes[c, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
    @staticmethod
    def extract_boundaries(mask, kernel_size=3):
        """
        使用Sobel算子提取分割掩码的边界
        Args:
            mask: 输入掩码 [B, 1, H, W]
            kernel_size: Sobel算子的大小
        Returns:
            边界图 [B, 1, H, W]
        """
        # 定义Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=torch.float32, device=mask.device)
        sobel_y = torch.tensor([[-1, -2, -1], 
                               [0, 0, 0], 
                               [1, 2, 1]], dtype=torch.float32, device=mask.device)
        
        # 调整Sobel算子的形状用于卷积
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        
        # 计算梯度
        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)
        
        # 计算梯度幅值 (边界)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return edge
    
    def compute_boundary_loss(self, pred, target):
        """
        计算预测和目标边界之间的损失
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 目标掩码 [B, C, H, W]
        Returns:
            边界损失
        """
        batch_size, num_channels = pred.shape[0], pred.shape[1]
        boundary_loss = 0.0
        
        # 跳过背景通道(0)，计算所有前景通道的边界损失
        foreground_channels = 0  # 记录前景通道数量
        for c in range(1, num_channels):  # 从索引1开始，跳过背景通道0
            # 提取当前通道
            pred_c = pred[:, c:c+1, :, :]  # [B, 1, H, W]
            target_c = target[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # 阈值化
            pred_binary = (pred_c > 0.5).float()
            
            # 提取边界
            pred_boundaries = self.extract_boundaries(pred_binary)
            target_boundaries = self.extract_boundaries(target_c)
            
            # 计算边界损失 (使用L1损失)
            channel_loss = F.l1_loss(pred_boundaries, target_boundaries)
            boundary_loss += channel_loss
            foreground_channels += 1
        
        # 通道数量归一化
        if foreground_channels > 0:
            boundary_loss = boundary_loss / foreground_channels
        
        return boundary_loss
        
    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v, boundary_weight=0.5):
        """
        计算多级输出的BCE损失，支持多通道，并添加边界损失
        Args:
            d0-d6: 模型各级输出
            labels_v: 真实标签
            boundary_weight: 边界损失的权重
        """
        # 对每个输出通道分别计算BCE损失
        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)
        
        # 计算基础BCE损失
        bce_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        
        # 计算边界损失 (仅对主输出d0计算)
        boundary_loss = self.compute_boundary_loss(d0, labels_v)
        
        # 总损失 = BCE损失 + 边界损失*权重
        total_loss = bce_loss + boundary_weight * boundary_loss
        
        # 输出损失信息
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f, boundary: %3f"%(
            loss0.data.item(), loss1.data.item(), loss2.data.item(), 
            loss3.data.item(), loss4.data.item(), loss5.data.item(), 
            loss6.data.item(), boundary_loss.data.item()))
        
        # 输出通道信息
        # num_channels = d0.shape[1]
        # print(f"处理了 {num_channels} 个通道，其中 {num_channels-1} 个前景通道的边界损失被计算")
        
        return loss0, total_loss
    
    @staticmethod
    def calculate_multichannel_accuracy(pred, target, threshold=0.5):
        """
        计算多通道预测的准确率
        Args:
            pred: 预测的输出 (已经经过sigmoid) [B, C, H, W]
            target: 真实标签 [B, C, H, W]
            threshold: 二值化阈值
        Returns:
            accuracy: 准确率
        """
        pred = (pred > threshold).float()
        target = (target > threshold).float()
        
        # 计算每个通道的准确率然后平均
        correct = (pred == target).float().sum()
        total = target.numel()
        
        return (correct / total).item()

# --------- 2. 数据准备 ---------
class DatasetPreparation:
    def __init__(self, config):
        self.config = config
        
    def get_data_paths(self):
        """获取训练数据的路径列表"""
        # 获取所有图像路径
        tra_img_name_list = glob.glob(self.config.data_dir + self.config.tra_image_dir + '*' + self.config.image_ext)
        
        # 构建对应的标签路径
        tra_lbl_name_list = []
        for img_path in tra_img_name_list:
            img_name = img_path.split(os.sep)[-1]
            aaa = img_name.split(".")
            bbb = aaa[0:-1]
            imidx = bbb[0]
            for i in range(1, len(bbb)):
                imidx = imidx + "." + bbb[i]
            tra_lbl_name_list.append(self.config.data_dir + self.config.tra_label_dir + imidx + self.config.label_ext)
            
        print("---")
        print("train images: ", len(tra_img_name_list))
        print("train labels: ", len(tra_lbl_name_list))
        print("---")
        
        return tra_img_name_list, tra_lbl_name_list
        
    def create_dataloader(self, img_name_list, lbl_name_list, batch_size, shuffle=True):
        """创建多通道分割的DataLoader"""
        dataset = MultiChannelSalObjDataset(
            img_name_list=img_name_list,
            lbl_name_list=lbl_name_list,
            transform=transforms.Compose([
                RescaleT(512),
                # RandomCrop(460),
                MultiChannelToTensorLab(flag=0, num_channels=self.config.num_classes)
            ]),
            num_channels=self.config.num_classes  # 设置通道数
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        return dataloader

# --------- 3. 模型定义 ---------
class ModelSetup:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        
    def _get_device(self):
        """获取可用的计算设备"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
            
    def create_model(self):
        """创建并初始化模型，支持多通道输出"""
        print(f"Using device: {self.device}")
        
        # 根据模型名称创建网络，注意这里输出通道是多通道的
        if self.config.model_name == 'u2net':
            net = U2NET(3, self.config.num_classes)
        elif self.config.model_name == 'u2netp':
            net = U2NETP(3, self.config.num_classes)
        else:
            raise ValueError(f"Unknown model name: {self.config.model_name}")
            
        # 加载预训练模型
        if os.path.exists(self.config.pretrained_model_path):
            print(f"Loading pretrained model from {self.config.pretrained_model_path}")
            # 如果是从单通道模型迁移到多通道模型，需要特殊处理
            if self.config.num_classes > 1:
                # 加载预训练权重
                pretrained_dict = torch.load(self.config.pretrained_model_path, map_location=self.device)
                model_dict = net.state_dict()
                
                # 筛选并调整合适的参数
                # 1. 过滤与预训练模型不兼容的参数
                # 2. 特殊处理最后输出层的参数
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                   if k in model_dict and 'outconv.weight' not in k and 'outconv.bias' not in k 
                                   and 'side1.weight' not in k and 'side1.bias' not in k
                                   and 'side2.weight' not in k and 'side2.bias' not in k
                                   and 'side3.weight' not in k and 'side3.bias' not in k
                                   and 'side4.weight' not in k and 'side4.bias' not in k
                                   and 'side5.weight' not in k and 'side5.bias' not in k
                                   and 'side6.weight' not in k and 'side6.bias' not in k}
                
                # 更新参数字典
                model_dict.update(pretrained_dict)
                
                # 加载经过筛选的参数
                net.load_state_dict(model_dict)
                print("Pretrained model partially loaded (backbone only)!")
            else:
                # 如果通道数相同，可以直接加载
                net.load_state_dict(torch.load(self.config.pretrained_model_path, map_location=self.device))
                print("Pretrained model loaded successfully!")
        else:
            print(f"Pretrained model not found at {self.config.pretrained_model_path}, starting from scratch")
            self.config.start_epoch = 0
            
        # 移动模型到指定设备
        net.to(self.device)
        return net
        
    def create_optimizer(self, model):
        """创建优化器"""
        return optim.Adam(
            model.parameters(), 
            lr=self.config.lr, 
            betas=self.config.betas, 
            eps=self.config.eps, 
            weight_decay=self.config.weight_decay
        )

# --------- 4. 训练流程 ---------
class Trainer:
    def __init__(self, config, model, optimizer, dataloader, loss_funcs, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.loss_funcs = loss_funcs
        self.device = device
        self.best_accuracy = 0.0
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)  # 从配置获取边界损失权重，默认0.5
        
    def train(self):
        """训练模型"""
        print("---start training...")
        print(f"Using boundary loss with weight: {self.boundary_weight}")
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
        
        for epoch in range(self.config.start_epoch, self.config.epoch_num):
            self.model.train()
            epoch_loss = 0.0
            epoch_tar_loss = 0.0
            epoch_accuracy = 0.0
            batch_count = 0
            
            for i, data in enumerate(self.dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1
                
                # 准备数据
                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                inputs_v = inputs.to(self.device)
                labels_v = labels.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播、计算损失、反向传播、优化
                d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
                loss2, loss = self.loss_funcs.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, self.boundary_weight)
                
                # 计算评价指标
                batch_accuracy = self.loss_funcs.calculate_multichannel_accuracy(d0, labels_v)
                epoch_accuracy += batch_accuracy
                
                loss.backward()
                self.optimizer.step()
                
                # 记录统计数据
                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()
                epoch_loss += loss.data.item()
                epoch_tar_loss += loss2.data.item()
                batch_count += 1
                
                # 定期保存模型
                if ite_num % self.config.save_freq == 0:
                    torch.save(
                        self.model.state_dict(), 
                        os.path.join(
                            self.config.model_dir, 
                            f"{self.config.model_name}_bce_itr_{ite_num}_train_{running_loss / ite_num4val:.4f}.pth"
                        )
                    )
                    
                    # 保存边界可视化（在删除变量之前）
                    vis_path = os.path.join(
                        self.config.model_dir, 
                        f"boundaries_itr_{ite_num}.png"
                    )
                    self.loss_funcs.visualize_boundaries(d0.detach(), labels_v.detach(), save_path=vis_path)
                    
                    running_loss = 0.0
                    running_tar_loss = 0.0
                    self.model.train()  # 继续训练
                    ite_num4val = 0
                
                # 释放内存（移动到边界可视化之后）
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss
                
                # 输出训练进度
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, accuracy: %3f\n" % (
                    epoch + 1, self.config.epoch_num, (i + 1) * self.config.batch_size_train, 
                    len(self.dataloader.dataset), ite_num, 
                    running_loss / ite_num4val, running_tar_loss / ite_num4val, batch_accuracy
                ))
            
            # 计算每个epoch的平均统计数据
            avg_epoch_loss = epoch_loss / batch_count
            avg_epoch_tar_loss = epoch_tar_loss / batch_count
            avg_epoch_accuracy = epoch_accuracy / batch_count
            
            # 输出epoch统计信息
            print(f"Epoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_epoch_loss:.4f}")
            print(f"Average Accuracy: {avg_epoch_accuracy:.4f}")
            
            # 每5个epoch保存一次模型
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch+1} with loss {avg_epoch_loss:.4f}")
                
                # 保存边界可视化
                self.model.eval()
                with torch.no_grad():
                    # 获取一个批次用于可视化
                    sample_data = next(iter(self.dataloader))
                    inputs, labels = sample_data['image'], sample_data['label']
                    inputs = inputs.type(torch.FloatTensor)
                    labels = labels.type(torch.FloatTensor)
                    inputs_v = inputs.to(self.device)
                    labels_v = labels.to(self.device)
                    
                    d0, _, _, _, _, _, _ = self.model(inputs_v)
                    
                    # 保存边界可视化
                    vis_path = os.path.join(
                        self.config.model_dir, 
                        f"boundaries_epoch_{epoch+1}.png"
                    )
                    self.loss_funcs.visualize_boundaries(d0, labels_v, save_path=vis_path)
                self.model.train()  # 恢复训练模式
            
            # 保存最佳模型
            if avg_epoch_accuracy > self.best_accuracy:
                self.best_accuracy = avg_epoch_accuracy
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_best_accuracy_{self.best_accuracy:.4f}_epoch_{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"New best accuracy achieved! Model saved with accuracy: {self.best_accuracy:.4f}")

# --------- 主函数 ---------
def main():
    # 初始化配置
    config = Config()
    
    # 确保模型保存目录存在
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    
    # 准备数据
    data_prep = DatasetPreparation(config)
    tra_img_name_list, tra_lbl_name_list = data_prep.get_data_paths()
    train_num = len(tra_img_name_list)
    dataloader = data_prep.create_dataloader(
        tra_img_name_list, 
        tra_lbl_name_list, 
        config.batch_size_train
    )
    
    # 设置模型
    model_setup = ModelSetup(config)
    model = model_setup.create_model()
    optimizer = model_setup.create_optimizer(model)
    
    # 设置损失函数
    loss_funcs = LossFunctions()
    
    # 训练模型
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        loss_funcs=loss_funcs,
        device=model_setup.device
    )
    
    trainer.train()

if __name__ == '__main__':
    main() 