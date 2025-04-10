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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import glob
import os
import argparse


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
        self.data_dir = "/data1/zhn/JZ/train_512/"
        self.tra_image_dir = os.path.join('imgs' + os.sep)
        self.tra_label_dir = os.path.join('masks' + os.sep)
        self.image_ext = '.jpg'
        self.label_ext = '.png'
        
        # 模型保存路径
        self.model_dir = os.path.join(os.getcwd(), 'JZ_saved_models_multichannel_512', self.model_name + os.sep)
        
        # 预训练模型 - 如果使用预训练的单通道模型，这里设置路径
        self.pretrained_model_path = ""
        self.start_epoch = 0  # 从哪个epoch开始训练
        
        # 训练参数
        self.epoch_num = 100000
        self.batch_size_train = 2
        self.batch_size_val = 1
        self.save_freq = 2000  # 保存模型的频率
        
        # 分割通道数量
        self.num_classes = 7  # 分割的类别数（通道数）
        
        # 边界损失参数
        self.use_boundary_loss = True  # 是否使用边界损失
        self.boundary_weight = 0.5  # 边界损失的权重
        
        # 优化器参数
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-08
        self.weight_decay = 0
        
        # 多GPU训练参数
        self.distributed = False  # 是否使用分布式训练
        self.local_rank = -1  # 本地GPU的rank，由启动脚本传入
        self.world_size = 1  # 总GPU数量
        self.gpu = 0  # 默认使用的GPU ID
        self.sync_bn = False  # 是否使用SyncBatchNorm

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
        else:
            boundary_loss = torch.tensor(0.0, device=pred.device)
        
        return boundary_loss
        
    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v, boundary_weight=0.5, use_boundary_loss=True):
        """
        计算多级输出的BCE损失，支持多通道，并添加边界损失
        Args:
            d0-d6: 模型各级输出
            labels_v: 真实标签
            boundary_weight: 边界损失的权重
            use_boundary_loss: 是否使用边界损失
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
        
        # 总损失初始化为BCE损失
        total_loss = bce_loss
        
        # 如果启用了边界损失，则计算并添加到总损失中
        boundary_loss_value = 0.0
        if use_boundary_loss:
            # 计算边界损失 (仅对主输出d0计算)
            boundary_loss = self.compute_boundary_loss(d0, labels_v)
            boundary_loss_value = boundary_loss.data.item()
            
            # 将边界损失添加到总损失中
            total_loss = bce_loss + boundary_weight * boundary_loss
        
        # 输出损失信息
        if use_boundary_loss:
            print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f, boundary: %3f"%(
                loss0.data.item(), loss1.data.item(), loss2.data.item(), 
                loss3.data.item(), loss4.data.item(), loss5.data.item(), 
                loss6.data.item(), boundary_loss_value))
        else:
            print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f"%(
                loss0.data.item(), loss1.data.item(), loss2.data.item(), 
                loss3.data.item(), loss4.data.item(), loss5.data.item(), 
                loss6.data.item()))
        
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
            
        # 只在主进程或单GPU模式下打印信息
        if not self.config.distributed or self.config.local_rank == 0:
            print("---")
            print("train images: ", len(tra_img_name_list))
            print("train labels: ", len(tra_lbl_name_list))
            print("---")
        
        return tra_img_name_list, tra_lbl_name_list
        
    def create_dataloader(self, img_name_list, lbl_name_list, batch_size, shuffle=True):
        """创建多通道分割的DataLoader，支持分布式训练"""
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
        
        # 为分布式训练创建数据采样器
        if self.config.distributed:
            sampler = DistributedSampler(
                dataset, 
                num_replicas=self.config.world_size,
                rank=self.config.local_rank,
                shuffle=shuffle
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            # 非分布式训练使用普通DataLoader
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=4,
                pin_memory=True
            )
            
        return dataloader

# --------- 3. 模型定义 ---------
class ModelSetup:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        
    def _get_device(self):
        """获取可用的计算设备"""
        if self.config.distributed:
            # 分布式训练时，每个进程使用指定的GPU
            return torch.device(f"cuda:{self.config.local_rank}")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device(f"cuda:{self.config.gpu}")
        else:
            return torch.device("cpu")
            
    def create_model(self):
        """创建并初始化模型，支持多通道输出和分布式训练"""
        print(f"使用设备: {self.device}")
        
        # 根据模型名称创建网络，注意这里输出通道是多通道的
        if self.config.model_name == 'u2net':
            net = U2NET(3, self.config.num_classes)
        elif self.config.model_name == 'u2netp':
            net = U2NETP(3, self.config.num_classes)
        else:
            raise ValueError(f"Unknown model name: {self.config.model_name}")
        
        # 如果启用了SyncBatchNorm，则转换模型中的所有BN层
        if self.config.sync_bn and self.config.distributed:
            print("使用SyncBatchNorm进行训练")
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            
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
        
        # 如果是分布式训练，使用DDP包装模型
        if self.config.distributed:
            print(f"在GPU {self.config.local_rank}上封装DDP模型")
            # 将模型封装为DDP模型 (注意: find_unused_parameters=True可以解决部分参数未被使用的问题)
            net = DDP(net, device_ids=[self.config.local_rank], output_device=self.config.local_rank, 
                       find_unused_parameters=True)
        elif torch.cuda.device_count() > 1 and not self.config.distributed:
            print(f"使用DataParallel，GPU数量: {torch.cuda.device_count()}")
            # 如果有多个GPU但不使用DDP，则使用DataParallel
            net = nn.DataParallel(net)
            
        return net
        
    def create_optimizer(self, model):
        """创建优化器"""
        # 找到模型的实际模块（如果是DDP或DataParallel，需要获取module）
        if isinstance(model, (nn.DataParallel, DDP)):
            model_without_wrapper = model.module
        else:
            model_without_wrapper = model
            
        # 创建优化器
        return optim.Adam(
            model_without_wrapper.parameters(), 
            lr=self.config.lr, 
            betas=self.config.betas, 
            eps=self.config.eps, 
            weight_decay=self.config.weight_decay
        )

# --------- 4. 训练流程 ---------
class Trainer:
    def __init__(self, config, model, optimizer, dataloader, loss_funcs, device, sampler=None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.loss_funcs = loss_funcs
        self.device = device
        self.best_accuracy = 0.0
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)  # 从配置获取边界损失权重，默认0.5
        self.use_boundary_loss = getattr(config, 'use_boundary_loss', True)  # 从配置获取是否使用边界损失，默认True
        self.sampler = sampler  # 分布式采样器，用于每个epoch设置
        
    def train(self):
        """训练模型"""
        # 只在主进程或非分布式训练中打印信息
        is_main_process = not self.config.distributed or self.config.local_rank == 0
        
        if is_main_process:
            print("---start training...")
            if self.use_boundary_loss:
                print(f"Using boundary loss with weight: {self.boundary_weight}")
            else:
                print("Boundary loss is disabled")
                
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
        
        for epoch in range(self.config.start_epoch, self.config.epoch_num):
            # 在分布式训练中，每个epoch需要设置不同的随机种子
            if self.config.distributed and self.sampler is not None:
                self.sampler.set_epoch(epoch)
                
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
                loss2, loss = self.loss_funcs.muti_bce_loss_fusion(
                    d0, d1, d2, d3, d4, d5, d6, labels_v, 
                    self.boundary_weight, self.use_boundary_loss
                )
                
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
                
                # 定期保存模型 (只在主进程或非分布式模式下)
                if ite_num % self.config.save_freq == 0 and is_main_process:
                    # 保存模型 (如果是DDP模型，保存module部分)
                    model_to_save = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model
                    
                    torch.save(
                        model_to_save.state_dict(), 
                        os.path.join(
                            self.config.model_dir, 
                            f"{self.config.model_name}_bce_itr_{ite_num}_train_{running_loss / max(ite_num4val, 1):.4f}.pth"
                        )
                    )
                    
                    # 保存边界可视化（在删除变量之前）
                    if self.use_boundary_loss:
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
                
                # 输出训练进度 (只在主进程或非分布式模式下)
                if is_main_process:
                    print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, accuracy: %3f\n" % (
                        epoch + 1, self.config.epoch_num, (i + 1) * self.config.batch_size_train, 
                        len(self.dataloader.dataset), ite_num, 
                        running_loss / max(ite_num4val, 1), running_tar_loss / max(ite_num4val, 1), batch_accuracy
                    ))
            
            # 计算每个epoch的平均统计数据
            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            avg_epoch_tar_loss = epoch_tar_loss / max(batch_count, 1)
            avg_epoch_accuracy = epoch_accuracy / max(batch_count, 1)
            
            # 在分布式训练中，收集所有进程的度量值并计算平均值
            if self.config.distributed:
                # 创建用于收集度量的张量
                loss_tensor = torch.tensor([avg_epoch_loss], device=self.device)
                accuracy_tensor = torch.tensor([avg_epoch_accuracy], device=self.device)
                
                # 使用all_reduce来平均所有进程的结果
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
                
                # 除以进程数得到平均值
                avg_epoch_loss = loss_tensor.item() / self.config.world_size
                avg_epoch_accuracy = accuracy_tensor.item() / self.config.world_size
            
            # 输出epoch统计信息 (只在主进程或非分布式模式下)
            if is_main_process:
                print(f"Epoch {epoch+1} Summary:")
                print(f"Average Loss: {avg_epoch_loss:.4f}")
                print(f"Average Accuracy: {avg_epoch_accuracy:.4f}")
            
            # 每5个epoch保存一次模型 (只在主进程或非分布式模式下)
            if (epoch + 1) % 5 == 0 and is_main_process:
                # 保存模型 (如果是DDP模型，保存module部分)
                model_to_save = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model
                
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}.pth"
                )
                torch.save(model_to_save.state_dict(), save_path)
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
                    if self.use_boundary_loss:
                        vis_path = os.path.join(
                            self.config.model_dir, 
                            f"boundaries_epoch_{epoch+1}.png"
                        )
                        self.loss_funcs.visualize_boundaries(d0, labels_v, save_path=vis_path)
                self.model.train()  # 恢复训练模式
            
            # 保存最佳模型 (只在主进程或非分布式模式下)
            if avg_epoch_accuracy > self.best_accuracy and is_main_process:
                self.best_accuracy = avg_epoch_accuracy
                
                # 保存模型 (如果是DDP模型，保存module部分)
                model_to_save = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model
                
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_best_accuracy_{self.best_accuracy:.4f}_epoch_{epoch+1}.pth"
                )
                torch.save(model_to_save.state_dict(), save_path)
                print(f"New best accuracy achieved! Model saved with accuracy: {self.best_accuracy:.4f}")

# --------- 主函数 ---------
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='U2Net多通道训练脚本')
    
    # 分布式训练相关参数
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练时的本地GPU排名')
    parser.add_argument('--sync-bn', action='store_true', help='是否使用同步批归一化')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID (非分布式训练时)')
    
    # 训练相关参数
    parser.add_argument('--batch-size', type=int, default=2, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100000, help='训练轮次')
    parser.add_argument('--no-boundary-loss', action='store_true', help='禁用边界损失')
    parser.add_argument('--boundary-weight', type=float, default=0.5, help='边界损失权重')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='u2net', choices=['u2net', 'u2netp'], help='模型类型')
    parser.add_argument('--num-classes', type=int, default=7, help='分割类别数（通道数）')
    parser.add_argument('--pretrained', type=str, default='', help='预训练模型路径')
    
    # 数据相关参数
    parser.add_argument('--data-dir', type=str, default='/data1/zhn/JZ/train_512/', help='数据目录')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化配置
    config = Config()
    
    # 从命令行参数更新配置
    config.model_name = args.model
    config.num_classes = args.num_classes
    config.batch_size_train = args.batch_size
    config.lr = args.lr
    config.epoch_num = args.epochs
    config.use_boundary_loss = not args.no_boundary_loss
    config.boundary_weight = args.boundary_weight
    config.pretrained_model_path = args.pretrained
    config.data_dir = args.data_dir
    
    # 分布式训练设置
    config.distributed = args.distributed
    config.local_rank = args.local_rank
    config.gpu = args.gpu
    config.sync_bn = args.sync_bn
    
    # 初始化分布式训练
    if config.distributed:
        # 初始化进程组
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            config.world_size = int(os.environ['WORLD_SIZE'])
            config.local_rank = int(os.environ['RANK'])
        else:
            config.world_size = 1
        
        # 设置设备
        torch.cuda.set_device(config.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        config.world_size = dist.get_world_size()
        
        print(f"初始化进程组: rank {config.local_rank}, world_size {config.world_size}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 确保模型保存目录存在 (仅在主进程)
    if not config.distributed or config.local_rank == 0:
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
    
    # 获取分布式采样器以传递给训练器
    sampler = None
    if config.distributed:
        sampler = dataloader.sampler
    
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
        device=model_setup.device,
        sampler=sampler
    )
    
    trainer.train()

if __name__ == '__main__':
    main() 