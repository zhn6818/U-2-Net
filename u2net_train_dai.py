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

from model import U2NET
from model import U2NETP

# --------- 配置参数 ---------
class Config:
    def __init__(self):
        # 基本配置
        self.model_name = 'u2net'  # 'u2netp'
        
        # 数据路径
        self.data_dir = "daizhuang/"
        self.tra_image_dir = os.path.join('images' + os.sep)
        self.tra_label_dir = os.path.join('masks' + os.sep)
        self.image_ext = '.jpg'
        self.label_ext = '.png'
        
        # 模型保存路径
        self.model_dir = os.path.join(os.getcwd(), 'daizhuang_saved_models_512', self.model_name + os.sep)
        
        # 预训练模型
        self.pretrained_model_path = ""
        self.start_epoch = 0  # 从哪个epoch开始训练
        
        # 训练参数
        self.epoch_num = 100000
        self.batch_size_train = 2
        self.batch_size_val = 1
        self.save_freq = 2000  # 保存模型的频率
        
        # 优化器参数
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-08
        self.weight_decay = 0

# --------- 1. 定义损失函数 ---------
class LossFunctions:
    def __init__(self):
        self.bce_loss = nn.BCELoss(size_average=True)
        
    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        """
        计算多级输出的BCE损失
        """
        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(
            loss0.data.item(), loss1.data.item(), loss2.data.item(), 
            loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

        return loss0, loss
    
    @staticmethod
    def calculate_accuracy(pred, target, threshold=0.5):
        """
        计算预测的准确率
        Args:
            pred: 预测的输出 (已经经过sigmoid)
            target: 真实标签
            threshold: 二值化阈值
        Returns:
            accuracy: 准确率
        """
        pred = (pred > threshold).float()
        target = (target > threshold).float()
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
        """创建DataLoader"""
        dataset = SalObjDataset(
            img_name_list=img_name_list,
            lbl_name_list=lbl_name_list,
            transform=transforms.Compose([
                RescaleT(512),
                RandomCrop(460),
                ToTensorLab(flag=0)
            ])
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
        """创建并初始化模型"""
        print(f"Using device: {self.device}")
        
        # 根据模型名称创建网络
        if self.config.model_name == 'u2net':
            net = U2NET(3, 1)
        elif self.config.model_name == 'u2netp':
            net = U2NETP(3, 1)
        else:
            raise ValueError(f"Unknown model name: {self.config.model_name}")
            
        # 加载预训练模型
        if os.path.exists(self.config.pretrained_model_path):
            print(f"Loading pretrained model from {self.config.pretrained_model_path}")
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
        
    def train(self):
        """训练模型"""
        print("---start training...")
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
                loss2, loss = self.loss_funcs.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
                
                # 计算准确率
                batch_accuracy = self.loss_funcs.calculate_accuracy(d0, labels_v)
                epoch_accuracy += batch_accuracy
                
                loss.backward()
                self.optimizer.step()
                
                # 记录统计数据
                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()
                epoch_loss += loss.data.item()
                epoch_tar_loss += loss2.data.item()
                batch_count += 1
                
                # 释放内存
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss
                
                # 输出训练进度
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, accuracy: %3f " % (
                    epoch + 1, self.config.epoch_num, (i + 1) * self.config.batch_size_train, 
                    len(self.dataloader.dataset), ite_num, 
                    running_loss / ite_num4val, running_tar_loss / ite_num4val, batch_accuracy
                ))
                
                # 定期保存模型
                if ite_num % self.config.save_freq == 0:
                    torch.save(
                        self.model.state_dict(), 
                        os.path.join(
                            self.config.model_dir, 
                            f"{self.config.model_name}_bce_itr_{ite_num}_train_{running_loss / ite_num4val:.4f}_tar_{running_tar_loss / ite_num4val:.4f}.pth"
                        )
                    )
                    running_loss = 0.0
                    running_tar_loss = 0.0
                    self.model.train()  # 继续训练
                    ite_num4val = 0
            
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
                    f"{self.config.model_name}_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}_acc_{avg_epoch_accuracy:.4f}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch+1} with loss {avg_epoch_loss:.4f} and accuracy {avg_epoch_accuracy:.4f}")
            
            # 保存最佳模型
            if avg_epoch_accuracy > self.best_accuracy:
                self.best_accuracy = avg_epoch_accuracy
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_best_acc_{self.best_accuracy:.4f}_epoch_{epoch+1}.pth"
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

