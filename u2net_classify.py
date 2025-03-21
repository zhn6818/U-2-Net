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
from skimage import io, transform, color

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# --------- 注意力模块和残差模块 ---------
class AttentionModule(nn.Module):
    """带有通道和空间注意力机制的模块"""
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 应用通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 应用空间注意力
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        return x

class ResidualBlock(nn.Module):
    """带有残差连接的卷积块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

# --------- 复杂分类头 ---------
class ClassifierHead(nn.Module):
    """复杂的分类头，使用多尺度特征和注意力机制"""
    def __init__(self, n_classes=3):
        super(ClassifierHead, self).__init__()
        
        # 输入图像处理分支
        self.img_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 分割特征处理
        self.d0_conv = nn.Conv2d(1, 64, kernel_size=1)
        self.d1_conv = nn.Conv2d(1, 64, kernel_size=1) 
        self.d2_conv = nn.Conv2d(1, 64, kernel_size=1)
        self.d3_conv = nn.Conv2d(1, 64, kernel_size=1)
        
        # 计算融合特征的通道数（原图特征 + 4个分割特征 = 5 * 64 = 320）
        fusion_channels = 64 * 5
        
        # 注意力模块 - 通道数必须与融合后的特征通道数一致
        self.attention = AttentionModule(fusion_channels)
        
        # 特征融合和处理
        self.fusion_conv = nn.Conv2d(fusion_channels, 128, kernel_size=1)  # 5个输入源：原图+4个分割特征
        self.residual_block = ResidualBlock(128)
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        
        # 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x, d0, d1, d2, d3):
        # 从原始图像获取特征
        img_features = self.img_branch(x)
        
        # 处理分割特征
        d0_feat = F.interpolate(self.d0_conv(d0), size=img_features.shape[2:], mode='bilinear')
        d1_feat = F.interpolate(self.d1_conv(d1), size=img_features.shape[2:], mode='bilinear')
        d2_feat = F.interpolate(self.d2_conv(d2), size=img_features.shape[2:], mode='bilinear')
        d3_feat = F.interpolate(self.d3_conv(d3), size=img_features.shape[2:], mode='bilinear')
        
        # 融合特征
        fused_features = torch.cat([img_features, d0_feat, d1_feat, d2_feat, d3_feat], dim=1)
        
        # 应用注意力机制 - 在降维前先应用注意力
        att_features = self.attention(fused_features)
        
        # 降维特征
        reduced_features = self.fusion_conv(att_features)
        
        # 残差块处理
        res_features = self.residual_block(reduced_features)
        
        # 全局池化
        pooled = self.global_pool(res_features).view(res_features.size(0), -1)
        pooled = self.dropout(pooled)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits

# --------- 带分类头的U2NET模型 ---------
class U2NetWithClassifier(nn.Module):
    def __init__(self, n_classes=3, pretrained_path=None, freeze_backbone=False, eval_backbone=True):
        """
        初始化分类器扩展的U2NET模型
        
        Args:
            n_classes: 分类类别数
            pretrained_path: 预训练模型路径
            freeze_backbone: 是否冻结U2NET主干网络
            eval_backbone: 是否始终将主干网络保持在评估模式
        """
        super(U2NetWithClassifier, self).__init__()
        
        # 基础U2NET模型
        self.u2net = U2NET(3, 1)  # 输入通道3，输出通道1
        
        # 分类头 - 将使用多个U2NET中间层特征
        self.classifier = ClassifierHead(n_classes=n_classes)
        
        # 是否保持主干网络在评估模式
        self.eval_backbone = eval_backbone
        
        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
            
        # 如果需要，冻结主干网络
        if freeze_backbone:
            self._freeze_backbone()
            print(f"U2NET主干网络已冻结，只有分类头参数可训练")
    
    def _freeze_backbone(self):
        """冻结U2NET主干网络参数和BatchNorm统计值"""
        # 冻结所有参数
        for name, param in self.named_parameters():
            if 'u2net' in name:
                param.requires_grad = False
        
        # 冻结BatchNorm层统计值
        for name, module in self.u2net.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()  # 设置为评估模式，不更新running统计量
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    
    def train(self, mode=True):
        """重写train方法，确保在训练模式下主干网络仍然可以保持评估模式"""
        if mode:
            # 设置整个模型为训练模式
            super(U2NetWithClassifier, self).train(mode)
            
            # 如果设置了eval_backbone，则将主干网络设为评估模式
            if self.eval_backbone:
                self.u2net.eval()
                
                # # 确保BatchNorm层仍处于评估模式
                # for m in self.u2net.modules():
                #     if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                #         m.eval()
        else:
            # 设置整个模型为评估模式
            super(U2NetWithClassifier, self).train(mode)
        
        return self
    
    def _load_pretrained_weights(self, pretrained_path):
        """加载预训练权重并处理兼容性问题"""
        # 确定当前设备
        device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
        
        print(f"尝试加载预训练模型: {pretrained_path} 到设备: {device}")
        
        try:
            # 先尝试直接加载
            state_dict = torch.load(pretrained_path, map_location=device)
            
            # 判断是否是带分类器的完整模型
            if any("classifier" in k for k in state_dict.keys()):
                print("加载的是带分类器的完整模型")
                self.load_state_dict(state_dict)
            else:
                # 只加载U2NET部分的权重
                print("加载的是U2NET分割模型权重，只更新主干网络部分")
                
                # 检查模型权重键名格式
                keys = list(state_dict.keys())
                print(f"模型权重包含 {len(keys)} 个参数")
                print(f"示例键名: {keys[:3]}")
                
                # 检查当前模型的u2net部分参数名称格式
                u2net_keys = [name for name, _ in self.u2net.named_parameters()]
                print(f"当前模型u2net部分包含 {len(u2net_keys)} 个参数")
                print(f"示例u2net参数名: {u2net_keys[:3]}")
                
                # 检查参数名前缀情况
                has_u2net_prefix = any("u2net" in k for k in state_dict.keys())
                print(f"预训练权重是否有u2net前缀: {has_u2net_prefix}")
                
                # 尝试直接加载到u2net子模块
                # 注意：这里改变加载策略，先尝试直接加载到u2net子模块
                try:
                    print("尝试直接加载权重到u2net子模块")
                    
                    # 临时将u2net设置为eval模式，确保加载后的统计值被正确应用
                    self.u2net.eval()
                    
                    # 直接加载到u2net
                    load_result = self.u2net.load_state_dict(state_dict, strict=False)
                    
                    # 打印哪些参数没有被加载
                    if load_result.missing_keys:
                        print(f"警告: 以下参数未在预训练权重中找到:")
                        print(f"缺失参数数量: {len(load_result.missing_keys)}")
                        if len(load_result.missing_keys) < 10:
                            print(f"缺失参数: {load_result.missing_keys}")
                            
                    if load_result.unexpected_keys:
                        print(f"信息: 以下预训练权重参数未在模型中使用:")
                        print(f"未使用参数数量: {len(load_result.unexpected_keys)}")
                        if len(load_result.unexpected_keys) < 10:
                            print(f"未使用参数: {load_result.unexpected_keys}")
                    
                    # 如果缺失或多余的参数不多，则视为成功
                    if len(load_result.missing_keys) < len(u2net_keys) * 0.2:  # 少于20%的参数缺失
                        load_success = True
                        print("成功直接加载到u2net子模块")
                    else:
                        load_success = False
                        print(f"直接加载到u2net子模块部分成功，但缺失较多参数({len(load_result.missing_keys)}个)")
                except Exception as e:
                    print(f"直接加载到u2net子模块失败: {e}")
                    load_success = False
                
                # 如果直接加载失败且权重没有u2net前缀，尝试添加前缀
                if not load_success and not has_u2net_prefix:
                    try:
                        print("尝试添加'u2net.'前缀后加载")
                        u2net_state_dict = {"u2net." + k: v for k, v in state_dict.items()}
                        
                        # 加载整个模型
                        load_result = self.load_state_dict(u2net_state_dict, strict=False)
                        
                        # 判断加载结果
                        if len(load_result.missing_keys) < len(list(self.named_parameters())) * 0.3:
                            print("成功通过添加前缀方式加载")
                            load_success = True
                        else:
                            print(f"添加前缀加载部分成功，但缺失较多参数")
                            load_success = False
                    except Exception as e:
                        print(f"添加前缀加载失败: {e}")
                        load_success = False
                        
                        # 最后尝试：检查权重格式，进行更智能的名称映射
                        try:
                            print("尝试更智能的参数名称映射...")
                            # 创建参数映射，将预训练权重的键映射到模型参数名
                            param_mapping = {}
                            for pretrained_key in state_dict.keys():
                                # 移除可能的前缀
                                clean_key = pretrained_key.split('.')[-2] + '.' + pretrained_key.split('.')[-1] \
                                    if len(pretrained_key.split('.')) > 2 else pretrained_key
                                
                                # 查找匹配的模型参数
                                for model_key in u2net_keys:
                                    if clean_key in model_key:
                                        param_mapping[pretrained_key] = model_key
                                        break
                            
                            # 创建新的state_dict
                            if param_mapping:
                                print(f"找到 {len(param_mapping)} 个参数映射")
                                mapped_state_dict = {model_key: state_dict[pretrained_key] 
                                                   for pretrained_key, model_key in param_mapping.items()}
                                
                                # 加载映射后的权重
                                missing, unexpected = self.u2net.load_state_dict(mapped_state_dict, strict=False)
                                print(f"智能映射加载完成。未映射参数: {len(missing)}, 多余参数: {len(unexpected)}")
                                load_success = len(missing) < len(u2net_keys) * 0.3
                            else:
                                print("未找到参数映射，使用随机初始化")
                                load_success = False
                        except Exception as e:
                            print(f"智能映射失败: {e}")
                            load_success = False
                
                # 加载完成后，验证参数是否正确初始化
                if load_success:
                    # 检查权重参数的标准差
                    abnormal_params = []
                    for name, param in self.u2net.named_parameters():
                        if 'weight' in name and not 'norm' in name:  # 排除norm层的weight
                            std = param.std().item()
                            if std < 1e-6 or std > 1.0:
                                abnormal_params.append((name, std))
                    
                    if abnormal_params:
                        print(f"警告: 检测到 {len(abnormal_params)} 个参数可能未正确初始化")
                        if len(abnormal_params) < 5:
                            for name, std in abnormal_params:
                                print(f"参数 {name} 标准差为 {std:.6f}")
                    else:
                        print("所有主干网络参数标准差正常，权重加载成功")
                
            print(f"成功加载权重到设备: {device}")
        except Exception as e:
            print(f"直接加载失败: {e}")
            print("尝试通过CPU进行加载...")
            
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                
                # 判断是否是带分类器的完整模型
                if any("classifier" in k for k in state_dict.keys()):
                    print("加载的是带分类器的完整模型")
                    self.load_state_dict(state_dict)
                else:
                    # 只加载U2NET部分的权重
                    print("加载的是U2NET分割模型权重，只更新主干网络部分")
                    
                    # 尝试直接加载到u2net子模块
                    try:
                        # 先将u2net设为eval模式
                        self.u2net.eval()
                        load_result = self.u2net.load_state_dict(state_dict, strict=False)
                        if len(load_result.missing_keys) < 20:  # 只有少量缺失
                            print("成功通过直接加载到u2net子模块")
                        else:
                            print(f"直接加载到u2net子模块部分成功，但有 {len(load_result.missing_keys)} 个参数缺失")
                    except Exception as e:
                        print(f"子模块加载失败: {e}")
                        # 添加前缀再尝试
                        try:
                            u2net_state_dict = {"u2net." + k: v for k, v in state_dict.items()}
                            load_result = self.load_state_dict(u2net_state_dict, strict=False)
                            if len(load_result.missing_keys) < 100:
                                print("成功通过添加前缀方式加载")
                            else:
                                print(f"添加前缀加载部分成功，但有 {len(load_result.missing_keys)} 个参数缺失")
                        except Exception as e:
                            print(f"添加前缀加载失败: {e}")
                
                print(f"成功通过CPU加载权重并迁移到设备: {device}")
            except Exception as e:
                print(f"加载权重失败: {e}")
                print("将使用随机初始化的权重")
                
        # 确保所有BatchNorm层已固定，如果eval_backbone为True
        if self.eval_backbone:
            self.u2net.eval()
            for m in self.u2net.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
    
    def forward(self, x):
        # 获取U2NET的所有输出
        d0, d1, d2, d3, d4, d5, d6 = self.u2net(x)
        
        # 将特定层的特征传递给分类头
        # 这里我们选择 d0（最终输出）和 d1-d3 作为多尺度特征
        class_logits = self.classifier(x, d0, d1, d2, d3)
        
        # 返回分割结果和分类结果
        return (d0, d1, d2, d3, d4, d5, d6), class_logits

# --------- 扩展数据集 ---------
class SegClassDataset(SalObjDataset):
    """扩展原有SalObjDataset，增加分类标签"""
    
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        super(SegClassDataset, self).__init__(img_name_list, lbl_name_list, transform)
        
    def __getitem__(self, idx):
        # 获取原有的图像和分割标签
        sample = super(SegClassDataset, self).__getitem__(idx)
        
        # 从原始mask文件直接读取分类标签，而不是处理过的sample['label']
        label_path = self.label_name_list[idx]
        
        # 读取原始mask图像
        try:
            if os.path.exists(label_path):
                # 读取原始标签图像
                original_mask = io.imread(label_path)
                
                # 分析mask中的像素值，确定类别
                unique_values = np.unique(original_mask)
                
                # 移除背景值（通常为0）
                unique_values = unique_values[unique_values > 0]
                
                # 如果有多个值，选择值最大的那个
                if len(unique_values) > 0:
                    class_label = int(np.max(unique_values))
                    # 确保类别值在1-3范围内
                    class_label = max(1, min(class_label, 3))
                else:
                    # 如果没有前景，设为默认类别（例如1）
                    class_label = 1
            else:
                # 如果标签文件不存在，设置默认类别
                print(f"警告: 标签文件不存在 {label_path}")
                class_label = 1
        except Exception as e:
            print(f"读取标签文件出错 {label_path}: {e}")
            # 发生错误时使用默认类别
            class_label = 1
            
        # 转换为从0开始的索引 (PyTorch分类常用方式)
        class_idx = class_label - 1
        
        # 添加分类标签到样本中
        sample['class_label'] = class_idx
        
        return sample

# --------- 组合损失函数 ---------
class CombinedLoss:
    """组合分割和分类损失"""
    
    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        self.bce_loss = nn.BCELoss(size_average=True)
        self.cls_loss = nn.CrossEntropyLoss()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        
    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        """计算多级输出的BCE损失 (与原有相同)"""
        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        
        return loss0, loss
        
    def __call__(self, seg_outputs, cls_outputs, seg_labels, cls_labels):
        # 计算分割损失
        seg_loss0, seg_loss_sum = self.muti_bce_loss_fusion(*seg_outputs, seg_labels)
        
        # 计算分类损失
        cls_loss = self.cls_loss(cls_outputs, cls_labels)
        
        # 组合两种损失
        total_loss = self.seg_weight * seg_loss_sum + self.cls_weight * cls_loss
        
        return seg_loss0, seg_loss_sum, cls_loss, total_loss

# --------- 配置参数扩展 ---------
class ConfigWithClassifier:
    """用于分类模型的配置参数"""
    
    def __init__(self):
        # 基本配置
        self.model_name = 'u2net_cls'  # 模型名称
        
        # 数据路径
        self.data_dir = "daizhuang/"
        self.tra_image_dir = os.path.join('images' + os.sep)
        self.tra_label_dir = os.path.join('masks' + os.sep)
        self.image_ext = '.jpg'
        self.label_ext = '.png'
        
        # 模型保存路径
        self.model_dir = os.path.join(os.getcwd(), 'daizhuang_saved_models_classify', self.model_name + os.sep)
        
        # 预训练模型
        self.pretrained_model_path = "daizhuang_saved_models_512/u2net/u2net_best_acc_0.9340_epoch_91.pth"
        self.start_epoch = 0  # 从哪个epoch开始训练
        
        # 训练参数
        self.epoch_num = 100
        self.batch_size_train = 2
        self.batch_size_val = 1
        self.save_freq = 2000  # 保存模型的频率
        
        # 优化器参数
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-08
        self.weight_decay = 0
        
        # 分类相关配置
        self.n_classes = 3  # 类别数量
        
        # 分阶段训练配置
        self.freeze_backbone = True  # 是否冻结U2NET主干网络
        self.phase1_epochs = 10      # 第一阶段训练epochs数
        
        # 损失权重
        self.seg_loss_weight = 1.0
        self.cls_loss_weight = 1.0
        
        # 微调学习率
        self.finetune_lr = self.lr * 0.1
        
        # 设备适配策略 - 确保CUDA训练的模型可以在其他设备上工作
        self.force_cpu_load = False  # 在特殊情况下可以强制通过CPU加载

# --------- 训练器扩展 ---------
class TrainerWithClassifier:
    """增加分类功能的训练器"""
    
    def __init__(self, config, model, optimizer, dataloader, loss_funcs, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.loss_funcs = loss_funcs
        self.device = device
        self.best_accuracy = 0.0
        self.best_cls_accuracy = 0.0
    
    def train(self, freeze_u2net=True, epochs_phase1=10):
        """训练模型
        Args:
            freeze_u2net: 是否冻结U2NET部分的参数
            epochs_phase1: 分阶段训练时第一阶段的epochs数
        """
        print("---start training...")
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        running_cls_loss = 0.0
        ite_num4val = 0
        
        total_epochs = self.config.epoch_num
        
        # 冻结U2NET部分 - 这个逻辑已经在main函数中处理，这里主要管理解冻
        if freeze_u2net:
            # 验证U2NET确实被冻结
            u2net_frozen = all(not param.requires_grad for name, param in self.model.named_parameters() if 'u2net' in name)
            if not u2net_frozen:
                print("警告：U2NET主干应该被冻结但某些参数仍可训练")
                # 重新冻结
                for name, param in self.model.named_parameters():
                    if 'u2net' in name:
                        param.requires_grad = False
            
            print("第一阶段训练：仅训练分类头，U2NET主干保持冻结")
        
        for epoch in range(self.config.start_epoch, total_epochs):
            # 如果到达阶段2，解冻U2NET
            if freeze_u2net and epoch >= epochs_phase1:
                # 解冻逻辑：明确地设置所有参数为可训练状态
                print(f"\n=== 进入训练第二阶段 (Epoch {epoch+1}/{total_epochs}) ===")
                print("解冻U2NET主干网络...")
                
                # 统计解冻的参数数量
                unfrozen_count = 0
                for name, param in self.model.named_parameters():
                    if 'u2net' in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                
                print(f"已解冻 {unfrozen_count} 个U2NET参数")
                freeze_u2net = False  # 只执行一次解冻
                
                # 重新设置优化器，可能需要更小的学习率
                self.optimizer = optim.Adam(
                    self.model.parameters(),  # 现在包含所有模型参数
                    lr=self.config.finetune_lr,  # 降低学习率进行微调
                    betas=self.config.betas,
                    eps=self.config.eps,
                    weight_decay=self.config.weight_decay
                )
                print(f"重新配置优化器，使用微调学习率: {self.config.finetune_lr}")
                print(f"=== 第二阶段：微调整个网络 ===")
            
            self.model.train()
            epoch_loss = 0.0
            epoch_tar_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_accuracy = 0.0
            epoch_cls_accuracy = 0.0
            batch_count = 0
            
            for i, data in enumerate(self.dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1
                
                # 准备数据
                inputs, seg_labels = data['image'], data['label']
                cls_labels = data['class_label']  # 获取分类标签
                
                inputs = inputs.type(torch.FloatTensor)
                seg_labels = seg_labels.type(torch.FloatTensor)
                cls_labels = cls_labels.type(torch.LongTensor)  # 分类标签通常是长整型
                
                inputs_v = inputs.to(self.device)
                seg_labels_v = seg_labels.to(self.device)
                cls_labels_v = cls_labels.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                seg_outputs, cls_outputs = self.model(inputs_v)
                
                # 计算损失
                seg_loss0, seg_loss_sum, cls_loss, total_loss = self.loss_funcs(
                    seg_outputs, cls_outputs, seg_labels_v, cls_labels_v
                )
                
                # 计算分割准确率
                batch_seg_accuracy = self.calculate_seg_accuracy(seg_outputs[0], seg_labels_v)
                # 计算分类准确率
                batch_cls_accuracy = self.calculate_cls_accuracy(cls_outputs, cls_labels_v)
                
                epoch_accuracy += batch_seg_accuracy
                epoch_cls_accuracy += batch_cls_accuracy
                
                # 反向传播和优化
                total_loss.backward()
                self.optimizer.step()
                
                # 记录统计数据
                running_loss += total_loss.item()
                running_tar_loss += seg_loss0.item()
                running_cls_loss += cls_loss.item()
                
                epoch_loss += total_loss.item()
                epoch_tar_loss += seg_loss0.item()
                epoch_cls_loss += cls_loss.item()
                batch_count += 1
                
                # 释放内存
                del seg_outputs, cls_outputs, seg_loss0, seg_loss_sum, cls_loss, total_loss
                
                # 输出训练进度
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] total loss: %3f, seg: %3f, cls: %3f, seg_acc: %3f, cls_acc: %3f" % (
                    epoch + 1, total_epochs, (i + 1) * self.config.batch_size_train, 
                    len(self.dataloader.dataset), ite_num, 
                    running_loss / ite_num4val, running_tar_loss / ite_num4val, running_cls_loss / ite_num4val,
                    batch_seg_accuracy, batch_cls_accuracy
                ))
                
                # 定期保存模型
                if ite_num % self.config.save_freq == 0:
                    save_path = os.path.join(
                        self.config.model_dir, 
                        f"{self.config.model_name}_itr_{ite_num}_loss_{running_loss / ite_num4val:.4f}.pth"
                    )
                    torch.save(self.model.state_dict(), save_path)
                    running_loss = 0.0
                    running_tar_loss = 0.0
                    running_cls_loss = 0.0
                    self.model.train()  # 继续训练
                    ite_num4val = 0
            
            # 计算每个epoch的平均统计数据
            avg_epoch_loss = epoch_loss / batch_count
            avg_epoch_tar_loss = epoch_tar_loss / batch_count
            avg_epoch_cls_loss = epoch_cls_loss / batch_count
            avg_epoch_accuracy = epoch_accuracy / batch_count
            avg_epoch_cls_accuracy = epoch_cls_accuracy / batch_count
            
            # 输出epoch统计信息
            print(f"Epoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_epoch_loss:.4f}, Seg Loss: {avg_epoch_tar_loss:.4f}, Cls Loss: {avg_epoch_cls_loss:.4f}")
            print(f"Average Seg Accuracy: {avg_epoch_accuracy:.4f}, Cls Accuracy: {avg_epoch_cls_accuracy:.4f}")
            
            # 每5个epoch保存一次模型
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}_segacc_{avg_epoch_accuracy:.4f}_clsacc_{avg_epoch_cls_accuracy:.4f}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch+1}")
            
            # 保存最佳分割模型
            if avg_epoch_accuracy > self.best_accuracy:
                self.best_accuracy = avg_epoch_accuracy
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_best_segacc_{self.best_accuracy:.4f}_epoch_{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"New best segmentation accuracy: {self.best_accuracy:.4f}")
                
            # 保存最佳分类模型
            if avg_epoch_cls_accuracy > self.best_cls_accuracy:
                self.best_cls_accuracy = avg_epoch_cls_accuracy
                save_path = os.path.join(
                    self.config.model_dir, 
                    f"{self.config.model_name}_best_clsacc_{self.best_cls_accuracy:.4f}_epoch_{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"New best classification accuracy: {self.best_cls_accuracy:.4f}")
    
    @staticmethod
    def calculate_seg_accuracy(pred, target, threshold=0.5):
        """计算分割预测的准确率
        Args:
            pred: 模型预测的分割掩码
            target: 目标分割掩码
            threshold: 二值化阈值
        Returns:
            准确率值，介于0-1之间
        """
        pred = (pred > threshold).float()
        target = (target > threshold).float()
        correct = (pred == target).float().sum()
        total = target.numel()
        return (correct / total).item()
    
    @staticmethod
    def calculate_cls_accuracy(pred, target):
        """计算分类预测的准确率"""
        _, predicted = torch.max(pred, 1)
        correct = (predicted == target).sum().item()
        total = target.size(0)
        return correct / total

# --------- 数据准备 ---------
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
        dataset = SegClassDataset(
            img_name_list=img_name_list,
            lbl_name_list=lbl_name_list,
            transform=transforms.Compose([
                RescaleT(512),
                RandomCrop(460),
                ToTensorLab(flag=0)
            ])
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        return dataloader, dataset

# --------- 主函数 ---------
def main():
    """主程序入口"""
    print("初始化配置...")
    config = ConfigWithClassifier()
    
    # 确保模型保存目录存在
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    
    # 准备数据集
    print("准备数据集...")
    data_prep = DatasetPreparation(config)
    tra_img_name_list, tra_lbl_name_list = data_prep.get_data_paths()
    dataloader, dataset = data_prep.create_dataloader(
        tra_img_name_list, 
        tra_lbl_name_list, 
        config.batch_size_train
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else 
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型路径并显示明确的设备映射信息
    if os.path.exists(config.pretrained_model_path):
        print(f"找到预训练模型: {config.pretrained_model_path}")
        print(f"该模型将被映射到当前设备: {device}")
    else:
        print(f"警告: 预训练模型 {config.pretrained_model_path} 不存在!")
        print("将创建一个新的未初始化模型。")
        return  # 如果没有预训练模型，终止程序
    
    # 首先创建和加载原始U2NET模型进行对比基准测试
    print("\n======== 第1步：加载原始U2NET模型用作基准参考 ========")
    original_u2net = U2NET(3, 1)
    original_u2net.to(device)
    
    try:
        # 加载预训练权重到原始模型
        original_u2net.load_state_dict(torch.load(config.pretrained_model_path, map_location=device))
        original_u2net.eval()  # 设置为评估模式
        print("原始U2NET模型加载成功，将作为基准参考")
    except Exception as e:
        print(f"加载原始U2NET模型失败: {e}")
        print("将创建一个新的模型继续")
    
    # 创建带分类头的扩展模型
    print("\n======== 第2步：创建带分类头的扩展U2NET模型 ========")
    model = U2NetWithClassifier(
        n_classes=config.n_classes, 
        pretrained_path=config.pretrained_model_path,
        freeze_backbone=config.freeze_backbone,
        eval_backbone=True  # 确保主干网络保持评估模式
    )
    model.to(device)
    
    # 严格验证两个模型的主干网络输出是否一致
    print("\n======== 第3步：严格验证冻结的主干网络输出与原始模型一致性 ========")
    model.eval()  # 先设置为评估模式进行测试
    with torch.no_grad():
        try:
            # 获取一个批次的数据
            sample_batch = next(iter(dataloader))
            inputs = sample_batch['image'].type(torch.FloatTensor).to(device)
            
            # 1. 通过原始U2NET模型前向传播
            original_outputs = original_u2net(inputs)
            orig_d0 = original_outputs[0]  # 获取最终分割结果
            
            # 2. 通过带分类头的U2NET模型前向传播
            u2net_with_cls_outputs, _ = model(inputs)
            cls_d0 = u2net_with_cls_outputs[0]  # 获取最终分割结果
            
            # 3. 计算两个输出的差异
            # 计算平均绝对误差
            mae = torch.mean(torch.abs(orig_d0 - cls_d0)).item()
            
            # 计算元素级别的最大差异
            max_diff = torch.max(torch.abs(orig_d0 - cls_d0)).item()
            
            print(f"两个模型输出的平均绝对误差 (MAE): {mae:.8f}")
            print(f"两个模型输出的最大绝对误差: {max_diff:.8f}")
            
            # 判断输出是否基本一致 (允许浮点误差)
            is_consistent = mae < 1e-5 and max_diff < 1e-4
            
            if is_consistent:
                print("验证成功: 冻结的主干网络输出与原始模型基本一致!")
            else:
                print("警告: 冻结的主干网络输出与原始模型存在差异!")
                print("这可能是由于BatchNorm层统计值的差异或其他因素导致")
                print("如果差异很小，通常不会对训练过程产生明显影响")
            
            # 检查一下两个模型中一些关键层的权重是否完全相同
            print("\n检查两个模型的一些关键层权重:")
            
            # 获取第一个卷积层权重
            try:
                # 找到原始模型的第一个卷积层
                for name, module in original_u2net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        orig_first_conv = module
                        orig_name = name
                        break
                
                # 找到扩展模型的第一个卷积层
                for name, module in model.u2net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        cls_first_conv = module
                        cls_name = name
                        break
                
                # 比较两个权重
                weight_diff = torch.mean(torch.abs(orig_first_conv.weight - cls_first_conv.weight)).item()
                print(f"第一个卷积层权重差异 ('{orig_name}' vs 'u2net.{cls_name}'): {weight_diff:.8f}")
                
                if weight_diff < 1e-8:
                    print("权重完全一致!")
                else:
                    print("警告: 权重存在差异!")
            except Exception as e:
                print(f"比较权重时出错: {e}")
            
            # 计算初始分割准确率
            seg_labels = sample_batch['label'].type(torch.FloatTensor).to(device)
            
            # 分别计算原始模型和扩展模型的准确率
            orig_accuracy = TrainerWithClassifier.calculate_seg_accuracy(orig_d0, seg_labels)
            cls_accuracy = TrainerWithClassifier.calculate_seg_accuracy(cls_d0, seg_labels)
            
            print(f"\n原始U2NET模型分割准确率: {orig_accuracy:.4f}")
            print(f"带分类头的U2NET模型分割准确率: {cls_accuracy:.4f}")
            print(f"准确率差异: {abs(orig_accuracy - cls_accuracy):.4f}")
            
            if abs(orig_accuracy - cls_accuracy) < 1e-4:
                print("准确率几乎完全一致!")
            else:
                print(f"准确率存在差异，但幅度{'可接受' if abs(orig_accuracy - cls_accuracy) < 0.05 else '较大'}")
                
        except Exception as e:
            print(f"验证模型一致性时出错: {e}")
    
    # 设置回训练模式准备训练
    print("\n======== 第4步：准备训练模式 ========")
    model.train()  # 返回训练模式，但主干网络仍处于评估模式
    
    # 验证冻结情况
    u2net_params_count = 0
    frozen_params_count = 0
    
    for name, param in model.named_parameters():
        if 'u2net' in name:
            u2net_params_count += 1
            if not param.requires_grad:
                frozen_params_count += 1
    
    print(f"U2NET主干网络共有 {u2net_params_count} 个参数，其中 {frozen_params_count} 个已冻结")
    
    if frozen_params_count == u2net_params_count:
        print("U2NET主干网络已完全冻结!")
    else:
        print(f"警告: 只有 {frozen_params_count}/{u2net_params_count} 个U2NET参数被冻结!")
        
        # 如果有未冻结的参数，打印它们的名称
        unfrozen_params = [name for name, param in model.named_parameters() 
                         if 'u2net' in name and param.requires_grad]
        
        if len(unfrozen_params) > 0:
            print(f"未冻结的U2NET参数示例: {unfrozen_params[:min(5, len(unfrozen_params))]}")
    
    # 验证BatchNorm层模式
    bn_layers_count = 0
    bn_eval_count = 0
    
    for name, module in model.u2net.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers_count += 1
            if not module.training:  # 如果模块处于eval模式
                bn_eval_count += 1
    
    print(f"U2NET主干网络共有 {bn_layers_count} 个BatchNorm层，其中 {bn_eval_count} 个处于评估模式")
    
    if bn_eval_count == bn_layers_count:
        print("所有BatchNorm层都已固定为评估模式!")
    else:
        print(f"警告: 只有 {bn_eval_count}/{bn_layers_count} 个BatchNorm层处于评估模式!")
    
    # 设置优化器 - 如果冻结backbone，只优化分类头
    print("\n======== 第5步：配置优化器和训练参数 ========")
    if config.freeze_backbone:
        # 验证哪些参数会被优化
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"可训练的参数: {len(trainable_params)}")
        if len(trainable_params) > 0:
            print(f"示例可训练参数: {trainable_params[:min(3, len(trainable_params))]}...")
        
        # 只优化分类头的参数 (requires_grad=True的参数)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        print("仅优化分类头部分参数，U2NET主干已冻结")
    else:
        # 确保所有参数都是可训练的
        for param in model.parameters():
            param.requires_grad = True
            
        # 优化所有参数
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        print("优化所有模型参数")
    
    # 设置损失函数
    loss_funcs = CombinedLoss(
        seg_weight=config.seg_loss_weight,
        cls_weight=config.cls_loss_weight
    )
    
    # 训练模型
    trainer = TrainerWithClassifier(
        config=config,
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        loss_funcs=loss_funcs,
        device=device
    )
    
    print("\n======== 第6步：开始训练过程 ========")
    print("=" * 50)
    print("训练配置:")
    print(f"- 分阶段训练: {'是' if config.freeze_backbone else '否'}")
    print(f"- 第一阶段epochs: {config.phase1_epochs}")
    print(f"- 总epochs: {config.epoch_num}")
    print(f"- 批次大小: {config.batch_size_train}")
    print(f"- 学习率: {config.lr}（微调: {config.finetune_lr}）")
    print(f"- 主干网络评估模式: {'是' if model.eval_backbone else '否'}")
    print("=" * 50)
    
    trainer.train(
        freeze_u2net=config.freeze_backbone,
        epochs_phase1=config.phase1_epochs
    )

if __name__ == '__main__':
    main() 