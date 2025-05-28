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
from data_loader import ColorJitter
from data_loader import RandomMaxFilter
from data_loader import RandomScratch
from model import U2NET
from model import U2NETP
from model import U2NET_GRAIN
from model import U2NETP_GRAIN

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5=None, d6=None, labels_v=None):
    """
    多尺度BCE损失融合函数
    支持原始U2NET(7个输出)和新的U2NET_GRAIN(5个输出)
    """
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    
    if d5 is not None and d6 is not None:
        # 原始U2NET的7个输出
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f"%(
            loss0.data.item(), loss1.data.item(), loss2.data.item(), 
            loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))
    else:
        # 新的U2NET_GRAIN的5个输出
        loss = loss0 + loss1 + loss2 + loss3 + loss4
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f"%(
            loss0.data.item(), loss1.data.item(), loss2.data.item(), 
            loss3.data.item(), loss4.data.item()))

    return loss0, loss

# 添加计算准确率的函数
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

# ------- 2. set the directory of training dataset --------

model_name = 'u2net_grain' # 可选: 'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
print(f"Model directory: {model_dir}")

# 添加预训练模型路径
pretrained_model_path = ""
# 从预训练模型文件名中提取起始epoch
start_epoch = 0  # 从文件名中提取的epoch数
print(f"Pretrained model: {pretrained_model_path}")
print(f"Starting from epoch: {start_epoch}")

epoch_num = 100000
batch_size_train = 2
batch_size_val = 1
train_num = 0
val_num = 0

# 初始化train.txt路径
train_txt_path = "train.txt"
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

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(512),
        ColorJitter(
            brightness=0.5, 
            contrast=0.5, 
            saturation=0.6, 
            hue=0.2,
            apply_prob=0.5  # 显式设置应用概率为0.5
        ),
        RandomMaxFilter(
            num_regions=10,            # 处理区域数量
            kernel_size_range=(5, 15), # 滤波核大小范围
            threshold=0.2,             # 标签像素阈值
            apply_prob=0.5,            # 设置为0.5的概率应用
        ),
        RandomScratch(
                line_width_range=(1, 2),  # 修改为1像素宽的线条
                color_range=(0, 30),
                apply_prob=0.5,  # 为了测试目的设置为100%应用
                num_lines=2
            ),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)

# ------- 3. define model --------
# 检测可用的设备
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)
elif(model_name=='u2net_grain'):
    net = U2NET_GRAIN(3, 1)
elif(model_name=='u2netp_grain'):
    net = U2NETP_GRAIN(3, 1)

# 加载预训练模型
if os.path.exists(pretrained_model_path):
    print(f"Loading pretrained model from {pretrained_model_path}")
    net.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print("Pretrained model loaded successfully!")
else:
    print(f"Pretrained model not found at {pretrained_model_path}, starting from scratch")
    start_epoch = 0

net.to(device)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
def train_model():
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000 # save the model every 2000 iterations
    
    # 添加最佳准确率跟踪
    best_accuracy = 0.0

    for epoch in range(start_epoch, epoch_num):  # 从start_epoch开始训练
        net.train()
        epoch_loss = 0.0
        epoch_tar_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # 将数据移动到对应设备
            inputs_v = inputs.to(device)
            labels_v = labels.to(device)
            
            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs_v)
            
            # 根据模型类型处理不同的输出数量
            if model_name in ['u2net', 'u2netp']:
                # 原始U2NET模型有7个输出
                d0, d1, d2, d3, d4, d5, d6 = outputs
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
                # del temporary outputs and loss
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss
            else:
                # 新的U2NET_GRAIN模型有5个输出
                d0, d1, d2, d3, d4 = outputs
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, labels_v=labels_v)
                # del temporary outputs and loss
                del d0, d1, d2, d3, d4, loss2, loss
            
            # 计算当前batch的准确率（使用d0作为最终输出）
            batch_accuracy = calculate_accuracy(outputs[0], labels_v)
            epoch_accuracy += batch_accuracy

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # 累加每个batch的loss用于计算epoch平均loss
            epoch_loss += loss.data.item()
            epoch_tar_loss += loss2.data.item()
            batch_count += 1

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, accuracy: %3f \n" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, 
            running_loss / ite_num4val, running_tar_loss / ite_num4val, batch_accuracy))

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
        
        # 在每个epoch结束时计算平均loss和准确率
        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_tar_loss = epoch_tar_loss / batch_count
        avg_epoch_accuracy = epoch_accuracy / batch_count
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print(f"Average Accuracy: {avg_epoch_accuracy:.4f}")
        
        # # 更新学习率
        # scheduler.step(avg_epoch_accuracy)
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(
                model_dir, 
                f"{model_name}_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}_acc_{avg_epoch_accuracy:.4f}.pth"
            )
            torch.save(net.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with loss {avg_epoch_loss:.4f} and accuracy {avg_epoch_accuracy:.4f}")
        
        # 如果准确率提高了，保存最佳模型
        if avg_epoch_accuracy > best_accuracy:
            best_accuracy = avg_epoch_accuracy
            save_path = os.path.join(
                model_dir, 
                f"{model_name}_best_acc_{best_accuracy:.4f}_epoch_{epoch+1}.pth"
            )
            torch.save(net.state_dict(), save_path)
            print(f"New best accuracy achieved! Model saved with accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    # 确保模型保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    train_model()

