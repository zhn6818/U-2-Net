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

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

# 添加边界提取函数
def extract_boundary(mask, kernel_size=3):
    """
    提取掩码图像的边界
    Args:
        mask: 输入掩码 [batch, 1, H, W]
        kernel_size: 用于腐蚀/膨胀的核大小
    Returns:
        boundary: 边界掩码 [batch, 1, H, W]
    """
    # 定义边界提取的形态学操作
    if kernel_size > 1:
        pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        boundary = pool(mask) - mask
        return boundary
    else:
        # 使用Sobel算子提取边界（更精细的边界）
        # 水平方向的Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        # 垂直方向的Sobel算子
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        
        # 应用Sobel算子
        edge_x = F.conv2d(mask, sobel_x, padding=1)
        edge_y = F.conv2d(mask, sobel_y, padding=1)
        
        # 计算梯度幅值
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        # 二值化边界
        edge = (edge > 0.05).float()
        return edge
        
# 添加边界损失函数
def boundary_loss(pred, target, weight=1.0):
    """
    计算边界损失
    Args:
        pred: 预测掩码 [batch, 1, H, W]
        target: 目标掩码 [batch, 1, H, W]
        weight: 边界损失的权重
    Returns:
        loss: 边界损失
    """
    # 提取预测和目标的边界
    pred_boundary = extract_boundary(pred)
    target_boundary = extract_boundary(target)
    
    # 计算边界的BCE损失
    loss = F.binary_cross_entropy(pred_boundary, target_boundary, reduction='mean')
    
    # 也可以使用Dice Loss作为边界损失
    # smooth = 1.0
    # pred_flat = pred_boundary.contiguous().view(-1)
    # target_flat = target_boundary.contiguous().view(-1)
    # intersection = (pred_flat * target_flat).sum()
    # dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    # loss = dice_loss
    
    return weight * loss

# 计算IOU的函数，用于评估分割质量
def calculate_iou(pred, target, threshold=0.5):
    """
    计算预测的IOU (Intersection over Union)
    Args:
        pred: 预测的输出 
        target: 真实标签
        threshold: 二值化阈值
    Returns:
        iou: IOU值
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum().item()
    union = (pred + target).sum().item() - intersection
    
    if union < 1e-6:
        return 0.0
    return intersection / union

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, boundary_weight=0.5):
    """
    多尺度BCE损失与边界损失的融合
    """
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    # 计算边界损失 (只对最终输出d0计算)
    bdry_loss = boundary_loss(d0, labels_v, weight=boundary_weight)
    
    # 融合所有损失
    bce_loss_sum = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    loss = bce_loss_sum + bdry_loss
    
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f, boundary: %3f\n"%(
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), 
        loss4.data.item(), loss5.data.item(), loss6.data.item(), bdry_loss.data.item()))

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

model_name = 'u2net' #'u2netp'

# data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
# tra_image_dir = os.path.join('im_aug' + os.sep)
# tra_label_dir = os.path.join('gt_aug' + os.sep)
data_dir = "U2net_data/train_data/"
tra_image_dir = os.path.join('im_aug' + os.sep)
tra_label_dir = os.path.join('gt_aug' + os.sep)
# tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
# tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), '317_saved_models_512', model_name + os.sep)
print(f"Model directory: {model_dir}")

# 添加预训练模型路径
pretrained_model_path = os.path.join(os.getcwd(), '317_saved_models_512', model_name, 'u2net_epoch_695_loss_1.7671.pth')
# 从预训练模型文件名中提取起始epoch
start_epoch = 695  # 从文件名中提取的epoch数
print(f"Pretrained model: {pretrained_model_path}")
print(f"Starting from epoch: {start_epoch}")

epoch_num = 100000
batch_size_train = 2
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	# tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + '_Segmentation' + label_ext)
	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

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
        RandomCrop(460),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

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

# 添加学习率调度器，基于验证准确率调整学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, 
    verbose=True, threshold=0.001, threshold_mode='rel'
)

# ------- 5. training process --------
def train_model(boundary_weight=0.5):
    print("---start training...")
    print(f"Using boundary loss with weight: {boundary_weight}")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    running_boundary_loss = 0.0
    ite_num4val = 0
    save_frq = 2000 # save the model every 2000 iterations
    
    # 添加最佳准确率和IOU跟踪
    best_accuracy = 0.0
    best_iou = 0.0

    for epoch in range(start_epoch, epoch_num):  # 从start_epoch开始训练
        net.train()
        epoch_loss = 0.0
        epoch_tar_loss = 0.0
        epoch_accuracy = 0.0
        epoch_iou = 0.0
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
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, boundary_weight)
            
            # 计算当前batch的准确率（使用d0作为最终输出）
            batch_accuracy = calculate_accuracy(d0, labels_v)
            epoch_accuracy += batch_accuracy
            
            # 计算当前batch的IOU
            batch_iou = calculate_iou(d0, labels_v)
            epoch_iou += batch_iou

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # 累加每个batch的loss用于计算epoch平均loss
            epoch_loss += loss.data.item()
            epoch_tar_loss += loss2.data.item()
            batch_count += 1

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, accuracy: %3f, iou: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, 
            running_loss / ite_num4val, running_tar_loss / ite_num4val, batch_accuracy, batch_iou))

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
        avg_epoch_iou = epoch_iou / batch_count
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print(f"Average Accuracy: {avg_epoch_accuracy:.4f}")
        print(f"Average IOU: {avg_epoch_iou:.4f}")
        
        # 更新学习率
        scheduler.step(avg_epoch_accuracy)
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(
                model_dir, 
                f"{model_name}_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}_acc_{avg_epoch_accuracy:.4f}_iou_{avg_epoch_iou:.4f}.pth"
            )
            torch.save(net.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with loss {avg_epoch_loss:.4f}, accuracy {avg_epoch_accuracy:.4f}, and IOU {avg_epoch_iou:.4f}")
        
        # 如果准确率提高了，保存最佳模型
        if avg_epoch_accuracy > best_accuracy:
            best_accuracy = avg_epoch_accuracy
            save_path = os.path.join(
                model_dir, 
                f"{model_name}_best_acc_{best_accuracy:.4f}_epoch_{epoch+1}.pth"
            )
            torch.save(net.state_dict(), save_path)
            print(f"New best accuracy achieved! Model saved with accuracy: {best_accuracy:.4f}")
            
        # 如果IOU提高了，保存最佳模型
        if avg_epoch_iou > best_iou:
            best_iou = avg_epoch_iou
            save_path = os.path.join(
                model_dir, 
                f"{model_name}_best_iou_{best_iou:.4f}_epoch_{epoch+1}.pth"
            )
            torch.save(net.state_dict(), save_path)
            print(f"New best IOU achieved! Model saved with IOU: {best_iou:.4f}")

if __name__ == '__main__':
    # 确保模型保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 设置边界损失的权重，您可以根据任务需要调整这个值
    # 较大的值会让模型更关注边界，但可能影响整体分割
    boundary_weight = 0.8  
    
    train_model(boundary_weight=boundary_weight)

