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

def edge_loss(pred, target):
    # 计算预测和目标的梯度
    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    
    # 计算边缘损失
    edge_loss_x = F.mse_loss(pred_dx, target_dx)
    edge_loss_y = F.mse_loss(pred_dy, target_dy)
    
    return edge_loss_x + edge_loss_y

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	# 添加边缘感知损失
	edge_weight = 0.5  # 可调整的权重
	edge_loss_val = edge_loss(d0, labels_v)
	
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + edge_weight * edge_loss_val
	
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f, edge: %3f\n"%(
		loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(),
		loss4.data.item(), loss5.data.item(), loss6.data.item(), edge_loss_val.data.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

# data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
# tra_image_dir = os.path.join('im_aug' + os.sep)
# tra_label_dir = os.path.join('gt_aug' + os.sep)
data_dir = "/Users/zhanghaining/JH/projects/daizhuang_imgprocess/trainData/"
tra_image_dir = os.path.join('img' + os.sep)
tra_label_dir = os.path.join('label' + os.sep)
# tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
# tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'daizhuang_combine_saved_models_512', model_name + os.sep)
print(f"Model directory: {model_dir}")

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
        RescaleT(512),  # 直接缩放到目标尺寸
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

net.to(device)

# ------- 4. define optimizer --------
print("---define optimizer...")
# 提高初始学习率
optimizer = optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',  # 根据准确率来调整学习率
    factor=0.5,  # 学习率调整因子
    patience=3,  # 当3个epoch准确率没有提升时降低学习率
    verbose=True
)

# ------- 5. training process --------
def calculate_accuracy(pred, gt, threshold=0.1):
    """计算准确率
    Args:
        pred: 预测结果 tensor
        gt: ground truth tensor
        threshold: 二值化阈值
    Returns:
        accuracy: 准确率
    """
    # 将预测结果二值化
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > 0.5).float()  # GT通常已经是二值的
    
    # 计算准确率
    correct = (pred_binary == gt_binary).float().sum()
    total = gt_binary.numel()
    
    return (correct / total).item()

def validate(net, val_loader, device):
    net.eval()
    total_loss = 0
    total_acc = 0
    batch_count = 0
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            
            # 计算准确率
            accuracy = calculate_accuracy(d0, labels)
            
            total_loss += loss.item()
            total_acc += accuracy
            batch_count += 1
            
    return total_loss / batch_count, total_acc / batch_count

def train_model():
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    running_acc = 0.0
    ite_num4val = 0
    save_frq = 2000
    best_acc = 0.0  # 记录最佳准确率
    
    for epoch in range(0, epoch_num):
        net.train()
        epoch_loss = 0.0
        epoch_tar_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            inputs_v = inputs.to(device)
            labels_v = labels.to(device)
            
            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            
            # 计算当前batch的准确率
            accuracy = calculate_accuracy(d0, labels_v)
            
            # 如果准确率提升，立即保存模型
            if accuracy > best_acc:
                best_acc = accuracy
                save_path = os.path.join(
                    model_dir, 
                    f"{model_name}_acc_{accuracy:.4f}.pth"
                )
                torch.save(net.state_dict(), save_path)
                print(f"New best accuracy: {accuracy:.4f}, model saved to {save_path}")

            # 更新统计信息
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()
            running_acc += accuracy
            
            epoch_loss += loss.data.item()
            epoch_tar_loss += loss2.data.item()
            epoch_acc += accuracy
            batch_count += 1

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, acc: %3f" % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, 
                running_loss / ite_num4val, running_tar_loss / ite_num4val, 
                running_acc / ite_num4val))

            if ite_num % save_frq == 0:
                # 验证集评估
                val_loss, val_acc = validate(net, salobj_dataloader, device)
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
                
                # 如果验证集准确率提升，保存模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(net.state_dict(), model_dir + model_name + 
                             f"_bce_itr_{ite_num}_train_{running_loss/ite_num4val:.3f}_tar_{running_tar_loss/ite_num4val:.3f}_acc_{val_acc:.3f}.pth")
                    print(f"Model saved with new best accuracy: {val_acc:.4f}")
                
                running_loss = 0.0
                running_tar_loss = 0.0
                running_acc = 0.0
                net.train()
                ite_num4val = 0

        # 更新学习率
        scheduler.step(epoch_loss / batch_count)

if __name__ == '__main__':
    # 确保模型保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    train_model()

