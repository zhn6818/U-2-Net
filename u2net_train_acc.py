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

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss

# 在muti_bce_loss_fusion函数后添加计算准确率的函数
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
batch_size_train = 16
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
    save_frq = 2000  # 仍然保留这个参数用于打印统计信息
    
    # 添加最佳准确率跟踪
    best_accuracy = 0.0
    
    for epoch in range(0, epoch_num):
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

            inputs_v = inputs.to(device)
            labels_v = labels.to(device)
            
            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            # 计算当前batch的准确率（使用d0作为最终输出）
            batch_accuracy = calculate_accuracy(d0, labels_v)
            epoch_accuracy += batch_accuracy

            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()
            epoch_loss += loss.data.item()
            epoch_tar_loss += loss2.data.item()
            batch_count += 1

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, accuracy: %3f" % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, 
                running_loss / ite_num4val, running_tar_loss / ite_num4val, batch_accuracy))

            if ite_num % save_frq == 0:
                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0
        
        # 计算epoch的平均准确率和损失
        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_accuracy = epoch_accuracy / batch_count
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print(f"Average Accuracy: {avg_epoch_accuracy:.4f}")
        
        # 只有当准确率提高时才保存模型
        if avg_epoch_accuracy > best_accuracy:
            best_accuracy = avg_epoch_accuracy
            save_path = os.path.join(
                model_dir, 
                f"{model_name}_best_acc_{best_accuracy:.4f}_epoch_{epoch+1}.pth"
            )
            # 只保存模型权重
            torch.save(net.state_dict(), save_path)
            print(f"New best accuracy achieved! Model saved with accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    # 确保模型保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    train_model()

