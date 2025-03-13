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

# 在定义优化器后添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# ------- 5. training process --------
def validate(net, val_loader, device):
    net.eval()
    total_loss = 0
    total_edge_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            edge_loss_val = edge_loss(d0, labels)
            
            total_loss += loss.item()
            total_edge_loss += edge_loss_val.item()
            
    return total_loss / len(val_loader), total_edge_loss / len(val_loader)

def train_model():
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000 # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()
        epoch_loss = 0.0
        epoch_tar_loss = 0.0
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
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

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

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
        
        # 在每个epoch结束时计算平均loss
        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_tar_loss = epoch_tar_loss / batch_count
        
        # 每两个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(
                model_dir, 
                f"{model_name}_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}.pth"
            )
            torch.save(net.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with loss {avg_epoch_loss:.4f}")

        # 在每个epoch结束时更新学习率
        scheduler.step(avg_epoch_loss)

        # 验证阶段
        val_loss, val_edge_loss = validate(net, salobj_dataloader, device)
        print(f"Validation Loss: {val_loss:.4f}, Edge Loss: {val_edge_loss:.4f}")

if __name__ == '__main__':
    # 确保模型保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    train_model()

