# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageEnhance

#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}
class RandomSizedCrop(object):
	"""
	随机大小裁剪
	在min_size和max_size之间随机选择裁剪尺寸，实现多尺度数据增强
	"""
	def __init__(self, min_size, max_size):
		"""
		初始化函数
		Args:
			min_size: 最小裁剪尺寸
			max_size: 最大裁剪尺寸
		"""
		self.min_size = min_size
		self.max_size = max_size
		
	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		
		# 随机水平翻转
		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]
		
		h, w = image.shape[:2]
		
		# 随机选择裁剪尺寸
		new_size = random.randint(self.min_size, self.max_size)
		
		# 确保裁剪大小不超过图像尺寸
		new_size = min(new_size, min(h, w))
		
		# 设置裁剪的高度和宽度
		new_h, new_w = new_size, new_size
		
		# 确保不会越界
		if h <= new_h or w <= new_w:
			# 如果图像尺寸小于裁剪尺寸，直接返回原图
			return {'imidx': imidx, 'image': image, 'label': label}
		
		# 计算随机裁剪的起始位置
		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		
		# 执行裁剪 (移除了减1操作，确保裁剪得到的是完整大小)
		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]
		
		return {'imidx': imidx, 'image': image, 'label': label}

class ColorJitter(object):
    """
    颜色抖动数据增强
    随机调整图像的亮度、对比度、饱和度和色调，保持标签不变
    
    Args:
        brightness (float or tuple): 亮度调整的范围，如0.8表示在[0.8, 1.2]范围内调整
        contrast (float or tuple): 对比度调整的范围
        saturation (float or tuple): 饱和度调整的范围
        hue (float or tuple): 色调调整的范围，通常在[-0.1, 0.1]之间
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        
        # 只对图像进行处理，标签保持不变
        # 确保图像是3通道RGB
        if image.shape[2] == 1:
            # 如果是单通道图像，复制到3个通道
            image = np.concatenate([image, image, image], axis=2)
        
        # 随机应用变换, 概率为0.5
        if random.random() < 0.5:
            # 转换为PIL Image进行色彩调整
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            
            # 随机调整亮度
            if random.random() < 0.5 and self.brightness > 0:
                brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                enhancer = ImageEnhance.Brightness(image_pil)
                image_pil = enhancer.enhance(brightness_factor)
            
            # 随机调整对比度
            if random.random() < 0.5 and self.contrast > 0:
                contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                enhancer = ImageEnhance.Contrast(image_pil)
                image_pil = enhancer.enhance(contrast_factor)
            
            # 随机调整饱和度
            if random.random() < 0.5 and self.saturation > 0:
                saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                enhancer = ImageEnhance.Color(image_pil)
                image_pil = enhancer.enhance(saturation_factor)
            
            # 随机调整色调
            if random.random() < 0.5 and self.hue > 0:
                hue_factor = random.uniform(-self.hue, self.hue)
                image_np = np.array(image_pil)
                # 转换到HSV空间
                image_hsv = color.rgb2hsv(image_np)
                # 调整色调 (H通道)
                image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_factor) % 1.0
                # 转回RGB
                image_np = color.hsv2rgb(image_hsv)
                # 确保值在[0,1]范围内
                image = np.clip(image_np, 0, 1)
            else:
                # 如果没有调整色调，将PIL图像转回numpy数组
                image = np.array(image_pil).astype(np.float32) / 255.0
        
        return {'imidx': imidx, 'image': image, 'label': label}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]


		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]


		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {
			'imidx': torch.from_numpy(imidx.copy()).long(),
			'image': torch.from_numpy(tmpImg.copy()).float(),
			'label': torch.from_numpy(tmpLbl.copy()).float()
		}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image = io.imread(self.image_name_list[idx])
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample

# ===================== 添加多通道分割任务支持 ======================
class MultiChannelToTensorLab(object):
    """
    将图像和多通道标签转换为Tensor格式。
    此类专为多通道分割任务设计，可以处理任意数量的输出通道。
    """
    def __init__(self, flag=0, num_channels=2):
        """
        初始化函数
        Args:
            flag: 颜色空间标志，与原始ToTensorLab相同
                 0: RGB颜色 (默认)
                 1: Lab颜色
                 2: RGB+Lab颜色
            num_channels: 标签中的通道数，默认为2
        """
        self.flag = flag
        self.num_channels = num_channels

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # 归一化多通道标签
        tmpLbl = np.zeros(label.shape)
        if(np.max(label) < 1e-6):
            label = label
        else:
            # 对每个通道分别归一化
            for c in range(label.shape[2]):
                channel = label[:,:,c]
                if np.max(channel) > 1e-6:
                    label[:,:,c] = channel / np.max(channel)

        # 图像处理部分与原来ToTensorLab相同
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # normalize image to range [0,1]
            tmpImg[:,:,0] = (tmpImgt[:,:,0] - np.min(tmpImgt[:,:,0])) / (np.max(tmpImgt[:,:,0]) - np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1] - np.min(tmpImgt[:,:,1])) / (np.max(tmpImgt[:,:,1]) - np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2] - np.min(tmpImgt[:,:,2])) / (np.max(tmpImgt[:,:,2]) - np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0] - np.min(tmpImgtl[:,:,0])) / (np.max(tmpImgtl[:,:,0]) - np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1] - np.min(tmpImgtl[:,:,1])) / (np.max(tmpImgtl[:,:,1]) - np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2] - np.min(tmpImgtl[:,:,2])) / (np.max(tmpImgtl[:,:,2]) - np.min(tmpImgtl[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0] - np.mean(tmpImg[:,:,0])) / np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1] - np.mean(tmpImg[:,:,1])) / np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2] - np.mean(tmpImg[:,:,2])) / np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3] - np.mean(tmpImg[:,:,3])) / np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4] - np.mean(tmpImg[:,:,4])) / np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5] - np.mean(tmpImg[:,:,5])) / np.std(tmpImg[:,:,5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            tmpImg[:,:,0] = (tmpImg[:,:,0] - np.min(tmpImg[:,:,0])) / (np.max(tmpImg[:,:,0]) - np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1] - np.min(tmpImg[:,:,1])) / (np.max(tmpImg[:,:,1]) - np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2] - np.min(tmpImg[:,:,2])) / (np.max(tmpImg[:,:,2]) - np.min(tmpImg[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0] - np.mean(tmpImg[:,:,0])) / np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1] - np.mean(tmpImg[:,:,1])) / np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2] - np.mean(tmpImg[:,:,2])) / np.std(tmpImg[:,:,2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
                tmpImg[:,:,1] = (image[:,:,0] - 0.485) / 0.229
                tmpImg[:,:,2] = (image[:,:,0] - 0.485) / 0.229
            else:
                tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
                tmpImg[:,:,1] = (image[:,:,1] - 0.456) / 0.224
                tmpImg[:,:,2] = (image[:,:,2] - 0.406) / 0.225

        # 转置后返回
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {
            'imidx': torch.from_numpy(imidx.copy()).long(),
            'image': torch.from_numpy(tmpImg.copy()).float(),
            'label': torch.from_numpy(tmpLbl.copy()).float()
        }

class MultiChannelSalObjDataset(Dataset):
    """
    多通道分割数据集，支持任意数量的分割通道
    """
    def __init__(self, img_name_list, lbl_name_list, transform=None, num_channels=2):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform
        self.num_channels = num_channels

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 加载图像
        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        # 处理图像格式
        if len(image.shape) == 2:  # 灰度图转换为3通道
            image = image[:, :, np.newaxis]
            image = np.concatenate([image, image, image], axis=2)
        
        # 读取标签
        if len(self.label_name_list) == 0:
            # 如果没有标签，创建全零标签
            label = np.zeros((image.shape[0], image.shape[1], self.num_channels))
        else:
            # 读取标签文件
            label = io.imread(self.label_name_list[idx])
            
            # 处理标签格式
            if len(label.shape) == 2:  # 单通道标签
                # 如果标签是单通道的，但需要多通道输出
                if self.num_channels > 1:
                    # 这里我们假设每个通道分别表示不同的类别
                    # 创建一个多通道的标签
                    multi_label = np.zeros((label.shape[0], label.shape[1], self.num_channels))
                    
                    # 根据像素值处理多通道标签，像素值代表类别
                    # 对于背景(值为0)和其他类别(值为1,2...)分别创建通道
                    for c in range(self.num_channels):
                        multi_label[:, :, c] = (label == c).astype(float)
                    
                    label = multi_label
                else:
                    # 单通道标签，保持不变，但添加通道维度
                    label = label[:, :, np.newaxis]
            
            # 确保标签具有正确的通道数
            if label.shape[2] != self.num_channels:
                raise ValueError(f"标签通道数 {label.shape[2]} 与期望的通道数 {self.num_channels} 不符")

        # 组装样本
        sample = {'imidx': imidx, 'image': image, 'label': label}
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample
