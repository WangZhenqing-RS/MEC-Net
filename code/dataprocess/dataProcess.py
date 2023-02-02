# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:04:28 2021

@author: DELL
"""

import torch.utils.data as D
from torchvision import transforms as T
import numpy as np
import torch
import albumentations as A
import cv2
from .build_building import build_building

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi=np.min(dst)
    ma=np.max(dst)
    res=(dst-mi)/(0.000000001+(ma-mi))
    res[np.isnan(res)]=0
    return res


#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集IoU
def cal_val_IoU(model, loader):
    val_IoU = []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output[output>=0.5] = 1
        output[output<0.5] = 0
        IoU = cal_IoU(output, target)
        val_IoU.append(IoU)
    return val_IoU

# 计算IoU
def cal_IoU(pred, mask):
    p = (mask == 1).int().reshape(-1)
    t = (pred == 1).int().reshape(-1)
    uion = p.sum() + t.sum()
    overlap = (p*t).sum()
    #  0.0001防止除零
    IoU = 2*overlap/(uion + 0.0001)
    return IoU.abs().data.cpu().numpy()

class OurDataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode, is_MEC, use_build_building):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.is_MEC = is_MEC
        self.use_build_building = use_build_building
        self.len = len(image_paths)
        self.transform = A.Compose([
            # # # 亮度变换
            # A.OneOf([
            #     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            #     A.RandomBrightness(limit=0.1, p=0.5),
            # ], p=1),
            # 垂直翻转
            A.HorizontalFlip(p=0.5),
            # 水平翻转
            A.VerticalFlip(p=0.5),
            # 旋转90度
            A.RandomRotate90(p=0.5),
            # # 平移.尺度加旋转变换
            # A.ShiftScaleRotate(rotate_limit=1, p=0.5),
            # # 增加模糊
            # A.OneOf([
            #     A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),
            # ], p=0.5),
            # # 随机擦除
            # A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
        ])
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == "train":
            
            label = cv2.imread(self.label_paths[index],0)
            
            # build_building 增强
            if self.use_build_building:
                # 如果图像中房屋较少
                if np.sum(label==255)/(label.shape[0]**2) <0.1:
                    if np.random.random()<0.5:
                        label[label==1]=255
                        random_value = np.random.random()
                        copy_index = int(np.random.random()*len(self.image_paths))
                        image_copy = cv2.imread(self.image_paths[copy_index])
                        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                        label_copy = cv2.imread(self.label_paths[copy_index],0)
                        label, image = build_building(image_copy, label_copy, image, label, random_value)
            
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            
            if self.is_MEC:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                grad = (255 * grade(gray)).astype(np.uint8)
                image = cv2.merge([image, grad])
            
            label = label / 255
            label = label.reshape((1,) + label.shape)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":
            label = cv2.imread(self.label_paths[index],0)
            if self.is_MEC:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                grad = (255 * grade(gray)).astype(np.uint8)
                image = cv2.merge([image, grad])
            
            label = label / 255
            label = label.reshape((1,) + label.shape)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "test":
            return self.as_tensor(image), self.image_paths[index]
    
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_paths, label_paths, mode, batch_size, is_MEC, 
                   use_build_building, shuffle, num_workers):
    
    dataset = OurDataset(image_paths, label_paths, mode, is_MEC, use_build_building)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

