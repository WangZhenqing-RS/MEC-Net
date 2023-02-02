# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:04:31 2022

@author: DELL
"""
import glob
import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import time
import torch.utils.data as D
from torchvision import transforms as T
from torch import nn
from nets.mecnet import MECNet
from nets.unetplusplus import UnetPlusPlus
from nets.hrnetv2 import hrnetv2
import torch.nn.functional as F
import tqdm

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

class OurDataset(D.Dataset):
    
    def __init__(self, image_paths, label_paths, mode, is_MEC):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.is_MEC = is_MEC
        self.as_tensor = T.Compose([
            T.ToTensor(),
        ])
    
    def __getitem__(self, index):
        
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_MEC:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            grad = (255 * grade(gray)).astype(np.uint8)
            image = cv2.merge([image, grad])
        
        return self.as_tensor(image), self.image_paths[index]
    
    def __len__(self):
        return self.len

def get_dataloader(image_paths, label_paths, mode, is_MEC, batch_size, 
                   shuffle, num_workers):
    dataset = OurDataset(image_paths, label_paths, mode, is_MEC)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

def test(models, model_paths, output_dir, test_loader, batch_size):
    
    for image, image_path in tqdm.tqdm(test_loader):
        
        output = np.zeros((image.shape[0],1,512,512))
        for i,model_path in enumerate(model_paths):
            model = models[i]
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                
                image_flip2 = torch.flip(image,[2])
                image_flip2 = image_flip2.cuda()
                image_flip3 = torch.flip(image,[3])
                image_flip3 = image_flip3.cuda()
                
                image = image.cuda()
                output1 = model(image).cpu().data.numpy() 
                output2 = torch.flip(model(image_flip2),[2]).cpu().data.numpy()
                output3 = torch.flip(model(image_flip3),[3]).cpu().data.numpy()
                
            output += output1 + output2 + output3
        
        output = output/(len(model_paths)*3)
        # output.shape: batch_size,classes,512,512
        for i in range(output.shape[0]):
            pred = output[i]
            threshold = 0.45
            pred[pred>=threshold] = 255
            pred[pred<threshold] = 0
            pred = np.uint8(pred)
            pred = np.squeeze(pred)
            image_name = image_path[i].split("\\")[-1]
            save_path = os.path.join(output_dir, image_name)
            # print(save_path)
            cv2.imwrite(save_path, pred)


class MyModel(nn.Module):
    def __init__(self, decoder_name, in_channels, classes):
        super().__init__()
        if decoder_name=="DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name="resnet101",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid",
                )
        elif decoder_name=="ResU-Net":
            self.model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid",
                )
        elif decoder_name=="PSP-Net":
            self.model = smp.PSPNet(
                encoder_name="resnet101",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid",
                )
        elif decoder_name=="HR-Net":
            self.model = hrnetv2(
                n_class=classes, 
                activation = "sigmoid",
                pretrained_model="pretrained_model/hrnetv2_w48-imagenet.pth")
        elif decoder_name=="MEC-Net":
            self.model = MECNet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation="sigmoid",
                )
        elif decoder_name=="Unet++":
            self.model = UnetPlusPlus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation="sigmoid",
                )
        self.name = decoder_name
    def forward(self, x):
        out = self.model(x)
        return out

class decoder_names:
    DeepLabV3Plus = "DeepLabV3Plus"
    ResUNet = "ResU-Net"
    PSPNet = "PSP-Net"
    HRNet = "HR-Net"
    MECNET = "MEC-Net"
    UentPlusPlus = "Unet++"

    
if __name__ == "__main__":
    
    # decoder_name_list = [
    #     decoder_names.PSPNet,
    #     decoder_names.ResUNet,
    #     decoder_names.HRNet,
    #     decoder_names.DeepLabV3Plus,
    #     decoder_names.UentPlusPlus,
    #     decoder_names.MECNET]
    
    # for decoder_name in decoder_name_list:
        
    #     model = MyModel(decoder_name = decoder_name,
    #                     in_channels=3, 
    #                     classes=1)
    #     models = [model]
    #     is_MEC = False
    #     if "MEC" in model.name:
    #         is_MEC = True
        
    #     model_paths = [f"../model/{model.name}_WHU_Building_Dataset.pth"]
    #     output_dir = f'../data/WHU_Building_Dataset/test/pred_{model.name}'
    #     if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    #     image_paths = glob.glob(r'../data/WHU_Building_Dataset/test/image/*.tif')
    #     batch_size = 16
    #     num_workers = 8
        
    #     test_loader = get_dataloader(image_paths, None, "test", is_MEC,
    #                                  batch_size, False, num_workers)
    #     save_paths = test(models, model_paths, output_dir, test_loader, batch_size)
    
      
    # model = MyModel(decoder_name = decoder_names.MECNET,
    #                 in_channels=3, 
    #                 classes=1)
    # models = [model]
    # is_MEC = False
    # if "MEC" in model.name:
    #     is_MEC = True
    
    # model_paths = [f"../model/{model.name}_WHU_Building_Dataset_build_building.pth"]
    # output_dir = f'../data/WHU_Building_Dataset/test/pred_{model.name}_build_building'
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # image_paths = glob.glob(r'../data/WHU_Building_Dataset/test/image/*.tif')
    # batch_size = 16
    # num_workers = 8
    # test_loader = get_dataloader(image_paths, None, "test", is_MEC,
    #                               batch_size, False, num_workers)
    # save_paths = test(models, model_paths, output_dir, test_loader, batch_size)
    
    
    models = [MyModel(decoder_name = decoder_names.ResUNet, in_channels=3, classes=1),
              MyModel(decoder_name = decoder_names.HRNet, in_channels=3, classes=1),
              MyModel(decoder_name = decoder_names.DeepLabV3Plus, in_channels=3, classes=1),
              MyModel(decoder_name = decoder_names.UentPlusPlus, in_channels=3, classes=1),]
    
    is_MEC = False
    
    model_paths = [f"../model/{models[0].name}_WHU_Building_Dataset.pth",
                   f"../model/{models[1].name}_WHU_Building_Dataset.pth",
                   f"../model/{models[2].name}_WHU_Building_Dataset.pth",
                   f"../model/{models[3].name}_WHU_Building_Dataset.pth"]
    
    output_dir = '../data/WHU_Building_Dataset/test/pred_fusion'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    image_paths = glob.glob(r'../data/WHU_Building_Dataset/test/image/*.tif')
    batch_size = 16
    num_workers = 8
    test_loader = get_dataloader(image_paths, None, "test", is_MEC,
                                  batch_size, False, num_workers)
    save_paths = test(models, model_paths, output_dir, test_loader, batch_size)
