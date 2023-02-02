# -*- coding: utf-8 -*-
"""
Code for model training
"""

import warnings
import time
import glob
import os
import random

import numpy as np
import torch
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from pytorch_toolbelt import losses as L

from dataprocess.dataProcess import get_dataloader, cal_val_IoU
from nets.mecnet import MECNet
from nets.unetplusplus import UnetPlusPlus
from nets.hrnetv2 import hrnetv2
from loss.SoftBCE import SoftBCELoss

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


def train(model, epoches, batch_size, train_image_paths, train_label_paths, 
          val_image_paths, val_label_paths, model_path, early_stop, is_MEC, 
          use_build_building):
    
    train_loader = get_dataloader(train_image_paths, train_label_paths,
                                  "train", batch_size, is_MEC, use_build_building,
                                  shuffle=True, num_workers=8)
    valid_loader = get_dataloader(val_image_paths, val_label_paths,
                                  "val", batch_size, is_MEC, use_build_building, 
                                  shuffle=False, num_workers=8)
    
    model.to(DEVICE)
    
    # AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=2, # 初始restart的epoch数目
        T_mult=2, # 重启之后因子,即每个restart后，T_0 = T_0 * T_mult
        eta_min=1e-5 # 最低学习率
        )
    
    # 损失函数采用SoftCrossEntropyLoss+DiceLoss
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    DiceLoss_fn=DiceLoss(mode='binary',
                         from_logits=False)
    # 交叉熵
    BCEloss_fn=SoftBCELoss(smooth_factor=0.1)
    loss_fn = L.JointLoss(first=DiceLoss_fn, second=BCEloss_fn,
                          first_weight=0.5, second_weight=0.5).cuda()
    
    log = ""
    header = r'Epoch/EpochNum | TrainLoss | ValidIoU | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.3f} | {:9.3f} | {:9.2f}'
    print(header)
    log += header+"\n"
    
    # 记录当前验证集最优IoU,以判定是否保存当前模型
    best_IoU = 0
    best_IoU_epoch = 0
    train_loss_epochs, val_IoU_epochs, lr_epochs = [], [], []
    # 开始训练
    for epoch in range(1, epoches+1):
        # 存储训练集每个batch的loss
        losses = []
        start_time = time.time()
        model.train()
        model.to(DEVICE)
        for batch_index, (image, target) in enumerate(train_loader):
            image, target = image.to(DEVICE), target.to(DEVICE)
            # 在反向传播前要手动将梯度清零
            optimizer.zero_grad()
            # 模型推理得到输出
            output = model(image)
            output=output.to(torch.float32)
            target=target.to(torch.float32)
            # 求解该batch的loss
            # print(output.shape, target.shape)
            loss = loss_fn(output, target)
            # 反向传播求解梯度
            loss.backward()
            # 更新权重参数
            optimizer.step()
            losses.append(loss.item())

        # 余弦退火调整学习率
        scheduler.step()
        # 计算验证集IoU
        val_IoU = cal_val_IoU(model, valid_loader)
        # 保存当前epoch的train_loss.val_IoU.lr_epochs
        train_loss_epochs.append(np.array(losses).mean())
        val_IoU_epochs.append(np.mean(val_IoU))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        
        # 输出进程
        log_raw_line = raw_line.format(epoch, epoches, np.array(losses).mean(), 
                                       np.mean(val_IoU), 
                                       (time.time()-start_time)/60**1)
        print(log_raw_line, end="")  
        log += log_raw_line
        
        if best_IoU < np.stack(val_IoU).mean():
            best_IoU = np.stack(val_IoU).mean()
            best_IoU_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("  valid IoU is improved. the model is saved.")
            log += "  valid IoU is improved. the model is saved.\n"
        else:
            print("")
            log += "\n"
            if (epoch - best_IoU_epoch) >= early_stop:
                break
    return train_loss_epochs, val_IoU_epochs, lr_epochs, log

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

class dataset_names:
    WHU_Building_Dataset = "WHU_Building_Dataset"
    Inria_Building_Dataset = "Inria_Building_Dataset"
    Massachusetts_Building_Dataset = "Massachusetts_Building_Dataset"
    
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True   
    torch.manual_seed(seed)
    
if __name__ == '__main__':
    
    decoder_name_list = [
        # decoder_names.PSPNet,
        # decoder_names.ResUNet,
        # decoder_names.HRNet,
        # decoder_names.DeepLabV3Plus,
        # decoder_names.UentPlusPlus,
        decoder_names.MECNET]
    
    for decoder_name in decoder_name_list:
        
        seed_it(2022)
        
        # 模型参数设置
        classes = 1
        channels = 3
        epoches = 60#125
        batch_size = 2
        early_stop = 50
        
        '''
        # ---------------------------------------------------------------------
        重要修改参数
        '''
        dataset_name = dataset_names.Inria_Building_Dataset
        use_build_building = False
        model = MyModel(decoder_name = decoder_name,
                        in_channels=3, 
                        classes=1)
        '''
        # ---------------------------------------------------------------------
        '''
        
        
        is_MEC = False
        if "MEC" in model.name:
            is_MEC = True
            
        train_image_paths = glob.glob(f'../data/{dataset_name}/train/image/*.tif')
        train_label_paths = glob.glob(f'../data/{dataset_name}/train/label/*.tif')
        val_image_paths = glob.glob(f'../data/{dataset_name}/val/image/*.tif')
        val_label_paths = glob.glob(f'../data/{dataset_name}/val/label/*.tif')
        
        model_name = model.name + "_" + dataset_name
        if use_build_building:
            model_name += "_build_building"
        model_path = f"../model/{model_name}.pth"
        
        print(model_name)
        train_loss_epochs, val_IoU_epochs, lr_epochs, log = train(model,
                                                                  epoches,
                                                                  batch_size,
                                                                  train_image_paths, 
                                                                  train_label_paths, 
                                                                  val_image_paths, 
                                                                  val_label_paths,
                                                                  model_path, 
                                                                  early_stop,
                                                                  is_MEC,
                                                                  use_build_building)
        
        with open(f"../log/{model_name}.txt","w") as f:
            f.write(model_name+"\n"+log)
        if(True):
            import matplotlib.pyplot as plt
            epochs = range(1, len(train_loss_epochs) + 1)
            plt.plot(epochs, train_loss_epochs, 'r', label = 'train loss')
            plt.plot(epochs, val_IoU_epochs, 'b', label = 'val IoU')
            plt.title('train loss and val IoU')
            plt.legend()
            plt.savefig(f"../log/{model_name}_train_log_train loss and val IoU.png",dpi = 300)
            plt.figure()
            plt.plot(epochs, lr_epochs, 'r', label = 'learning rate')
            plt.title('learning rate')
            plt.legend()
            plt.savefig(f"../log/{model_name}_train_log_learning rate.png", dpi = 300)
            # plt.show()