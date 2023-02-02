# -*- coding: utf-8 -*-
"""
It is used to clearly view the model structure and the number of parameters
"""

import torch
from nets.mecnet import MECNet
from nets.unetplusplus import UnetPlusPlus
# from torchsummary import summary
from torch.autograd import Variable
import segmentation_models_pytorch as smp
import torch.nn as nn
# from thop import profile
# from pthflops import count_ops
import sys
from ptflops import get_model_complexity_info
from nets.hrnetv2 import hrnetv2
import time

class decoder_names:
    DeepLabV3Plus = "DeepLabV3Plus"
    ResUNet = "ResU-Net"
    PSPNet = "PSP-Net"
    HRNet = "HR-Net"

class MyModel(nn.Module):
    def __init__(self, decoder_name, in_channels, classes):
        super().__init__()
        if decoder_name=="DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name="resnet101",
                encoder_weights=None,
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid",
                )
        elif decoder_name=="ResU-Net":
            self.model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid",
                )
        elif decoder_name=="PSP-Net":
            self.model = smp.PSPNet(
                encoder_name="resnet101",
                encoder_weights=None,
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid",
                )
        elif decoder_name=="HR-Net":
            self.model = hrnetv2(n_class=classes, pretrained_model="pretrained_model/hrnetv2_w48-imagenet.pth")
        self.name = decoder_name
    def forward(self, x):
        out = self.model(x)
        return out
        
def get_model_flops(model):
    ost = sys.stdout
    flops, params = get_model_complexity_info(model, (3, 256, 256),
                            as_strings=True,
                            print_per_layer_stat=False,
                            ost=ost)
    
    print(f"{model.name} flops: {flops}  params: {params}.")
    

def infer_time(model):
    model.eval()
    x = torch.randn((1,3,512,512)).cuda()
    if model.name=="MEC-Net":
        x = torch.randn((1,4,512,512)).cuda()
    
    start_time = time.time()
    with torch.no_grad():
        y = model(x)
    infer_time = round(time.time()-start_time,4)
    print(f"{model.name} infer time: {infer_time}s")
        
if __name__ == '__main__':
    
    x = Variable(torch.rand(1,3,512,512)).cuda()
    # model = UnetPP_scSE(pretrained = True).cuda()
    # model = UnetPP_MEC(pretrained = True).cuda()
    # model = UnetPP(pretrained = True).cuda()
    
    model = UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
        ).cuda()
    # get_model_flops(model)
    infer_time(model)
    
    model = MECNet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
        ).cuda()
    # get_model_flops(model)
    infer_time(model)
    
    model = MyModel(decoder_name = decoder_names.ResUNet,
                    in_channels=3, 
                    classes=1).cuda()
    # get_model_flops(model)
    infer_time(model)
    
    model = MyModel(decoder_name = decoder_names.DeepLabV3Plus,
                    in_channels=3, 
                    classes=1).cuda()
    # get_model_flops(model)
    infer_time(model)
    
    model = MyModel(decoder_name = decoder_names.PSPNet,
                    in_channels=3, 
                    classes=1).cuda()
    # get_model_flops(model)
    infer_time(model)
    
    model = MyModel(decoder_name = decoder_names.HRNet,
                    in_channels=3, 
                    classes=1).cuda()
    # get_model_flops(model)
    infer_time(model)
    
    
    
    # torchsummary计算参数量
    # summary(model, input_size=(4,256,256))
    
    # 自定义代码计算参数量
    # param = count_param(model)
    # y = model(x)[0]
    # print('Output shape:',y.shape)
    # print(model.name+' total parameters: %.2fM (%d)'%(param/1e6,param))
    
    # thop计算参数量
    # inputs = torch.randn(1, 4, 256, 256).cuda()
    # flops, params = profile(model, (inputs,))
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000**3, params / 1000**2))
    
    # pthflops计算参数量
    # count_ops(model, inputs, mode="jit", print_readable=False, verbose=False)