# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:35:30 2022

@author: DELL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base.modules import Attention


class decoder_block(nn.Module):
    def __init__(self, high_in_size, low_in_size, out_size, n_concat=2, attention="scse"):
        super(decoder_block, self).__init__()
        self.conv1 = nn.Conv2d(low_in_size+(n_concat-1)*out_size, out_size, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(high_in_size, out_size, 1))
        self.relu = nn.ReLU(inplace = True)
        self.attention = Attention(attention, in_channels=out_size)

    def forward(self, high_feature, *low_feature):
        outputs = self.up(high_feature)
        for feature in low_feature:
            outputs = torch.cat([outputs, feature], 1)
        
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.attention(outputs)
        return outputs
    
class MEC_block(nn.Module):
    def __init__(self, high_in_size, low_in_size, out_size, n_concat=2):
        super(MEC_block, self).__init__()
        edge_channel = 1
        self.conv1 = nn.Conv2d(low_in_size+(n_concat-1)*out_size + edge_channel, out_size, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(high_in_size, out_size, 1))
        self.relu = nn.ReLU(inplace = True)
        self.attention = Attention("scse", in_channels=out_size)

    def forward(self, high_feature, *low_feature):
        outputs = self.up(high_feature)
        for feature in low_feature:
            outputs = torch.cat([outputs, feature], 1)
        
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.attention(outputs)
        return outputs




class MEC_Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()
        
        encoder_channels = encoder_channels[1:]
        decoder_channels = decoder_channels[::-1]
        
        self.up_concat01 = MEC_block(encoder_channels[1], encoder_channels[0], decoder_channels[0], 2)
        self.up_concat11 = decoder_block(encoder_channels[2], encoder_channels[1], decoder_channels[1], 2)
        self.up_concat21 = decoder_block(encoder_channels[3], encoder_channels[2], decoder_channels[2], 2)
        self.up_concat31 = decoder_block(encoder_channels[4], encoder_channels[3], decoder_channels[3], 2)
        
        self.up_concat02 = MEC_block(decoder_channels[1], encoder_channels[0], decoder_channels[0], 3)
        self.up_concat12 = decoder_block(decoder_channels[2], encoder_channels[1], decoder_channels[1], 3)
        self.up_concat22 = decoder_block(decoder_channels[3], encoder_channels[2], decoder_channels[2], 3)
        
        self.up_concat03 = MEC_block(decoder_channels[1], encoder_channels[0], decoder_channels[0], 4)
        self.up_concat13 = decoder_block(decoder_channels[2], encoder_channels[1], decoder_channels[1], 4)
        
        self.up_concat04 = MEC_block(decoder_channels[1], encoder_channels[0], decoder_channels[0], 5)

    def forward(self, *features):
        
        inputs = features[0]
        print(inputs.shape)
        canny = inputs[:, -1, :, :]
        canny = torch.unsqueeze(canny, dim=1)
        canny = F.interpolate(canny, scale_factor=0.5)
        
        x_0_0 = features[1]
        x_1_0 = features[2]
        x_2_0 = features[3]
        x_3_0 = features[4]
        x_4_0 = features[5]
        
        
        # column : 1
        x_0_1 = self.up_concat01(x_1_0, x_0_0, canny)
        x_1_1 = self.up_concat11(x_2_0, x_1_0)
        x_2_1 = self.up_concat21(x_3_0, x_2_0)
        x_3_1 = self.up_concat31(x_4_0, x_3_0)
        
        # column : 2
        x_0_2 = self.up_concat02(x_1_1, x_0_0, x_0_1, canny)
        x_1_2 = self.up_concat12(x_2_1, x_1_0, x_1_1)
        x_2_2 = self.up_concat22(x_3_1, x_2_0, x_2_1)
        # column : 3
        x_0_3 = self.up_concat03(x_1_2, x_0_0, x_0_1, x_0_2, canny)
        x_1_3 = self.up_concat13(x_2_2, x_1_0, x_1_1, x_1_2)
        # column : 4
        x_0_4 = self.up_concat04(x_1_3, x_0_0, x_0_1, x_0_2, x_0_3, canny)
        
        return x_0_4

class UnetPlusPlus_Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()
        
        encoder_channels = encoder_channels[1:]
        decoder_channels = decoder_channels[::-1]
        
        self.up_concat01 = decoder_block(encoder_channels[1], encoder_channels[0], decoder_channels[0], 2, attention=None)
        self.up_concat11 = decoder_block(encoder_channels[2], encoder_channels[1], decoder_channels[1], 2, attention=None)
        self.up_concat21 = decoder_block(encoder_channels[3], encoder_channels[2], decoder_channels[2], 2, attention=None)
        self.up_concat31 = decoder_block(encoder_channels[4], encoder_channels[3], decoder_channels[3], 2, attention=None)
        
        self.up_concat02 = decoder_block(decoder_channels[1], encoder_channels[0], decoder_channels[0], 3, attention=None)
        self.up_concat12 = decoder_block(decoder_channels[2], encoder_channels[1], decoder_channels[1], 3, attention=None)
        self.up_concat22 = decoder_block(decoder_channels[3], encoder_channels[2], decoder_channels[2], 3, attention=None)
        
        self.up_concat03 = decoder_block(decoder_channels[1], encoder_channels[0], decoder_channels[0], 4, attention=None)
        self.up_concat13 = decoder_block(decoder_channels[2], encoder_channels[1], decoder_channels[1], 4, attention=None)
        
        self.up_concat04 = decoder_block(decoder_channels[1], encoder_channels[0], decoder_channels[0], 5, attention=None)

    def forward(self, *features):
        
        # inputs = features[0]
        # canny = inputs[:, -1, :, :]
        # canny = torch.unsqueeze(canny, dim=1)
        # canny = F.interpolate(canny, scale_factor=0.5)
        
        x_0_0 = features[1]
        x_1_0 = features[2]
        x_2_0 = features[3]
        x_3_0 = features[4]
        x_4_0 = features[5]
        
        
        # column : 1
        x_0_1 = self.up_concat01(x_1_0, x_0_0)
        x_1_1 = self.up_concat11(x_2_0, x_1_0)
        x_2_1 = self.up_concat21(x_3_0, x_2_0)
        x_3_1 = self.up_concat31(x_4_0, x_3_0)
        
        # column : 2
        x_0_2 = self.up_concat02(x_1_1, x_0_0, x_0_1)
        x_1_2 = self.up_concat12(x_2_1, x_1_0, x_1_1)
        x_2_2 = self.up_concat22(x_3_1, x_2_0, x_2_1)
        # column : 3
        x_0_3 = self.up_concat03(x_1_2, x_0_0, x_0_1, x_0_2)
        x_1_3 = self.up_concat13(x_2_2, x_1_0, x_1_1, x_1_2)
        # column : 4
        x_0_4 = self.up_concat04(x_1_3, x_0_0, x_0_1, x_0_2, x_0_3)
        
        return x_0_4
    