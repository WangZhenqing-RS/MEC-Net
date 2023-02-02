# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:17:59 2022

@author: DELL
"""
from typing import Optional, Union, List
from .decoder import MEC_Decoder
from .encoder import get_encoder
from .base import SegmentationModel
from .base import SegmentationHead


class MECNet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: List[int] = (256, 128, 64, 32),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = MEC_Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=2,
        )
        
        self.name = "MEC-Net"
        self.initialize()
