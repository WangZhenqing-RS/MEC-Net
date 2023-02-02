# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:16:42 2022

@author: DELL
"""

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

class SoftBCELoss(nn.Module):

    def __init__(
        self,
        smooth_factor: Optional[float] = None,
    ):

        super().__init__()
        self.smooth_factor = smooth_factor

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy(
            y_pred, soft_targets
        )

        loss = loss.mean()

        return loss