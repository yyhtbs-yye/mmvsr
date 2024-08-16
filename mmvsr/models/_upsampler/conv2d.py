# Copyright (c) Yuhang Ye. All rights reserved.
from logging import WARNING

import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmvsr.models.archs import PixelShufflePack
from mmvsr.registry import MODELS

@MODELS.register_module()
class Type1Upsampler(BaseModule): # BasicVSR Upsampler
    def __init__(self, mid_channels=64):

        super().__init__()

        self.mid_channels = mid_channels

        # upsample
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feats, lrs):
        
        feats = self.lrelu(self.fusion(feats))
        feats = self.lrelu(self.upsample1(feats))
        feats = self.lrelu(self.upsample2(feats))
        feats = self.lrelu(self.conv_hr(feats))
        feats = self.conv_last(feats)
        feats += self.img_upsample(lrs)
        
        return feats