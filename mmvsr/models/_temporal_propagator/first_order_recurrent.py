# Copyright (c) OpenMMLab. All rights reserved.
from logging import WARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from mmvsr.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmvsr.models.utils import flow_warp, make_layer
from mmvsr.registry import MODELS

@MODELS.register_module()
class FirstOrderRecurrentPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, num_blocks=30, backbone=None):

        super().__init__()

        self.mid_channels = mid_channels

        # propagation branches
        if backbone is None:
            self.backbone = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        else:
            self.backbone = backbone

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feats, flows, prev_feats):

        n, t, c, h, w = feats.size()

        outputs = []
        feat_prop = feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            curr_feat = feats[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                flow = flows[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([curr_feat, feat_prop, prev_feats[i]], dim=1)
            feat_prop = self.backbone(feat_prop)

            outputs.append(feat_prop)

        return torch.stack(outputs, dim=1)

class ResidualBlocksWithInputConv(BaseModule):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)
