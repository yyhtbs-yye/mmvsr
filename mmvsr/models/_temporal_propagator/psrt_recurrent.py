from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.models.archs import ResidualBlockNoBN
from mmvsr.models.utils import flow_warp, make_layer
from mmvsr.registry import MODELS

import einops

C_DIM = -3

@MODELS.register_module()
class PSRTRecurrentPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, n_frames=7,
                 fextor_def=None, fextor_args=None,
                 warper_def=None, warper_args=None,
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels

        self.is_reversed = is_reversed

        if fextor_def is None:
            self.fextor = ResidualBlocksWithInputConv(in_channels=mid_channels+3, out_channels=mid_channels, num_blocks=30)
        else:        
            self.fextor = fextor_def(**fextor_args)

        if warper_def is None:
            self.warper = Warper()
        else: 
            self.warper = warper_def(**warper_args)        
        
        self.feat_indices = list(range(-1, -n_frames - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(n_frames))

    def forward(self, curr_feats, flows, prev_feats=None):

        n, t, c, h, w = curr_feats.size()

        out_feats = list()
        if prev_feats:
            prop_feat = prev_feats
        else:
            prop_feat = curr_feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            
            curr_feat = curr_feats[:, self.feat_indices[i], ...]
            n1_cond = curr_feat
            n2_cond = curr_feat

            if i > 0:
                n1_flow = flows[:, self.feat_indices[i - 1], ...]
                n1_cond = self.warper(prop_feat, n1_flow.permute(0, 2, 3, 1))
                
                n2_feat = torch.zeros_like(prop_feat)
                n2_flow = torch.zeros_like(n1_flow)
                n2_cond = torch.zeros_like(n1_cond)
                if i > 1:
                    n2_flow = flows[:, self.feat_indices[i - 2], :, :, :]
                    # Compute second-order optical flow using first-order flow.
                    n2_flow = n1_flow + self.warper(n2_flow, n1_flow.permute(0, 2, 3, 1))
                    n2_feat = out_feats[-2] # The position of 'n-2' to match 'n'
                    n2_cond = self.warper(n2_feat, n2_flow.permute(0, 2, 3, 1))

            cat_feat = torch.stack([curr_feat, n1_cond, n2_cond], dim=1)
            prop_feat = self.fextor(cat_feat) + curr_feat

            out_feats.append(prop_feat.clone())

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)

class Warper(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, feat, flow):
        return flow_warp(feat, flow)

class ResidualBlocksWithInputConv(BaseModule):

    def __init__(self, in_channels, out_channels=64, num_blocks=30, ndim=4):
        super().__init__()

        if ndim == 4:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
            )
        elif ndim == 5: # For the case of 3 inputs frames
            self.main = nn.Sequential(
                nn.Conv2d(in_channels*3, out_channels, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
            )

        self.ndim = ndim

    def forward(self, feat):
        if self.ndim == 5:
            feat = einops.rearrange(feat, 'b t c h w -> b (t c) h w')

        return self.main(feat)
