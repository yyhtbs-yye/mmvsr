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
    
    def __init__(self, mid_channels=64,
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
        

    def forward(self, curr_feats, flows, history_feats=None, history_flows=None):

        n, t, c, h, w = curr_feats.size()

        feat_indices = list(range(-1, -t - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(t))

        out_feats = list()
        if history_feats is not None:
            prop_feat = history_feats[:, -1, ...]
            prev_prop_feat = history_feats[:, -2, ...]
        else:
            prop_feat = curr_feats.new_zeros(n, self.mid_channels, h, w)
            prev_prop_feat = curr_feats.new_zeros(n, self.mid_channels, h, w)

        if history_feats is not None: # history_flows = [-2, -1]
            n1_flow = history_flows[:, -1, ...]
            n1_cond = self.warper(prop_feat, n1_flow.permute(0, 2, 3, 1))

            n2_flow = history_flows[:, -2, ...]
            n2_flow = n1_flow + self.warper(n2_flow, n1_flow.permute(0, 2, 3, 1))
            n2_cond = self.warper(prev_prop_feat, n2_flow.permute(0, 2, 3, 1))
        else:
            n1_cond = curr_feats[:, feat_indices[0], ...]
            n2_cond = curr_feats[:, feat_indices[0], ...]


        for i in range(0, t):
            
            curr_feat = curr_feats[:, feat_indices[i], ...]
            
            if i == 1:
                n1_flow = flows[:, feat_indices[0], ...]
                n1_cond = self.warper(prop_feat, n1_flow.permute(0, 2, 3, 1))
                if history_feats is not None:
                    n2_flow = history_flows[:, -1, ...]
                    n2_flow = n1_flow + self.warper(n2_flow, n1_flow.permute(0, 2, 3, 1))
                    n2_cond = self.warper(history_feats[:, -1, ...], n2_flow.permute(0, 2, 3, 1))
                else:
                    n2_cond = curr_feat

            elif i > 1:
                n1_flow = flows[:, feat_indices[i - 1], ...]
                n1_cond = self.warper(prop_feat, n1_flow.permute(0, 2, 3, 1))
                n2_flow = flows[:, feat_indices[i - 2], ...]
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
