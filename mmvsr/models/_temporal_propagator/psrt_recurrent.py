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

        if history_feats is not None:
            assert history_feats.shape[1] == 2 # history feature should be T=2
            assert history_flows.shape[1] == 2 # history flow should be of T=2
            history_feats = [history_feats[:, 0, ...], history_feats[:, 1, ...]]
            history_flows = [history_flows[:, 0, ...], history_flows[:, 1, ...]]
        else:
            history_feats = [curr_feats[:, feat_indices[0], ...], curr_feats[:, feat_indices[0], ...]]
            history_flows = [flows.new_zeros(n, 2, h, w), flows.new_zeros(n, 2, h, w)]
            
        out_feats = []

        for i in range(0, t):
            
            x = curr_feats[:, feat_indices[i], ...]
            y2, y1 = history_feats
            f2, f1 = history_flows
            a1 = self.warper(y1, f1.permute(0, 2, 3, 1))
            f2 = f1 + self.warper(f2, f1.permute(0, 2, 3, 1))
            a2 = self.warper(y2, f2.permute(0, 2, 3, 1))

            c = torch.stack([a2, a1, x], dim=1)
            o = self.fextor(c) + x

            out_feats.append(o.clone())

            if i == t - 1: # for the last iter, need to to update history
                break
            # update history feats and flows
            history_feats = [history_feats[1], o]
            history_flows = [history_flows[1], flows[:, feat_indices[i], ...]]

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
