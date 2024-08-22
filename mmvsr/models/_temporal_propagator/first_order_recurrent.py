from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.models.archs import ResidualBlockNoBN
from mmvsr.models.utils import flow_warp, make_layer
from mmvsr.registry import MODELS

C_DIM = -3

@MODELS.register_module()
class FirstOrderRecurrentPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, num_blocks=30, 
                 fextor_def=None, aligner_def=None,
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        self.is_reversed = is_reversed

        # Function definitions or classes to create fextor and aligner
        self.fextor_def = fextor_def if fextor_def is not None else ResidualBlocksWithInputConv
        self.aligner_def = aligner_def if aligner_def is not None else Alignment

        # Placeholders for dynamically created modules
        self.fextor = None
        self.aligner = None

        self.is_first = True


    def _initialize_submodules(self, curr_feats, prev_feats, device):

        # This is to mimic the Dense Connection
        input_channels = self.mid_channels + sum([it.shape[C_DIM] for it in prev_feats]) + curr_feats.shape[C_DIM]
        
        if self.fextor is None:
            self.fextor = self.fextor_def(input_channels, self.mid_channels, self.num_blocks).to(device)

        if self.aligner is None:
            self.aligner = self.aligner_def().to(device)

        self.is_first = False

    def forward(self, curr_feats, flows, prev_feats=[]):

        n, t, c, h, w = curr_feats.size()

        if self.is_first:
            self._initialize_submodules(curr_feats, prev_feats, device=curr_feats.device)
            self.feat_indices = list(range(-1, -t - 1, -1)) \
                                    if self.is_reversed \
                                        else list(range(t))

        outputs = list()
        feat_prop = curr_feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            curr_feat = curr_feats[:, self.feat_indices[i], :, :, :]
            if i > 0:
                flow = flows[:, self.feat_indices[i - 1], :, :, :]
                feat_prop = self.aligner(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([curr_feat, feat_prop, *[it[:, self.feat_indices[i], :, :, :] 
                                                                for it in prev_feats]], dim=C_DIM)
            
            feat_prop = self.fextor(feat_prop)

            outputs.append(feat_prop.clone())

        if self.is_reversed:
            outputs = outputs[::-1]

        return outputs

class Alignment(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, feat, flow):
        return flow_warp(feat, flow)

class ResidualBlocksWithInputConv(BaseModule):

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
        )

    def forward(self, feat):
        return self.main(feat)
