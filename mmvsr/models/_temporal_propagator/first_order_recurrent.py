from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.models.archs import ResidualBlockNoBN
from mmvsr.models.utils import flow_warp, make_layer
from mmvsr.registry import MODELS

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
        self.aligner_def = aligner_def if aligner_def is not None else FlowWarp

        # Placeholders for dynamically created modules
        self.fextor = None
        self.aligner = None

        self.is_first = True


    def _initialize_submodules(self, prev_feats, device):

        # This is to mimic the Dense Connection
        input_channels = (2 + len(prev_feats)) * self.mid_channels
        
        if self.fextor is None:
            self.fextor = self.fextor_def(input_channels, self.mid_channels, self.num_blocks).to(device)

        if self.aligner is None:
            self.aligner = self.aligner_def().to(device)

        self.is_first = False

    def forward(self, feats, flows, prev_feats=[], feat_indices=None):

        if self.is_first:
            self._initialize_submodules(prev_feats, device=feats.device)

        n, t, c, h, w = feats.size()

        feat_indices = list(range(t - 1, -1, -1)) if self.is_reversed else list(range(t))
        flow_indices = list(range(t - 2, -1, -1)) if self.is_reversed else list(range(t - 1))

        outputs = []
        feat_prop = feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            curr_feat = feats[:, feat_indices[i], :, :, :]
            if i > 0:  # no warping required for the first timestep [0]
                flow = flows[:, flow_indices[i - 1], :, :, :]
                feat_prop = self.aligner(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([curr_feat, feat_prop, *[it[:, feat_indices[i], :, :, :] 
                                                                for it in prev_feats]], dim=1)
            feat_prop = self.fextor(feat_prop)

            outputs.append(feat_prop)

        return torch.stack(outputs, dim=1)

class FlowWarp(BaseModule):
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
