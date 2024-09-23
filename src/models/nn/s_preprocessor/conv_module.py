from mmengine.model import BaseModule

from src.registry import MODELS
from src.models.nn.ts_backbones.conv_module import ResidualBlocksWithInputConv

@MODELS.register_module()
class BasicVSRPlusPlusSpatial(BaseModule):

    def __init__(self, in_channels=3, mid_channels=64):
        super().__init__()

        self.feat_extract = ResidualBlocksWithInputConv(in_channels, mid_channels, 5)

    def forward(self, x):
        n, t, c, h, w = x.size()
        return self.feat_extract(x.view(-1, c, h, w)).view(n, t, -1, h, w)
