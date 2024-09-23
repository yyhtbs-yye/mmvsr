# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from src.registry import MODELS

MODELS.register_module('Deconv', module=nn.ConvTranspose2d)
# TODO: octave conv
