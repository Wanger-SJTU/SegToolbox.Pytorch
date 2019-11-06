# -*- coding: UTF-8 -*-
# __init__.py
# @author wanger
# @description 
# @created 2019-10-28T15:54:57.838Z+08:00
# @last-modified 2019-10-31T21:13:31.540Z+08:00
#

import torch.nn as nn

from .backbone import VGGFeatures

from .head.fcn import FCNhead
from .head.fcn import Classifier
from .head.NonLocal import NonlocalGroup
from .head.NonLocal import NonLocalPatch

__all__ = ["createSegModel"]

backbones = {
    "VGG":VGGFeatures,
    "vgg":VGGFeatures
}

heads = {
    'fcnhead':FCNhead,
    'NonlocalGroup':NonlocalGroup,
    'NonLocalPatch':NonLocalPatch
}

class SegModel(nn.Module):
    def __init__(self, cfg):
        super(SegModel, self).__init__()
        self.features = backbones[cfg.MODEL.BACKBONE](cfg)
        self.head = heads[cfg.MODEL.HEAD](cfg)
    
    
    def forward(self, x):
        features = self.features(x) 
        return self.head(*features)


def createSegModel(cfg):
    return SegModel(cfg)