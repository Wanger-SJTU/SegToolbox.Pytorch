# -*- coding: UTF-8 -*-
# __init__.py
# @author wanger
# @description 
# @created 2019-10-28T15:54:57.838Z+08:00
# @last-modified 2019-11-07T22:25:12.685Z+08:00
#

import torch.nn as nn
from .backbone import VGGFeatures

from .head.fcn import FCN8s
from .head.fcn import FCN16s
from .head.fcn import FCN32s
from .head.fcn import Classifier
from .head.NonLocal import NonlocalGroup
from .head.NonLocal import NonLocalPatch

__all__ = ["createSegModel"]

backbones = {
    "VGG":VGGFeatures,
    "vgg":VGGFeatures
}

heads = {
    'fcn8s':FCN8s,
    'fcn16s':FCN16s,
    'fcn32s':FCN32s,
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