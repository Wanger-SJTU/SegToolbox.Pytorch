# -*- coding: UTF-8 -*-
# __init__.py
# @author wanger
# @description 
# @created 2019-10-28T15:54:57.838Z+08:00
# @last-modified 2019-11-13T11:48:53.125Z+08:00
#

import torch.nn as nn

from .head import *
from .backbone import *
from .utils.misc import variable_summaries

__all__ = ["createSegModel"]

backbones = {
    "VGG":VGGFeatures,
    "vgg":VGGFeatures
}

heads = {
    'fcn8s':FCN8s,
    'fcn16s':FCN16s,
    'fcn32s':FCN32s
}

class SegModel(nn.Module):
    def __init__(self, cfg, writer=None):
        super(SegModel, self).__init__()
        self.features = backbones[cfg.MODEL.BACKBONE](cfg)
        self.head = heads[cfg.MODEL.HEAD](cfg)
        self.opts = cfg
        self.writer = writer
    def forward(self, x):
        features = self.features(x)
        if self.training and self.opts.TENSORBOARD.HIST and \
            self.writer is not None:
            variable_summaries(self.writer, *features)
        return self.head(*features)

def variable_summaries(writer, *features):
    for i,f in enumerate(features):
        writer.variable_summaries("hist", "featuremap"+str(i), f)

def createSegModel(cfg, writer=None):
    return SegModel(cfg, writer)