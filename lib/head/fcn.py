# -*- coding: UTF-8 -*-
# fcn.py
# @author wanger
# @description 
# @created 2019-10-28T15:52:24.054Z+08:00
# @last-modified 2019-11-06T21:37:51.578Z+08:00
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        inChannels = sum(cfg.MODEL.INCHANNLES)
        self.conv = nn.Conv2d(inChannels, num_class, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)

class FCNhead(nn.Module):
    def __init__(self, cfg):
        super(FCNhead, self).__init__()
        self.scale = cfg.MODEL.SCALES
        self.mode = cfg.MODEL.MODE 
        convs = []
        for inChannel in cfg.MODEL.INCHANNEL:
            convs.append(nn.Conv2d(inChannel, cfg.MODEL.NUM_CLASSES, 1))
        self.convs = nn.ModuleList(convs)
        
    def forward(self, *args):
        res = None
        for idx, (feature, scale) in enumerate(zip(args[-len(self.scale):], self.scale)):
            tmp = self.convs[idx](F.interpolate(feature, scale_factor=scale, 
                                                mode=self.mode, align_corners=False))
            if res is None:
                res = tmp
            else:
                res += tmp
        return res