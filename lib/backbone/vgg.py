# -*- coding: UTF-8 -*-
# vgg.py
# @author wanger
# @description 
# @created 2019-10-28T10:50:06.128Z+08:00
# @last-modified 2019-11-11T03:16:46.782Z+08:00
#

import torch.nn as nn

from torchvision.models import vgg16_bn
from torchvision.models import vgg16

class VGGFeatures(nn.Module):
    def __init__(self, cfg, use_bn=True):
        super(VGGFeatures, self).__init__()
        
        if use_bn:
            features = list(vgg16_bn(pretrained=cfg.MODEL.PRETRAIN).features)
        else:
            features = list(vgg16(pretrained=cfg.MODEL.PRETRAIN).features)
            
        stages = []
        srt = 0
        for end in range(len(features)):
            if isinstance(features[end], nn.MaxPool2d):
                stages.append(features[srt:end+1])
                srt = end+1
        self.stages1 = nn.Sequential(*stages[0])
        self.stages2 = nn.Sequential(*stages[1])
        self.stages3 = nn.Sequential(*stages[2])
        self.stages4 = nn.Sequential(*stages[3])
        self.stages5 = nn.Sequential(*stages[4])
        if not cfg.MODEL.PRETRAIN:
            self._initialize_weights()
            
    def forward(self, img):
        x1 = self.stages1(img)
        x2 = self.stages2(x1)
        x3 = self.stages3(x2)
        x4 = self.stages4(x3)
        x5 = self.stages5(x4)
        return x1,x2,x3,x4,x5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)