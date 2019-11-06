# -*- coding: UTF-8 -*-
# vgg.py
# @author wanger
# @description 
# @created 2019-10-28T10:50:06.128Z+08:00
# @last-modified 2019-11-04T21:46:15.538Z+08:00
#

import torch.nn as nn

from torchvision.models import vgg16_bn
from torchvision.models import vgg16

class VGGFeatures(nn.Module):
    def __init__(self, use_bn=True):
        super(VGGFeatures, self).__init__()
        
        if use_bn:
            features = list(vgg16_bn(pretrained=True).features)
        else:
            features = list(vgg16(pretrained=True).features)
        stages = []
        srt = 0
        for end in range(len(features)):
            if isinstance(features[end], nn.MaxPool2d):
                stages.append(features[srt:end])
                srt = end
        self.stages1 = nn.Sequential(*stages[0])
        self.stages2 = nn.Sequential(*stages[1])
        self.stages3 = nn.Sequential(*stages[2])
        self.stages4 = nn.Sequential(*stages[3])
        self.stages5 = nn.Sequential(*stages[4])
    
    def forward(self, img):
        x1 = self.stages1(img)
        x2 = self.stages2(x1)
        x3 = self.stages3(x2)
        x4 = self.stages4(x3)
        x5 = self.stages5(x4)
        return x1,x2,x3,x4,x5
