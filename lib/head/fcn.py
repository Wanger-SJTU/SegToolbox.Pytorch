# -*- coding: UTF-8 -*-
# fcn.py
# @author wanger
# @description 
# @created 2019-10-28T15:52:24.054Z+08:00
# @last-modified 2019-11-14T01:05:26.640Z+08:00
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN8s(nn.Module):
    def __init__(self, cfg):
        super(FCN8s, self).__init__()
        #pool5
        self.fc = nn.Sequential(nn.Conv2d(cfg.MODEL.INCHANNEL[-1], 4096, 7, padding=3),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout2d(),
                                  nn.Conv2d(4096, 4096, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout2d())

        self.score_fr = nn.Conv2d(4096, cfg.MODEL.NUM_CLASS, 1)
        #pool3
        self.score_pool3 = nn.Conv2d(cfg.MODEL.INCHANNEL[-3], cfg.MODEL.NUM_CLASS, 1)
        #pool4
        self.score_pool4 = nn.Conv2d(cfg.MODEL.INCHANNEL[-2], cfg.MODEL.NUM_CLASS, 1)
        
        upscores = []
        for factor in cfg.MODEL.SCALES:
            upscores.append(nn.Upsample(scale_factor=factor, 
                            mode=cfg.MODEL.MODE, align_corners=True))
        self.upscores = nn.ModuleList(upscores)
        
    def forward(self, *args):
        *_,f3,f4,f5 = args # 1/8 1/16 1/32
        f5 = self.upscores[-1](self.score_fr(self.fc(f5)))
        f4 = self.score_pool4(f4)
        f3 = self.score_pool3(f3)
        return self.upscores[-3](self.upscores[-2](f5+f4)+f3)
        
class FCN16s(nn.Module):
    def __init__(self, cfg):
        super(FCN16s, self).__init__()
        self.scale = cfg.MODEL.SCALES
        self.fc = nn.Sequential(nn.Conv2d(cfg.MODEL.INCHANNEL[-1],
                                           4096, 7, padding=3),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout2d(cfg.MODEL.DROPOUT_RATE),
                                  nn.Conv2d(4096, 4096, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout2d(cfg.MODEL.DROPOUT_RATE))
        self.score_fr    = nn.Conv2d(4096, cfg.MODEL.NUM_CLASSES, 1)
        self.score_pool4 = nn.Conv2d(cfg.MODEL.INCHANNEL[-2], 
                                     cfg.MODEL.NUM_CLASSES, 1)

        #self.upscore2 = nn.ConvTranspose2d( n_class, n_class, 4, stride=2, bias=False)
        # self.upscore16 = nn.ConvTranspose2d(
        #     n_class, n_class, 32, stride=16, bias=False)
        # self.upscore2  = nn.Upsample(scale_factor=2,  mode=cfg.MODEL.MODE, align_corners=True)
        # self.upscore16 = nn.Upsample(scale_factor=16, mode=cfg.MODEL.MODE, align_corners=True)
        upscores = []
        for factor in cfg.MODEL.SCALES:
            upscores.append(nn.Upsample(scale_factor=factor, 
                            mode=cfg.MODEL.MODE, align_corners=True))
        self.upscores = nn.ModuleList(upscores)
        
        
    def forward(self, *args):
        *_, p4,p5 = args #1/16  1/32 if vgg
        feature = self.upscores[-1](self.score_fr(self.fc(p5)))
        score_pool4 = self.score_pool4(p4)
        return self.upscores[-2](feature+score_pool4)
        
class FCN32s(nn.Module):
    def __init__(self, cfg):
        super(FCN32s, self).__init__()
        
        self.fc = nn.Sequential(nn.Conv2d(cfg.MODEL.INCHANNEL[-1],
                                           4096, 7, padding=3),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout2d(cfg.MODEL.DROPOUT_RATE),
                                  nn.Conv2d(4096, 4096, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout2d(cfg.MODEL.DROPOUT_RATE)) 
        self.score_fr = nn.Conv2d(4096, cfg.MODEL.NUM_CLASSES, 1)
        self.upscore = nn.Upsample(scale_factor=cfg.MODEL.SCALES[-1], mode=cfg.MODEL.MODE, align_corners=False)
        
    def forward(self, *args):
        f = self.score_fr(self.fc(args[-1]))
        return self.upscore(f)