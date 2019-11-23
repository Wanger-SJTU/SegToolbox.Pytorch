
import torch
import torch.nn as nn
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(cfg.MODEL.INCHANNEL[-1], 
                            cfg.MODEL.NUM_CLASSES, bias=False)
    
    def forward(self, *inputs):
        shape = inputs[-1].size()
        feat  = inputs[-1].view(shape[0],shape[1],-1)
        return self.fc(feat.mean(dim=2))


class SegClassifer(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.up == nn.Upsample(scale_factor=cfg.MODEL.FACTOR, mode=cfg.MODEL.MODE)
        self.conv = nn.Conv2d(cfg.MODEL.INCHANNLES[-1], 
                              cfg.MODEL.NUM_CLASSES, 3, 1, 1)

    def forward(self, *inputs):
        return self.conv(self.up(inputs[-1]))