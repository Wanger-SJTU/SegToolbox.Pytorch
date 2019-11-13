
# for future

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