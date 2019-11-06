# -*- coding: UTF-8 -*-
# nonlocal.py
# @author wanger
# @description 
# @created 2019-10-28T12:48:12.393Z+08:00
# @last-modified 2019-10-28T15:56:13.913Z+08:00
#

import torch.nn as nn

from torch.nn.functional import softmax

class NonlocalGroup(nn.Module):
    def __init__(self, q_channnels, k_channnels, mid_channels, use_bn=False, softmax=False, drop_out=1):
        super(NonlocalGroup, self).__init__()
        assert 0 < drop_out <= 1
        if use_bn:
            self.q_conv = nn.Sequential(
                nn.Conv2d(q_channels, mid_channels, 1, 1, 0, 0),
                nn.BatchNorm2d())
            self.k_conv = nn.Sequential(
                nn.Conv2d(k_channels, mid_channels, 1, 1, 0, 0),
                nn.BatchNorm2d())
        else:
            self.q_conv = nn.Conv2d(q_channels, mid_channels, 1, 1, 0, 0)
            self.k_conv = nn.Conv2d(k_channels, mid_channels, 1, 1, 0, 0)

        self.softmax = softmax
        if drop_out != 1:
            self.dropout = nn.Dropout2d(drop_out)
        else:
            self.dropout = None
        
        
    def forward(self, q, k):
        pass


class NonLocalPatch(nn.Module):
    pass
        
