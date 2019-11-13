# -*- coding: UTF-8 -*-
# __init__.py
# @author wanger
# @description 
# @created 2019-10-28T12:47:45.545Z+08:00
# @last-modified 2019-11-13T11:48:32.612Z+08:00
#

from .vgg import VGGFeatures
from .resnet import resnet18
from .resnet import resnet34
from .resnet import resnet50
from .resnet import resnet101
from .resnet import resnet152
from .resnet import resnext101_32x8d
from .resnet import resnext50_32x4d

from .resnet_diltated import resnet_dilated_18
from .resnet_diltated import resnet_dilated_34
from .resnet_diltated import resnet_dilated_50
from .resnet_diltated import resnet_dilated_101
from .resnet_diltated import resnet_dilated_152

__all__ = ['VGGFeatures',
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext101_32x8d",
            "resnext50_32x4d",
            "resnet_dilated_18",
            "resnet_dilated_34",
            "resnet_dilated_50",
            "resnet_dilated_101",
            "resnet_dilated_152"]