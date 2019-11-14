# -*- coding: UTF-8 -*-
# __init__.py
# @author wanger
# @description 
# @created 2019-10-28T12:47:45.545Z+08:00
# @last-modified 2019-11-14
#

from .vgg import VGGFeatures
from .resnet import resnet

from .resnet_diltated import resnet_dilated
from .resnet_diltated import resnet_dilated
from .resnet_diltated import resnet_dilated
from .resnet_diltated import resnet_dilated
from .resnet_diltated import resnet_dilated

__all__ = ['backbones']

backbones = {
    "VGG":VGGFeatures,
    "vgg":VGGFeatures,
    "resnet18"  :resnet,
    "resnet34"  :resnet,
    "resnet50"  :resnet,
    "resnet101" :resnet,
    "resnet152" :resnet,
    'wide_resnet50_2' :resnet,
    'wide_resnet101_2':resnet,
    "resnext101_32x8d"  :resnet,
    "resnext50_32x4d"   :resnet,    
    "resnet_dilated_18" :resnet_dilated,
    "resnet_dilated_34" :resnet_dilated,
    "resnet_dilated_50" :resnet_dilated,
    "resnet_dilated_101":resnet_dilated,
    "resnet_dilated_152":resnet_dilated,
}