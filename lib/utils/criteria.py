# -*- coding: UTF-8 -*-
# criteria.py
# @author wanger
# @description 
# @created 2019-10-31T10:22:36.005Z+08:00
# @last-modified 2019-11-05T14:49:22.299Z+08:00
#


import torch
import torch.nn as nn
import torch.nn.functional as F


class PSP_loss(object):
    """docstring for PSP_loss"""
    def __init__(self, ignore_index=-1, alpha=0.4, aux_loss=nn.CrossEntropyLoss()):
        super(PSP_loss, self).__init__()
        class_weight = torch.Tensor([0.1, 1, 1, 1, 1, 1, 1, 1,1]).cuda()
        # self.seg_criterion = nn.NLLLoss(weight=class_weight,ignore_index=255)
        self.seg_criterion = CELoss2d(num_classes=9)
        self.cls_criterion = aux_loss
        self.alpha = 0.6

    def __call__(self, y, y_cls, out, out_cls):
        seg_loss = self.seg_criterion(F.log_softmax(out, dim=1), y)
        # seg_loss = self.seg_criterion(out, y)
        cls_loss = self.cls_criterion(out_cls, y_cls)
        return seg_loss + self.alpha * cls_loss

class PSP_loss_Multi(object):
    """docstring for PSP_loss"""
    def __init__(self, ignore_index=-1, alpha=0.4):
        super(PSP_loss_Multi, self).__init__()
        class_weight = torch.Tensor([0.1, 1, 1, 1, 1, 1, 1, 1,1]).cuda()
        self.seg_criterion = nn.NLLLoss(weight=class_weight,ignore_index=255)
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.alpha = 0.6

    def __call__(self, y, y_cls, out, out_cls):
        seg_loss = self.seg_criterion(F.log_softmax(out, dim=1), y)
        # seg_loss = self.seg_criterion(out, y)
        cls_loss = self.cls_criterion(out_cls, y_cls)
        return seg_loss + self.alpha * cls_loss

class CELoss2d(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(CELoss2d, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        outputs = F.log_softmax(outputs, dim=1)
        ignored = targets != self.ignore_index
        targets = torch_onehot(targets, self.num_classes, self.ignore_index)
        spatial_loss = targets*outputs
        spatial_loss = spatial_loss.sum(dim=1)
        spatial_loss = spatial_loss * ignored.float()
        total = outputs.size()[0]*outputs.size()[2]*outputs.size()[3]#*outputs.size()[1]  
        return -1*spatial_loss.sum()/total 

def CrossEntropyLoss2d(outputs, targets, ignore_index=255):
    loss = nn.NLLLoss(ignore_index=ignore_index)
    return loss(F.log_softmax(outputs, dim=1), targets)

def torch_onehot(predicted, num_classes, ignore_index=255):
    if predicted.max() == ignore_index:
        one_hot = torch.eye(ignore_index+1)[predicted.long()]
        one_hot = one_hot.permute(0, 3, 1, 2)
        return one_hot[:,:num_classes,:,:].cuda()
    else:
        onehot=torch.eye(int(num_classes))[predicted.long()]
        onehot = onehot.permute(0, 3, 1, 2)
        return onehot.cuda()
