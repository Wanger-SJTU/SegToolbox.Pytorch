# -*- coding: utf-8 -*-
# criteria.py
# @author wanger
# @description 
# @license      MIT License
#    
#    Copyright (c) 2019 wanger
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#



import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

__all__ = ['SegmentationLosses','CrossEntropyLoss2d' ]

class CrossEntropyLoss2d(nn.NLLLoss):
    def __init__(self, weight, ignore_index):
        super(CrossEntropyLoss2d, self).__init__(weight=weight, ignore_index=ignore_index)
    
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        return super(CrossEntropyLoss2d,self).forward(pred, target)


class SegmentationLosses(CrossEntropyLoss2d):
    """2D Cross Entropy Loss with Auxilary Loss
        args:
            se_loss: added segmentation loss
            se_weight: weight of se_loss
            nclass: num of classes
            aux: use auxiliary loss or not
            aux_weight: weigth of aux_loss
            weight: class weight
            size_average: return mean value of loss
            ignore_index
    """
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 reduction='mean', ignore_index=-1):
        '''``'none'`` | ``'mean'`` | ``'sum'``'''
        super(SegmentationLosses, self).__init__(weight, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight, reduction=reduction) 

    def forward(self, *inputs):
        func = lambda x: [y for l in x for y in func(l)] \
            if isinstance(x, Iterable) and \
                not isinstance(x, torch.Tensor) else [x]
        inputs = func(inputs)
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif self.se_loss:# ??
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2, {"loss1":loss1, 'loss2':loss2}
        elif self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2, {"seg_loss":loss1, 'cls_loss':loss2}
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3, \
                     {"loss1":loss1, 'loss2':loss2, 'loss3':loss3}

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, nclass).cuda().long()
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect.long()

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

# def CrossEntropyLoss2d(outputs, targets, ignore_index=255):
#     loss = nn.NLLLoss(ignore_index=ignore_index)
#     return loss(F.log_softmax(outputs, dim=1), targets)

def torch_onehot(predicted, num_classes, ignore_index=255):
    if predicted.max() == ignore_index:
        one_hot = torch.eye(ignore_index+1)[predicted.long()]
        one_hot = one_hot.permute(0, 3, 1, 2)
        return one_hot[:,:num_classes,:,:].cuda()
    else:
        onehot=torch.eye(int(num_classes))[predicted.long()]
        onehot = onehot.permute(0, 3, 1, 2)
        return onehot.cuda()

