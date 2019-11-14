
import pdb

import torch
import numpy as np
from collections import OrderedDict

def load_state_dict(src, target):
    # pdb.set_trace()
    for k,v in src.items():
        if 'bn' in k:
            continue
        if k in target.state_dict().keys():
            try:
                v = v.numpy()
            except RuntimeError:
                v = v.detach().numpy()
            try:
                target.state_dict()[k].copy_(torch.from_numpy(v))
            except:
                pdb.set_trace()
    set_requires_grad(target, True)
    return target


def load_bn_params(src_model, dst_paras):
    for key,value in dst_paras.items():
    # if 'module.' not in key:
    #   key = 'module.'+ key
        value = value.cpu().numpy()
        src_model.state_dict()[key].copy_(torch.from_numpy(np.array(value)).cuda()) 

def get_adabn_params(model):
    new_dict = OrderedDict()
    for name, value in model.state_dict().items():
        if 'running_mean' in name or 'running_var' in name:
            new_dict[name] =  value
    return new_dict

def get_bn_params(model):
    new_dict = OrderedDict()
    for name, value in model.state_dict().items():
        if 'bn' in name:
            new_dict[name] = value
    return new_dict

def reset_adabn_params(model):
    new_dict = OrderedDict()
    for name, value in model.state_dict().items():
        if 'running_mean' in name:
            new_dict[name] = 0
        if 'running_var' in name:
            new_dict[name] =  1

    for key,value in new_dict.items():
        model.state_dict()[key].copy_(torch.from_numpy(np.array(value)))

    
def reset_bn_params(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.reset_parameters()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad