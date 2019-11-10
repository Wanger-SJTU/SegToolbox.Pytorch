# -*- coding: UTF-8 -*-
# optims.py
# @author wanger
# @description 
# @created 2019-11-05T15:02:08.902Z+08:00
# @last-modified 2019-11-10T12:36:51.126Z+08:00
#

from torch.optim import Adadelta  
from torch.optim import Adagrad  
from torch.optim import Adam  
from torch.optim import AdamW 
from torch.optim import SparseAdam
from torch.optim import Adamax 
from torch.optim import ASGD 
from torch.optim import SGD 
from torch.optim import Rprop
from torch.optim import RMSprop
# from torch.optim import Optimizer
from torch.optim import LBFGS
from torch.optim import lr_scheduler

optims = {
    "adadelta":Adadelta,
    "adagrad":Adagrad,
    "adam":Adam,
    "adamw":AdamW,
    "sparseadam":SparseAdam,
    "adamax":Adamax,
    "asgd":ASGD,
    "sgd":SGD,
   # "rprop":Rprop,
    "rmsprop":RMSprop
}

def getOptimizer(cfg, model):
    if cfg.SOLVER.OPTIM.lower() not in optims.keys():
        raise ValueError("optim {} not in current selections".format(cfg.SOLVER.OPTIM))
    if cfg.SOLVER.OPTIM.lower() == 'adam':
        return optims['adam'](model.parameters(), cfg.SOLVER.BASE_LR, 
                                betas=(cfg.SOLVER.MOMENTUM, 0.999), eps=1e-8,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY, amsgrad=False)

    return optims[cfg.SOLVER.OPTIM.lower()](model.parameters(), 
                                lr=cfg.SOLVER.BASE_LR, 
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)

def get_lr(cfg, iter, epoch):
    # 'step', 'steps_with_lrs', 'steps_with_relative_lrs', 'steps_with_decay, 'fixed''
    if cfg.SOLVER.LR_POLICY == "fixed":
        return cfg.SOLVER.BASE_LR
    elif cfg.SOLVER.LR_POLICY == "steps_with_lrs":
        raise NotImplementedError
    elif cfg.SOLVER.LR_POLICY == "steps_with_relative_lrs":
        raise NotImplementedError
    elif cfg.SOLVER.LR_POLICY == "step":
        raise NotImplementedError
    elif cfg.SOLVER.LR_POLICY == "exponential":
        raise NotImplementedError
    elif cfg.SOLVER.LR_POLICY == "Polynomial":
        raise NotImplementedError
    
def adjustLearningRate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr