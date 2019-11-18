# -*- coding: utf-8 -*-
# get_lr_value.py
# @author wanger
# @description 
# @license      MIT License
# https://zhuanlan.zhihu.com/p/31424275
# https://arxiv.org/abs/1506.01186

import os
import math
import torch
import tqdm
import random
import numpy as np
import torchvision as tv

from data import index2rgb
from data import getDataset
from lib import createSegModel
from lib.core.config import config
from lib.core.config import log_cfg
from lib.core.options import Options
from lib.core.config import cfg_from_file
from lib.utils.optims import getOptimizer
from lib.utils.optims import adjustLearningRate
from lib.utils.criteria import CrossEntropyLoss2d
from lib.utils.criteria import SegmentationLosses

from lib.utils import transforms
from lib.utils import reMaskLabel
from lib.utils.vis import Visualizer 
from lib.utils.misc import MySummaryWriter
from lib.utils.metrics import label_accuracy_score

# configs initialization
opts = Options()
opts = opts.parse()

cfg = config
cfg_from_file(opts.config)

cfg.TRAIN.DROPOUT_RATE = opts.ratio
cfg.MODEL.PRETRAIN = opts.pretrain
cfg.TENSORBOARD.HIST = opts.hist
assert len(cfg.EXPERIMENT_NAME) > 0

if cfg.MODEL.PRETRAIN:
    cfg.EXPERIMENT_NAME += "/get_lr/pretrain"
else:
    cfg.EXPERIMENT_NAME += "/get_lr/scratch" 
log_cfg() # record training hyper-parameters

tensorboard_dir = os.path.join(
    cfg.TENSORBOARD.DIR,
    cfg.EXPERIMENT_NAME,
    "ratio_"+str(opts.ratio))

checkpoint_dir = os.path.join(
   cfg.CHECKPOINT.DIR,
   cfg.EXPERIMENT_NAME,
   "ratio_"+str(opts.ratio)
)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

writer = MySummaryWriter(log_dir=tensorboard_dir)

# for model reproducible
SEED = 1123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# for speed up training
torch.backends.cudnn.benchmark = True 

# data transformation
val_transforms = tv.transforms.Compose([
    transforms.ValPadding(cfg.MODEL.FACTOR, cfg.DATASET.IGNOREIDX),
    transforms.ValToTensor()
])

train_transforms = tv.transforms.Compose([
    transforms.RandomResize((1, 1.5)),
    transforms.RandomCropPad(cfg.TRAIN.CROP_SIZE, cfg.DATASET.IGNOREIDX),
    # transforms.RandomGaussBlur(),
    transforms.ToTensor()
])

# prepare dataset and model 
model = createSegModel(cfg, writer).cuda()
# print(model)
# writer.add_graph(model=model, input_to_model=, verbose=True)
optim = getOptimizer(cfg, model)
model = model.cuda()
train_set = getDataset(cfg, 'train', transforms=train_transforms)
train_loader = torch.utils.data.dataloader.DataLoader(train_set, 
    batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE, 
    **cfg.DATALOADER)

Seg_loss = SegmentationLosses(nclass=cfg.MODEL.NUM_CLASSES, 
            ignore_index=cfg.DATASET.IGNOREIDX, **cfg.TRAIN.LOSS)

val_set = getDataset(cfg, 'val', transforms=val_transforms)
val_loader = torch.utils.data.dataloader.DataLoader(val_set, 
    batch_size=cfg.TEST.BATCH_SIZE, shuffle=cfg.TEST.SHUFFLE, 
    **cfg.DATALOADER)

def train():
    lr_mult = (1 / 1e-5) ** (1 / 100)
    lr, best_loss = 1e-5, float('inf')
    epoches = math.ceil(100/len(train_loader))
    for epoch in tqdm.tqdm(range(epoches), total=epoches, ncols=80):
        loader = tqdm.tqdm(train_loader, total=len(train_loader), 
                            ncols=80, leave=False)
        for idx,item in enumerate(loader): 
            img = item[0].cuda()
            lbl = item[1].cuda()
            adjustLearningRate(optim, lr)
            outputs = model(img)
            probs,*others = [outputs,] if isinstance(outputs, torch.Tensor) else outputs
            optim.zero_grad()
            loss,*loss_detail = Seg_loss(probs, others, lbl)# pred, lbl            loss.backward()
            optim.step()
            
            lbl_pred = probs.data.max(1)[1].cpu().detach().numpy()
            lbl_true = lbl.data.cpu().numpy()
            src_img  = img.data.cpu().numpy()
            
            metrics = label_accuracy_score(lbl_true, lbl_pred, cfg.MODEL.NUM_CLASSES)
            
            best_loss = min(best_loss, loss.data.detach().item())
            loss_dic = {"loss": loss.data.detach().item(),
                         "lr":lr}
            if not loss_detail:
                for k,v in loss_detail:
                    loss_dic.update({"loss_"+k:v.data.detach().item()}) 
            writer.add_scalars("loss_lr", loss_dic)
            lr *= lr_mult
            loader.set_description(desc="[{}:{}] loss: {:.2f} ".format(
                epoch, writer.global_step, loss.data.item())) 
            writer.addStep()

if __name__ == "__main__":
    train()
    writer.close()