#
# cls.py
# @author bulbasaur
# @description 
# @created 2019-10-29T20:11:52.502Z+08:00
# @last-modified 2019-11-14
#

import os
import math
import torch
import tqdm
import random
import numpy as np
import torchvision as tv

from lib import buildModel
from data import index2rgb
from data import getDataset
from lib.core.config import config
from lib.core.config import log_cfg
from lib.core.options import Options
from lib.core.config import cfg_from_file
from lib.utils.optims import getOptimizer
from lib.utils.criteria import SegBCELoss
from lib.utils.optims import adjustLearningRate

from lib.utils import transforms
from lib.utils import reMaskLabel
from lib.utils.vis import Visualizer 
from lib.utils.metrics import cls_score
from lib.utils.misc import MySummaryWriter

# configs initialization
opts = Options()
opts = opts.parse()

cfg = config
cfg_from_file(opts.config)
cfg.MODEL.PRETRAIN = opts.pretrain

cfg.EXPERIMENT_NAME = "cls/"+cfg.EXPERIMENT_NAME
log_cfg() # record training hyper-parameters

assert len(cfg.EXPERIMENT_NAME) > 0

tensorboard_dir = os.path.join(
    cfg.TENSORBOARD.DIR,
    cfg.EXPERIMENT_NAME,
    "ratio_"+str(opts.ratio))

checkpoint_dir = os.path.join(
   cfg.CHECKPOINT.DIR,
   cfg.EXPERIMENT_NAME,
   "ratio_"+str(opts.ratio))

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
model = buildModel(cfg, writer).cuda()
# print(model)
# writer.add_graph(model=model, input_to_model=, verbose=True)
optim = getOptimizer(cfg, model)
model = model.cuda()
train_set = getDataset(cfg, 'train', transforms=train_transforms)
train_loader = torch.utils.data.dataloader.DataLoader(train_set, 
    batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE, 
    **cfg.DATALOADER)

Cls_loss = SegBCELoss(nclass=cfg.MODEL.NUM_CLASSES, 
                    ignore_index=cfg.DATASET.IGNOREIDX)

val_set = getDataset(cfg, 'val', transforms=val_transforms)
val_loader = torch.utils.data.dataloader.DataLoader(val_set, 
    batch_size=cfg.TEST.BATCH_SIZE, shuffle=cfg.TEST.SHUFFLE, 
    **cfg.DATALOADER)

def train():
    adjustLearningRate(optim, cfg.SOLVER.BASE_LR)
    epoches = math.ceil(cfg.SOLVER.MAX_ITER/len(train_loader))
    for epoch in tqdm.tqdm(range(epoches), total=epoches, ncols=80):
        loader = tqdm.tqdm(train_loader, total=len(train_loader), 
                            ncols=80, leave=False)
        for idx,item in enumerate(loader):
            img,lbl = item[0].cuda(), item[1].cuda()
            probs = model(img)
            optim.zero_grad()
            loss = Cls_loss(probs, lbl)
            loss.backward()
            optim.step()
            
            lbl_pred = probs.data.cpu().detach().numpy()
            lbl_true = lbl.data.cpu().detach().numpy()
            
            metrics = cls_score(lbl_pred, lbl_true, cfg.MODEL.NUM_CLASSES)

            for k, v in metrics.items():
                writer.add_scalar("train/"+k, v) 
            writer.add_scalar("train/loss",loss.data.detach().item()) 
            loader.set_description(desc="[{}:{}] loss: {:.2f} acc:{:.2f}".format(
                epoch, writer.global_step, loss.data.item(), metrics['acc']))
            
            if writer.global_step % cfg.TRAIN.EVAL_PERIOD == 0:
                val(epoch, save=True)
                model.train() 
            if writer.global_step > cfg.SOLVER.MAX_ITER or best_acc > 0.98:
                break
            writer.addStep()

val_idx  = 0
best_acc = 0
def val(epoch,save=False):
    loader= tqdm.tqdm(val_loader, total=len(val_loader), ncols=80, leave=False)
    model.eval()
    acc = 0
    global val_idx, best_acc
    for item in loader:
        val_idx += 1
        img, lbl = item[0].cuda(), item[1].cuda()
        with torch.no_grad():
            outputs = model(img)
            loss = Cls_loss(outputs, lbl)
        
        lbl_pred = outputs.data.cpu().numpy()
        lbl_true = lbl.data.cpu().numpy()
        
        metrics = cls_score(lbl_pred, lbl_true, cfg.MODEL.NUM_CLASSES)
        
        for k,v in metrics.items():
            writer.add_scalar("val/"+k, v,  val_idx) 
        writer.add_scalar("val/loss",loss.data.detach().item(), val_idx)
        
        loader.set_description(desc="[val:{}] loss: {:.2f} acc:{:.2f}".format(
            val_idx, loss.data.item(), metrics['acc']))
        acc += metrics['acc']
        
    if cfg.CHECKPOINT.CHECKPOINT_MODEL:
        filename = 'epoch_{0}_iteration_{1}.pth'.format(
                        str(epoch),str(writer.global_step))
        path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch'    : epoch,
            'iteration': writer.global_step,
            'optim'    : optim.state_dict(),
            'model_state_dict': model.state_dict(),
            'acc' : acc/len(val_loader)
        }, path)

    if save:
        filename = 'lastest.pth'.format(
                        str(epoch),str(writer.global_step))
        path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch'    : epoch,
            'iteration': writer.global_step,
            'optim'    : optim.state_dict(),
            'model_state_dict': model.state_dict(),
            'acc' : acc/len(val_loader)
        }, path)

    if acc/len(val_loader) > best_acc:
        best_acc = acc/len(val_loader)
        filename = 'best_model.pth'
        path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch'    : epoch,
            'iteration': writer.global_step,
            'optim'    : optim.state_dict(),
            'model_state_dict': model.state_dict(),
            'acc'      : acc/len(val_loader)
        }, path)  
    
    path = os.path.join(checkpoint_dir, "acc_res.txt")
    with open(path, 'a+', encoding='utf8') as f:
        f.write("{}\t:{} \t  acc:{} \n".format(epoch, writer.global_step, acc/len(val_loader))) 


def load_check_point():
    pass


if __name__ == "__main__":
    # eval_data()
    train()
    # val()
    writer.close()