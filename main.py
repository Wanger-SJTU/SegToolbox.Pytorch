#
# main.py
# @author bulbasaur
# @description 
# @created 2019-10-29T20:11:52.502Z+08:00
# @last-modified 2019-11-06T13:55:47.677Z+08:00
#

import os
import math

import numpy as np
import torch
import tqdm
import torchvision as tv

from torch.optim import Adam

from data import getDataset
from lib import createSegModel
from lib.core.config import config
from lib.core.options import Options
from lib.core.config import cfg_from_file
from lib.utils.criteria import CrossEntropyLoss2d

from lib.utils import transforms
from lib.utils import reMaskLabel
from lib.utils.vis import Visualizer
from lib.utils.colorize import index2rgb
from lib.utils.metrics import label_accuracy_score
from torch.utils.tensorboard import SummaryWriter

cfg = config
cfg_from_file("./configs/voc_vgg_fcn.yaml")
opts = Options()
opts = opts.parse()

cfg.TRAIN.DROPOUT_RATE = opts.ratio

SEED = 1123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

transforms = tv.transforms.Compose([
    transforms.RandomCropPad(cfg.TRAIN.CROP_SIZE),
    transforms.ToTensor()
])

visualizer = Visualizer(envs=cfg.MODEL.MODEL_NAME+"_ratio_"+str(opts.ratio))

tensorboard_dir = os.path.join(
    cfg.TENSORBOARD.DIR,
    cfg.MODEL.MODEL_NAME,
    "ratio_"+str(opts.ratio)
)
checkpoint_dir = os.path.join(
   cfg.CHECKPOINT.DIR,
   cfg.MODEL.MODEL_NAME,
   "ratio_"+str(opts.ratio)
)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

writer = SummaryWriter(log_dir=tensorboard_dir)
model = createSegModel(cfg).cuda()

optim = Adam(model.parameters(), cfg.SOLVER.BASE_LR)

train_set = getDataset(cfg, 'train', transforms=transforms)

train_loader = torch.utils.data.dataloader.DataLoader(train_set, 
    batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE)

val_set = getDataset(cfg, 'val', transforms=transforms)

val_loader = torch.utils.data.dataloader.DataLoader(val_set, 
    batch_size=cfg.TEST.BATCH_SIZE, shuffle=cfg.TEST.SHUFFLE)

#training 
def train():
    iteration = 0 
    epoches =math.ceil(cfg.SOLVER.MAX_ITER/len(train_loader))
    
    for epoch in tqdm.tqdm(range(epoches), total=epoches, ncols=80):
        loader = tqdm.tqdm(train_loader, total=len(train_loader), 
                            ncols=80, leave=False)
        for item in loader:
            iteration += 1
            
            img = item[0].cuda()
            new_lbl = reMaskLabel(item[1].numpy().copy(), opts.ratio, cfg.DATASET.IGNOREIDX)
            new_lbl = torch.from_numpy(new_lbl).long().cuda()
            
            probs = model(img)
            optim.zero_grad()
            loss = CrossEntropyLoss2d(probs, new_lbl, 21)
            loss.backward()
            optim.step()

            lbl_pred = probs.data.max(1)[1].cpu().numpy()
            lbl_true = item[1].data.cpu().numpy()
            lbl_new  = new_lbl.data.cpu().numpy()
            src_img  = img.data.cpu().numpy()
            
            metrics=[]
            # for lt, lp in zip(lbl_true, lbl_pred):
            #     # (acc, acc_cls, mean_iu, fwavacc)
            res = label_accuracy_score(lbl_true, lbl_pred, n_class=9)
            metrics.append(res)
            metrics = np.mean(metrics, axis=0)
            writer.add_scalar("train/loss",  loss.data.item(), iteration)
            writer.add_scalar("train/acc",     metrics[0], iteration)
            writer.add_scalar("train/acc_cls", metrics[1], iteration)
            writer.add_scalar("train/mean_iu", metrics[2], iteration)
            writer.add_scalar("train/fwavacc", metrics[3], iteration)
            loader.set_description(desc="[{}:{}] loss: {:.2f} iou:{:.2f}".format(
                epoch,iteration, loss.data.item(), metrics[2]))
            if iteration % 50 == 0:
                data = {'src_img':src_img[0], 
                        'src_target':index2rgb(lbl_true[0], cfg.DATASET.NAME), 
                        'src_target_new':index2rgb(lbl_new[0], cfg.DATASET.NAME), 
                        'src_pred':index2rgb(lbl_pred[0], cfg.DATASET.NAME)}
                visualizer.show_visuals(data, epoch=epoch, step=iteration)
            if iteration % cfg.TRAIN.EVAL_PERIOD == 0:
                val(epoch, iteration)
                model.train()
            if iteration > cfg.SOLVER.MAX_ITER:
                break

val_idx = 0
def val(epoch, iteration):
    loader= tqdm.tqdm(val_loader, total=len(val_loader), ncols=80)
    model.eval()
    iou = 0
    global val_idx
    for item in loader:
        val_idx += 1
        img,lbl = item[0].cuda(), item[1].cuda()
        with torch.no_grad():
            probs = model(img)
        loss = CrossEntropyLoss2d(probs, lbl, 21) 
        lbl_pred = probs.data.max(1)[1].cpu().numpy()
        lbl_true = lbl.data.cpu().numpy()
        src_img  = img.data.cpu().numpy()

        if iteration % 30 == 0:
            data = {'val_src_img':src_img[0], 
                    'val_src_target':index2rgb(lbl_true[0], cfg.DATASET.NAME), 
                    'val_src_pred':index2rgb(lbl_pred[0], cfg.DATASET.NAME)}
            visualizer.show_visuals(data, epoch=0, step=val_idx)
            
        metrics=[]
        for lt, lp in zip(lbl_true, lbl_pred):
            # (acc, acc_cls, mean_iu, fwavacc)
            res = label_accuracy_score( [lt], [lp], n_class=9)
            metrics.append(res)
        metrics = np.nanmean(metrics, axis=0)
        writer.add_scalar("val/loss",    loss.data.item(), val_idx)
        writer.add_scalar("val/acc",     metrics[0],       val_idx)
        writer.add_scalar("val/acc_cls", metrics[1],       val_idx)
        writer.add_scalar("val/mean_iu", metrics[2],       val_idx)
        writer.add_scalar("val/fwavacc", metrics[3],       val_idx)
        loader.set_description(desc="[val:{}] loss: {:.2f} iou:{:.2f}".format(
            val_idx, loss.data.item(), metrics[2]))
        iou += metrics[2]
        
    filename = 'epoch_{0}_iteration_{1}.pth'.format(
                    str(epoch),str(iteration))
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch'    : epoch,
        'iteration': iteration,
        'optim'    : optim.state_dict(),
        'model_state_dict': model.state_dict(),
        'iou' : iou/len(val_loader)
    }, path)
      

if __name__ == "__main__":
    train()