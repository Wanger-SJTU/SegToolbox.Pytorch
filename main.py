#
# main.py
# @author bulbasaur
# @description 
# @created 2019-10-29T20:11:52.502Z+08:00
# @last-modified 2019-11-13T13:39:14.806Z+08:00
#

import os
import math
import torch
import tqdm
import numpy as np
import torchvision as tv

from data import getDataset
from lib import createSegModel
from lib.core.config import config
from lib.core.options import Options
from lib.core.config import cfg_from_file
from lib.utils.optims import getOptimizer
from lib.utils.optims import adjustLearningRate
from lib.utils.criteria import CrossEntropyLoss2d

from lib.utils import transforms
from lib.utils import reMaskLabel
from lib.utils.vis import Visualizer
from lib.utils.colorize import index2rgb
from lib.utils.misc import MySummaryWriter
from lib.utils.metrics import label_accuracy_score

# configs initialization
opts = Options()
opts = opts.parse()

cfg = config
cfg_from_file(opts.config)
cfg.TRAIN.DROPOUT_RATE = opts.ratio

# visualization
visualizer = Visualizer(envs=cfg.MODEL.MODEL_NAME+"_ratio_"+str(opts.ratio))

tensorboard_dir = os.path.join(
    cfg.TENSORBOARD.DIR,
    cfg.MODEL.MODEL_NAME,#+"_eval",
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

writer = MySummaryWriter(log_dir=tensorboard_dir)

# for model reproducible
SEED = 1123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# for speed up training
torch.backends.cudnn.benchmark = True 

# data transformation
val_transforms = tv.transforms.Compose([
    transforms.ValPadding(cfg.MODEL.FACTOR, cfg.DATASET.IGNOREIDX),
    transforms.ValToTensor()
])

train_transforms = tv.transforms.Compose([
    transforms.RandomCropPad(cfg.TRAIN.CROP_SIZE, cfg.DATASET.IGNOREIDX),
    transforms.ToTensor()
])

# prepare dataset and model 
model = createSegModel(cfg, writer).cuda()
optim = getOptimizer(cfg, model)

train_set = getDataset(cfg, 'train', transforms=train_transforms)
train_loader = torch.utils.data.dataloader.DataLoader(train_set, 
    batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE, 
    **cfg.DATALOADER)

val_set = getDataset(cfg, 'val', transforms=val_transforms)
val_loader = torch.utils.data.dataloader.DataLoader(val_set, 
    batch_size=cfg.TEST.BATCH_SIZE, shuffle=cfg.TEST.SHUFFLE, 
    **cfg.DATALOADER)

#training

def train():
    if cfg.SOLVER.WARMUP.WARMUP_ON:
        warmUpModel()
    adjustLearningRate(optim, cfg.SOLVER.BASE_LR)
    epoches = math.ceil(cfg.SOLVER.MAX_ITER/len(train_loader))
    for epoch in tqdm.tqdm(range(epoches), total=epoches, ncols=80):
        loader = tqdm.tqdm(train_loader, total=len(train_loader), 
                            ncols=80, leave=False)
        for item in loader: 
            writer.addStep()
            img = item[0].cuda()
            new_lbl = reMaskLabel(item[1].numpy().copy(), opts.ratio, cfg.DATASET.IGNOREIDX)
            new_lbl = torch.from_numpy(new_lbl).long().cuda()
            lbl_true = item[1]
            # import pdb;pdb.set_trace()
            probs = model(img)
            optim.zero_grad()
            loss = CrossEntropyLoss2d(probs, new_lbl, cfg.DATASET.IGNOREIDX)
            loss.backward()
            optim.step()
            
            lbl_pred = probs.data.max(1)[1].cpu().detach().numpy()
            lbl_true = lbl_true.data.cpu().numpy()
            lbl_new  = new_lbl.data.cpu().numpy()
            src_img  = img.data.cpu().numpy()
            
            metrics = label_accuracy_score(lbl_true, lbl_pred, cfg.MODEL.NUM_CLASSES)
           
            writer.add_scalar("train/loss",loss.data.detach().item()) 
            writer.add_scalar("train/acc",     metrics[0])
            writer.add_scalar("train/acc_cls", metrics[1])
            writer.add_scalar("train/mean_iu", metrics[2])
            writer.add_scalar("train/fwavacc", metrics[3])
            loader.set_description(desc="[{}:{}] loss: {:.2f} iou:{:.2f}".format(
                epoch, writer.global_step, loss.data.item(), metrics[2]))
            if writer.global_step % 20 == 0:
                data = {'src_img':src_img[0], 
                        'src_target':index2rgb(lbl_true[0], cfg.DATASET.NAME), 
                        'src_target_new':index2rgb(lbl_new[0], cfg.DATASET.NAME), 
                        'src_pred':index2rgb(lbl_pred[0], cfg.DATASET.NAME)}
                visualizer.show_visuals(data, epoch=epoch, step=writer.global_step)
            if writer.global_step % cfg.TRAIN.EVAL_PERIOD == 0:
            #if writer.global_step % 1 == 0:
                val(epoch)
                model.train()
            if writer.global_step > cfg.SOLVER.MAX_ITER:
                break

val_idx = 0
def val(epoch):
    loader= tqdm.tqdm(val_loader, total=len(val_loader), ncols=80, leave=False)
    model.eval()
    iou = 0
    global val_idx
    for item in loader:
        val_idx += 1
        img, lbl = item[0].cuda(), item[1].cuda()
        pos = list(map(int, item[2].numpy()[0])) if len(item) > 2 else None
        with torch.no_grad():
            probs = model(img)
            loss = CrossEntropyLoss2d(probs, lbl, cfg.DATASET.IGNOREIDX) 
        
        lbl_pred = probs.data.max(1)[1].cpu().numpy()
        lbl_true = lbl.data.cpu().numpy()
        src_img  = img.data.cpu().numpy()
        shape = lbl_true.shape
        if pos is not None:
            lbl_pred = lbl_pred[:,pos[0]:shape[1]-pos[1],pos[2]:shape[2]-pos[3]] 
            lbl_true = lbl_true[:,pos[0]:shape[1]-pos[1],pos[2]:shape[2]-pos[3]]
            src_img  = src_img[:,:,pos[0]:shape[1]-pos[1],pos[2]:shape[2]-pos[3]]
        
        if val_idx % 50 == 0:
            data = {'val_src_img':src_img[0], 
                    'val_src_target':index2rgb(lbl_true[0], cfg.DATASET.NAME), 
                    'val_src_pred':index2rgb(lbl_pred[0],   cfg.DATASET.NAME)}
            visualizer.show_visuals(data, epoch=epoch, step=val_idx)
        
        metrics = label_accuracy_score(lbl_true, lbl_pred, cfg.MODEL.NUM_CLASSES)
        
        writer.add_scalar("val/loss",    loss.data.detach().item(), val_idx)
        writer.add_scalar("val/acc",     metrics[0],       val_idx)
        writer.add_scalar("val/acc_cls", metrics[1],       val_idx)
        writer.add_scalar("val/mean_iu", metrics[2],       val_idx)
        writer.add_scalar("val/fwavacc", metrics[3],       val_idx)
        loader.set_description(desc="[val:{}] loss: {:.2f} iou:{:.2f}".format(
            val_idx, loss.data.item(), metrics[2]))
        iou += metrics[2]
        
    if cfg.CHECKPOINT.CHECKPOINT_MODEL:
        filename = 'epoch_{0}_iteration_{1}.pth'.format(
                        str(epoch),str(writer.global_step))
        path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch'    : epoch,
            'iteration': writer.global_step,
            'optim'    : optim.state_dict(),
            'model_state_dict': model.state_dict(),
            'iou' : iou/len(val_loader)
        }, path)
        
    path = os.path.join(checkpoint_dir, "iou_res.txt")
    with open(path, 'a+', encoding='utf8') as f:
        f.write("{}\t:{} \t  iou:{} \n".format(epoch, writer.global_step, iou/len(val_loader))) 

def warmUpModel(): 
    epoches = cfg.SOLVER.WARMUP.WARMUP_EPOCH
    adjustLearningRate(optim, cfg.SOLVER.WARMUP.WARMUP_START_LR)
    for epoch in tqdm.tqdm(range(epoches), total=epoches, ncols=80):
        loader = tqdm.tqdm(train_loader, total=len(train_loader), 
                            ncols=80, leave=False)
        for item in loader: 
            writer.addStep()
            img = item[0].cuda()
            new_lbl = reMaskLabel(item[1].numpy().copy(), opts.ratio, cfg.DATASET.IGNOREIDX)
            new_lbl = torch.from_numpy(new_lbl).long().cuda()
            lbl_true = item[1] 
            probs = model(img)
            optim.zero_grad()
            loss = CrossEntropyLoss2d(probs, new_lbl, cfg.DATASET.IGNOREIDX)
            loss.backward()
            optim.step()
            
            lbl_pred = probs.data.max(1)[1].cpu().detach().numpy()
            lbl_true = lbl_true.data.cpu().numpy()
            lbl_new  = new_lbl.data.cpu().numpy()
            src_img  = img.data.cpu().numpy() 
            metrics = label_accuracy_score(lbl_true, lbl_pred, cfg.MODEL.NUM_CLASSES)
           
            writer.add_scalar("train/loss",loss.data.detach().item()) 
            writer.add_scalar("train/acc",     metrics[0])
            writer.add_scalar("train/acc_cls", metrics[1])
            writer.add_scalar("train/mean_iu", metrics[2])
            writer.add_scalar("train/fwavacc", metrics[3])
            #writer.add_scalar("train/lr",      lr)
            loader.set_description(desc="WARMUP: [{}:{}] loss: {:.2f} iou:{:.2f}".format(
                epoch, writer.global_step, loss.data.item(), metrics[2]))
            if writer.global_step % 20 == 0:
                data = {'src_img':src_img[0], 
                        'src_target':index2rgb(lbl_true[0], cfg.DATASET.NAME), 
                        'src_target_new':index2rgb(lbl_new[0], cfg.DATASET.NAME), 
                        'src_pred':index2rgb(lbl_pred[0], cfg.DATASET.NAME)}
                visualizer.show_visuals(data, epoch=epoch, step=writer.global_step)
            if writer.global_step % cfg.TRAIN.EVAL_PERIOD == 0:
            #if writer.global_step % 1 == 0:
                val(epoch)
                model.train()
            
if __name__ == "__main__":
    train()
    # val(0)
    writer.close()