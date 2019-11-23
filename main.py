#
# main.py
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
if cfg.MODEL.PRETRAIN:
    cfg.EXPERIMENT_NAME = "pretrain/"+cfg.EXPERIMENT_NAME
else:
    cfg.EXPERIMENT_NAME = "scratch/"+cfg.EXPERIMENT_NAME

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
    if cfg.SOLVER.WARMUP.WARMUP_ON:
        warmUpModel()
    adjustLearningRate(optim, cfg.SOLVER.BASE_LR)
    epoches = math.ceil(cfg.SOLVER.MAX_ITER/len(train_loader))
    for epoch in tqdm.tqdm(range(epoches), total=epoches, ncols=80):
        loader = tqdm.tqdm(train_loader, total=len(train_loader), 
                            ncols=80, leave=False)
        for idx,item in enumerate(loader):
            img = item[0].cuda()
            new_lbl = reMaskLabel(item[1].numpy().copy(), opts.ratio, cfg.DATASET.IGNOREIDX)
            new_lbl = torch.from_numpy(new_lbl).long().cuda()
            lbl_true = item[1]
            outputs = model(img)
            probs,*others = [outputs,] if isinstance(outputs, torch.Tensor) else outputs
            optim.zero_grad()
            loss,*loss_detail = Seg_loss(probs, others, new_lbl)# pred, lbl            loss.backward()
            loss.backward()
            optim.step()
            
            lbl_pred = probs.data.max(1)[1].cpu().detach().numpy()
            lbl_true = lbl_true.data.cpu().numpy()
            lbl_new  = new_lbl.data.cpu().numpy()
            src_img  = img.data.cpu().numpy()
            
            metrics = label_accuracy_score(lbl_true, lbl_pred, cfg.MODEL.NUM_CLASSES)
            
            if not loss_detail:
                for k,v in loss_detail:
                    writer.add_scalar("train/"+k,v.data.detach().item()) 
            writer.add_scalar("train/loss",loss.data.detach().item()) 
            writer.add_scalar("train/acc",     metrics[0])
            writer.add_scalar("train/acc_cls", metrics[1])
            writer.add_scalar("train/mean_iu", metrics[2])
            writer.add_scalar("train/fwavacc", metrics[3])
            loader.set_description(desc="[{}:{}] loss: {:.2f} iou:{:.2f}".format(
                epoch, writer.global_step, loss.data.item(), metrics[2]))
            if writer.global_step % 100 == 0:
                # writer.variable_summaries("grad_out", 'probs', probs.grad)
                # writer.model_para_summaries(model)
                # writer.model_para_grad_summaries(model)
                writer.add_image('train/img',     src_img[0].astype(np.uint8))
                writer.add_image('train/target',  index2rgb(lbl_true[0], cfg.DATASET.NAME))
                writer.add_image('train/target_new', index2rgb(lbl_new[0], cfg.DATASET.NAME))
                writer.add_image('train/pred',    index2rgb(lbl_pred[0], cfg.DATASET.NAME))

            if writer.global_step % cfg.TRAIN.EVAL_PERIOD == 0:
            #if writer.global_step % 1 == 0:
                val(epoch, save=True)
                model.train() 
            if writer.global_step > cfg.SOLVER.MAX_ITER:
                break
            writer.addStep()

val_idx = 0
best_miou = 0
def val(epoch,save=False):
    loader= tqdm.tqdm(val_loader, total=len(val_loader), ncols=80, leave=False)
    model.eval()
    iou = 0
    global val_idx,best_miou
    for item in loader:
        val_idx += 1
        img, lbl = item[0].cuda(), item[1].cuda()
        pos = list(map(int, item[2].numpy()[0])) if len(item) > 2 else None
        with torch.no_grad():
            outputs = model(img)
            probs,*others = [outputs,] if isinstance(outputs, torch.Tensor) else outputs
            loss,*loss_detail = Seg_loss(probs, others, lbl)# pred, lbl 
        
        lbl_pred = probs.data.max(1)[1].cpu().numpy()
        lbl_true = lbl.data.cpu().numpy()
        src_img  = img.data.cpu().numpy()
        shape = lbl_true.shape
        if pos is not None:
            lbl_pred = lbl_pred[:,pos[0]:shape[1]-pos[1],pos[2]:shape[2]-pos[3]] 
            lbl_true = lbl_true[:,pos[0]:shape[1]-pos[1],pos[2]:shape[2]-pos[3]]
            src_img  = src_img[:,:,pos[0]:shape[1]-pos[1],pos[2]:shape[2]-pos[3]]
        
        if val_idx % 50 == 0:
            writer.add_image('val/img', src_img[0].astype(np.uint8),val_idx)
            writer.add_image('val/target', index2rgb(lbl_true[0], cfg.DATASET.NAME),val_idx)
            writer.add_image('val/pred', index2rgb(lbl_pred[0],   cfg.DATASET.NAME),val_idx)
        
        metrics = label_accuracy_score(lbl_true, lbl_pred, cfg.MODEL.NUM_CLASSES)
        
        if not loss_detail:
            for k,v in loss_detail:
                writer.add_scalar("val/"+k,v.data.detach().item(),val_idx) 
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
    if save:
        filename = 'lastest.pth'.format(
                        str(epoch),str(writer.global_step))
        path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch'    : epoch,
            'iteration': writer.global_step,
            'optim'    : optim.state_dict(),
            'model_state_dict': model.state_dict(),
            'iou' : iou/len(val_loader)
        }, path)

    if iou/len(val_loader) > best_miou:
        best_miou = iou/len(val_loader)
        filename = 'bestmodel.pth'
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
            new_lbl = reMaskLabel(item[1].numpy().copy(), 
                        opts.ratio, cfg.DATASET.IGNOREIDX)
            new_lbl = torch.from_numpy(new_lbl).long().cuda()
            lbl_true = item[1] 
            
            outputs = model(img)
            probs,*others = [outputs,] if isinstance(outputs, torch.Tensor) \
                                       else outputs

            optim.zero_grad()
            loss = CrossEntropyLoss2d(probs, new_lbl, cfg.DATASET.IGNOREIDX)
            loss.backward()
            optim.step()
            
            lbl_pred = probs.data.max(1)[1].cpu().detach().numpy()
            lbl_true = lbl_true.data.cpu().numpy()
            lbl_new  = new_lbl.data.cpu().numpy()
            src_img  = img.data.cpu().numpy() 
            metrics = label_accuracy_score(lbl_true, lbl_pred, cfg.MODEL.NUM_CLASSES)

            if not loss_detail:
                for k,v in loss_detail:
                    writer.add_scalar("train/"+k,v.data.detach().item()) 

            writer.add_scalar("train/loss",loss.data.detach().item()) 
            writer.add_scalar("train/acc",     metrics[0])
            writer.add_scalar("train/acc_cls", metrics[1])
            writer.add_scalar("train/mean_iu", metrics[2])
            writer.add_scalar("train/fwavacc", metrics[3])
            # writer.add_scalar("train/lr",      lr)
            loader.set_description(desc="WARMUP: [{}:{}] loss: {:.2f} iou:{:.2f}".format(
                epoch, writer.global_step, loss.data.item(), metrics[2]))
            if writer.global_step % 100 == 0:
                # writer.variable_summaries("grad_out", 'probs', probs.grad)
                writer.model_para_summaries(model)
                # writer.model_para_grad_summaries(model)
                writer.add_image('train/img',     src_img[0].astype(np.uint8))
                writer.add_image('train/target',  index2rgb(lbl_true[0], cfg.DATASET.NAME))
                writer.add_image('train/target_new', index2rgb(lbl_new[0], cfg.DATASET.NAME))
                writer.add_image('train/pred',    index2rgb(lbl_pred[0], cfg.DATASET.NAME))

    val(epoch)

def load_check_point():
    pass


if __name__ == "__main__":
    # eval_data()
    train()
    # val()
    writer.close()