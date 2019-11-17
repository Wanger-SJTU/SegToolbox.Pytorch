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
log_cfg() # record training hyper-parameters

# visualization
visualizer = Visualizer(envs=cfg.MODEL.MODEL_NAME+"_ratio_"+str(opts.ratio))

if cfg.MODEL.PRETRAIN:
    cfg.MODEL.MODEL_NAME += "/pretrain"
else:
    cfg.MODEL.MODEL_NAME += "/scratch"

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

# if not os.path.exists(tensorboard_dir):
#     os.makedirs(tensorboard_dir)
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)

# writer = MySummaryWriter(log_dir=tensorboard_dir)

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
    transforms.RandomResize(),
    transforms.RandomCropPad(cfg.TRAIN.CROP_SIZE, cfg.DATASET.IGNOREIDX),
    transforms.RandomGaussBlur(),
    transforms.ToTensor()
])

# prepare dataset and model 
model = createSegModel(cfg, None).cuda()
optim = getOptimizer(cfg, model)

train_set = getDataset(cfg, 'train', transforms=train_transforms)
train_loader = torch.utils.data.dataloader.DataLoader(train_set, 
    batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE, 
    **cfg.DATALOADER)

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
            if epoch == 8 and idx ==22:
                import pdb;pdb.set_trace()
           
            img = item[0].cuda()
            new_lbl = item[1].cuda()
            # reMaskLabel(item[1].numpy().copy(), opts.ratio, cfg.DATASET.IGNOREIDX)
            # new_lbl = torch.from_numpy(new_lbl).long().cuda()
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
           
           
if __name__ == "__main__":
    # eval_data()
    train()
    # val(epoch)
    writer.close()