
import pdb

from .voc import VOCSegmentation
from .ade2k import ADE2K

datasets = {
    'voc'   :VOCSegmentation,
    'ade2k' :ADE2K
}

__all__ = ["getDataset"]
def getDataset(cfg, 
               image_set='train',
               transform=None,
               target_transform=None,
               transforms=None):
    if image_set == 'train' or image_set == 'trainval' :
        return datasets[cfg.DATASET.NAME](root=cfg.DATASET.DIR,
                                      image_set=image_set,
                                      transform=transform,
                                      target_transform=target_transform,
                                      transforms=transforms,
                                      loadMemory=cfg.TRAIN.LODEMEMORY)
    else:
        return datasets[cfg.DATASET.NAME](root=cfg.DATASET.DIR,
                                      image_set=image_set,
                                      transform=transform,
                                      target_transform=target_transform,
                                      transforms=transforms,
                                      loadMemory=cfg.TEST.LODEMEMORY)
