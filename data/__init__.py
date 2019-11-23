
import pdb

from .voc import VOCSegmentation
from .voc import VOC_FIX
from .ade2k import ADE2K
from .colorize import index2rgb

datasets = {
    'voc'   :VOCSegmentation,
    'ade2k' :ADE2K,
    'voc_fix':VOC_FIX
}

__all__ = ["getDataset", "index2rgb"]

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


def getRawDataset(name, root, image_set='train'):
    return datasets[name](root=root, image_set=image_set, path=True)

def getFixDataset(cfg, image_set='train',
                transform=None,
                target_transform=None,
                transforms=None):
    return datasets[cfg.DATASET.NAME](root=cfg.DATASET.DIR,
                        image_set=image_set,
                        transform=transform,
                        target_transform=target_transform,
                        transforms=transforms,
                        loadMemory=cfg.TRAIN.LODEMEMORY,
                        ratio=cfg.TRAIN.DROPOUT_RATE)