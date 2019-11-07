

from torchvision.datasets.voc import VOCSegmentation
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
    return datasets[cfg.DATASET.NAME](root=cfg.DATASET.DIR,
                                      image_set=image_set,
                                      transform=transform,
                                      target_transform=target_transform,
                                      transforms=transforms)