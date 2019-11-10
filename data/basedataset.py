# -*- coding: UTF-8 -*-
# basedataset.py
# @author wanger
# @description 
# @created 2019-11-06T13:39:41.110Z+08:00
# @last-modified 2019-11-10T21:19:41.730Z+08:00
#

import pdb
import os
from PIL import Image
import numpy as np

from torch.utils.data import dataset


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(dataset.Dataset):
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 loadMemory=False):
        super(BaseDataset, self).__init__()
        assert image_set in ("train", "val", "trainval")
        self.root = root
        self.transforms = transforms
        self.target_transform = target_transform
        self.transform = transform
        self.loadMemory = loadMemory
        self.images = []
        self.masks  = []
        
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        if not self.loadMemory:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index])#.convert('P')
        else:
            img = self.images[index]
            target = self.masks[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        imgmap = (img, target)
        if self.transforms is not None:
            imgmap = self.transforms(imgmap) 
        return imgmap, self.masks[index]

    def __len__(self):
        return len(self.images)
        
    def loadImgInMemory(self):
        print("--"*5)
        print("loading images into memory")
        images = [Image.open(x).convert('RGB') for x in self.images]
        masks  = [Image.open(x) for x in self.masks]
        self.images = images
        self.masks = masks
        print("loaded {} images into memory".format(len(images)))
        print("--"*5)
