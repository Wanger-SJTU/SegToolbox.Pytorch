# -*- coding: UTF-8 -*-
# basedataset.py
# @author wanger
# @description 
# @created 2019-11-06T13:39:41.110Z+08:00
# @last-modified 2019-11-06T13:46:44.037Z+08:00
#

import os
from PIL import Image
import numpy as np

from torch.utils.data import dataset

class BaseDataset(dataset.Dataset):
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(BaseDataset, self).__init__()
        assert image_set in ("train", "val")
        self.root = root
        self.transforms = transforms
        self.target_transform = target_transform
        self.transform = transform
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
        # print(self.images[index])
        img    = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])#.convert('P')

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        if self.transforms is not None:
            img, target = self.transforms((img,target))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target #, self.masks[index]

    def __len__(self):
        return len(self.images)
