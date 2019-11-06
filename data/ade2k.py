# -*- coding: UTF-8 -*-
# ade2k.py
# @author wanger
# @description 
# @created 2019-11-05T19:39:10.854Z+08:00
# @last-modified 2019-11-06T13:49:22.370Z+08:00
#

import os
from PIL import Image
import numpy as np
from torch.utils.data import dataset
from .basedataset import BaseDataset

class ADE2K(BaseDataset):
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(ADE2K, self).__init__(root, image_set, transform, 
                                    target_transform, transforms)
        assert image_set in ("train", "val")
        
        dataset_file = os.path.join(root, image_set)+'.txt'
        if not os.path.exists(dataset_file):
            raise RuntimeError('{} not exists'.format(dataset_file))
        with open(dataset_file, 'r', encoding='utf8') as f:
            file_names = [x.strip().split('.')[0] for x in f.readlines()]
            
        image_set = 'training' if image_set == 'train' else "validation"
       
        self.images = [os.path.join(root, 'images',      image_set, x + ".jpg") for x in file_names]
        self.masks  = [os.path.join(root, 'annotations', image_set, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))