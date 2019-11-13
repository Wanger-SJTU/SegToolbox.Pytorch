
import os
from PIL import Image
import numpy as np
from torch.utils.data import dataset
from .basedataset import BaseDataset

class CityScapes(BaseDataset):
     def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 loadMemory=False,
                 transforms=None):
        super(CityScapes, self).__init__(root, image_set, transform, 
                                    target_transform, transforms,
                                    loadMemory,auxiliaryLoss)
                                    
        assert image_set in ("train", "val") 
        dataset_file = os.path.join(root, image_set)+'.txt'
        if not os.path.exists(dataset_file):
            raise RuntimeError('{} not exists'.format(dataset_file))
        with open(dataset_file, 'r', encoding='utf8') as f:
            file_names = [x.strip().split('.')[0] for x in f.readlines()]
        self.images = [os.path.join(root, 'images',      image_set, x + ".jpg") for x in file_names]
        self.masks  = [os.path.join(root, 'annotations', image_set, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))