
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
                 transforms=None):
        super(CityScapes, self).__init__(root, image_set, transform, 
                                    target_transform, transforms)
        self.images = []
        self.masks  = []