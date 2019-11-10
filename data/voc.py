
import os
import sys
import collections
from .basedataset import BaseDataset
from PIL import Image

class VOCSegmentation(BaseDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 loadMemory=False):
        super(VOCSegmentation, self).__init__(root, image_set, transform, target_transform, transforms, loadMemory)
        
        assert image_set in  ("train", "trainval", "val")
        self.image_set = image_set 

        voc_root = os.path.join(self.root, "VOCdevkit", "VOC2012")
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClassIndex')

        if not os.path.isdir(voc_root):
            print(voc_root)
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

        if self.loadMemory:
            self.loadImgInMemory()