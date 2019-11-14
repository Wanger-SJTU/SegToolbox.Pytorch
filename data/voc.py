
import os
import sys
import collections
from .basedataset import BaseDataset,Label
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
                 loadMemory=False,
                 auxiliaryLoss=False):

        super(VOCSegmentation, self).__init__(root, 
                    image_set, transform, target_transform, 
                    transforms, loadMemory, auxiliaryLoss)
        
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

VOC_labels = [
    #       name                     id       color
    Label( 'background',             0 ,   (  0,  0,  0) ),#
    Label( 'aeroplane',              1 ,   (128,  0,  0) ),#
    Label( 'bicycle',                2 ,   (  0,128,  0) ),#
    Label( 'bird',                   3 ,   (128,128,  0) ),#
    Label( 'boat',                   4 ,   (  0,  0,128) ),#
    Label( 'bottle',                 5 ,   (128,  0,128) ),#
    Label( 'bus',                    6 ,   (  0,128,128) ),#
    Label( 'car',                    7 ,   (128,128,128) ),#
    Label( 'cat',                    8 ,   ( 64,  0,  0) ),#
    Label( 'chair',                  9 ,   (192,  0,  0) ),#
    Label( 'cow',                    10 ,  ( 64,128,  0) ),#
    Label( 'diningtable',            11 ,  (192,128,  0) ),#
    Label( 'dog',                    12 ,  ( 64,  0,128) ),#
    Label( 'horse',                  13 ,  (192,  0,128) ),#
    Label( 'motorbike',              14 ,  ( 64,128,128) ),#
    Label( 'person',                 15 ,  (192,128,128) ),#
    Label( 'potted plant',           16 ,  (  0, 64,  0) ),#?
    Label( 'sheep',                  17 ,  (128, 64,  0) ),
    Label( 'sofa',                   18 ,  (  0,192,  0) ),#
    Label( 'train',                  19 ,  (128,192,  0) ),#
    Label( 'tv/monitor',             20 ,  (  0, 64,128) ),#
    Label( 'unlabeled',              21 ,  (224,224,192) )#
]