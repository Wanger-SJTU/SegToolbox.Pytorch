

import numpy as np
from PIL import Image

from .ade2k import ADE2K_labels
from .voc import VOC_labels

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information




label_info = {
    'voc':VOC_labels,
    'ade2k':ADE2K_labels
}
#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------
# id to label object
def idx2label(palette=VOC_labels):
    id2label = { label.id : label for label in palette}
    return id2label

def index2rgb(indexed, palette='voc'):
    id2label = idx2label(label_info[palette])
    if isinstance(indexed, Image.Image):
        indexed = np.array(indexed)
    w, h = indexed.shape
    rgb_img = np.zeros((w, h, 3))

    for i in id2label.keys():
        mask = indexed == i
        rgb_img[mask] = id2label[i].color
    return rgb_img.transpose(2,0,1)

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

def rgb2index(img,num_class):
    palette = np.array(img.getpalette()).reshape(256,3)
    img = np.array(img)
    indexed = np.zeros(shape=img.shape)
    for i in np.unique(img):
        for j in range(num_class):
            color = np.array(VOC_labels[j].color)
            if all(color == palette[i]):
                mask = img == i
                indexed[mask] = j
    return indexed
