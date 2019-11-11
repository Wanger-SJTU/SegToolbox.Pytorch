
from collections import namedtuple 
import numpy as np
from PIL import Image

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information

Label = namedtuple( 'Labelinfo' , [
    'name'  , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'id'    ,  # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.    
    'color'         # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of VOC_labels labels
#--------------------------------------------------------------------------------
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

ADE2K_labels = [
     #       name                     id       color
     Label("wall",                   0  ,    (120,120,120)),
     Label("building",               1  ,    (180,120,120)),
     Label("sky",                    2  ,    (6,  230,230)),
     Label("floor",                  3  ,    (80, 50,  50)),
     Label("tree",                   4  ,    (4,  200,  3)),
     Label("ceiling",                5  ,    (120,120, 80)),
     Label("road",                   6  ,    (140,140,140)),
     Label("bed",                    7  ,    (204,  5,255)),
     Label("windowpane",             8  ,    (230,230,230)),
     Label("grass",                  9  ,    (  4,250,  7)),
     Label("cabinet",                10 ,    (224,  5,255)),
     Label("sidewalk",               11 ,    (235,255,  7)),
     Label("person",                 12 ,    (150,  5, 61)),
     Label("earth",                  13 ,    (120,120, 70)),
     Label("door",                   14 ,    (  8,255, 51)),
     Label("table",                  15 ,    (255,  6, 82)),
     Label("mountain",               16 ,    (143,255,140)),
     Label("plant",                  17 ,    (204,255,  4)),
     Label("curtain",                18 ,    (255, 51,  7)),
     Label("chair",                  19 ,    (204, 70,  3)),
     Label("car",                    20 ,    (  0,102,200)),
     Label("water",                  21 ,    ( 61,230,250)),
     Label("painting",               22 ,    (255,  6, 51)),
     Label("sofa",                   23 ,    ( 11,102,255)),
     Label("shelf",                  24 ,    (255,  7, 71)),
     Label("house",                  25 ,    (255,  9,224)),
     Label("sea",                    26 ,    (  9,  7,230)),
     Label("mirror",                 27 ,    (220,220,220)),
     Label("rug",                    28 ,    (255,  9, 92)),
     Label("field",                  29 ,    (112,  9,255)),
     Label("armchair",               30 ,    (  8,255,214)),
     Label("seat",                   31 ,    (  7,255,224)),
     Label("fence",                  32 ,    (255,184,  6)),
     Label("desk",                   33 ,    ( 10,255, 71)),
     Label("rock",                   34 ,    (255, 41, 10)),
     Label("wardrobe",               35 ,    (  7,255,255)),
     Label("lamp",                   36 ,    (224,255,  8)),
     Label("bathtub",                37 ,    (102,  8,255)),
     Label("railing",                38 ,    (255,61,   6)),
     Label("cushion",                39 ,    (255,194,  7)),
     Label("base",                   40 ,    (255,122,  8)),
     Label("box",                    41 ,    (  0,255, 20)),
     Label("column",                 42 ,    (255,  8, 41)),
     Label("signboard",              43 ,    (255,  5,153)),
     Label("chest",                  44 ,    (  6, 51,255)),
     Label("counter",                45 ,    (235, 12,255)),
     Label("sand",                   46 ,    (160,150, 20)),
     Label("sink",                   47 ,    (  0,163,255)),
     Label("skyscraper",             48 ,    (140,140,140)),
     Label("fireplace",              49 ,    (250, 10, 15)),
     Label("refrigerator",           50 ,    ( 20,255,  0)),
     Label("grandstand",             51 ,    ( 31,255,  0)),
     Label("path",                   52 ,    (255, 31,  0)),
     Label("stairs",                 53 ,    (255,224,  0)),
     Label("runway",                 54 ,    (153,255,  0)),
     Label("case",                   55 ,    (  0,  0,255)),
     Label("pool table",             56 ,    (255, 71,  0)),
     Label("pillow",                 57 ,    (  0,235,255)),
     Label("screen door",            58 ,    (  0,173,255)),
     Label("stairway",               59 ,    ( 31,  0,255)),
     Label("river",                  60 ,    ( 11,200,200)),
     Label("bridge",                 61 ,    (255, 82,  0)),
     Label("bookcase",               62 ,    (  0,255,245)),
     Label("blind",                  63 ,    (  0, 61,255)),
     Label("coffee table",           64 ,    (  0,255,112)),
     Label("toilet",                 65 ,    (  0,255,133)),
     Label("flower",                 66 ,    (255,  0,  0)),
     Label("book",                   67 ,    (255,163,  0)),
     Label("hill",                   68 ,    (255,102,  0)),
     Label("bench",                  69 ,    (194,255,  0)),
     Label("countertop",             70 ,    (  0,143,255)),
     Label("stove",                  71 ,    ( 51,255,  0)),
     Label("palm",                   72 ,    (  0, 82,255)),
     Label("kitchen island",         73 ,    (  0,255, 41)),
     Label("computer",               74 ,    (  0,255,173)),
     Label("swivel chair",           75 ,    ( 10,  0,255)),
     Label("boat",                   76 ,    (173,255,  0)),
     Label("bar",                    77 ,    (  0,255,153)),
     Label("arcade machine",         78 ,    (255, 92,  0)),
     Label("hovel",                  79 ,    (255,  0,255)),
     Label("bus",                    80 ,    (255,  0,245)),
     Label("towel",                  81 ,    (255,  0,102)),
     Label("light",                  82 ,    (255,173,  0)),
     Label("truck",                  83 ,    (255,  0, 20)),
     Label("tower",                  84 ,    (255,184,184)),
     Label("chandelier",             85 ,    (  0, 31,255)),
     Label("awning",                 86 ,    (  0,255, 61)),
     Label("streetlight",            87 ,    (  0, 71,255)),
     Label("booth",                  88 ,    (255,  0,204)),
     Label("television",             89 ,    (  0,255,194)),
     Label("airplane",               90 ,    (  0,255, 82)),
     Label("dirt track",             91 ,    (  0, 10,255)),
     Label("apparel",                92 ,    (  0,112,255)),
     Label("pole",                   93 ,    ( 51,  0,255)),
     Label("land",                   94 ,    (  0,194,255)),
     Label("bannister",              95 ,    (  0,122,255)),
     Label("escalator",              96 ,    (  0,255,163)),
     Label("ottoman",                97 ,    (255,153,  0)),
     Label("bottle",                 98 ,    (  0,255, 10)),
     Label("buffet",                 99 ,    (255,112,  0)),
     Label("poster",                 100,    (143,255,  0)),
     Label("stage",                  101,    ( 82,  0,255)),
     Label("van",                    102,    (163,255,  0)),
     Label("ship",                   103,    (255,235,  0)),
     Label("fountain",               104,    (  8,184,170)),
     Label("conveyer belt",          105,    (133,  0,255)),
     Label("canopy",                 106,    (  0,255, 92)),
     Label("washer",                 107,    (184,  0,255)),
     Label("plaything",              108,    (255,  0, 31)),
     Label("swimming pool",          109,    (  0,184,255)),
     Label("stool",                  110,    (  0,214,255)),
     Label("barrel",                 111,    (255,  0,112)),
     Label("basket",                 112,    ( 92,255,  0)),
     Label("waterfall",              113,    (  0,224,255)),
     Label("tent",                   114,    (112,224,255)),
     Label("bag",                    115,    ( 70,184,160)),
     Label("minibike",               116,    (163,  0,255)),
     Label("cradle",                 117,    (153,  0,255)),
     Label("oven",                   118,    ( 71,255,  0)),
     Label("ball",                   119,    (255,  0,163)),
     Label("food",                   120,    (255,204,  0)),
     Label("step",                   121,    (255,  0,143)),
     Label("tank",                   122,    (  0,255,235)),
     Label("trade name",             123,    (133,255,  0)),
     Label("microwave",              124,    (255,  0,235)),
     Label("pot",                    125,    (245,  0,255)),
     Label("animal",                 126,    (255,  0,122)),
     Label("bicycle",                127,    (255,245,  0)),
     Label("lake",                   128,    ( 10,190,212)),
     Label("dishwasher",             129,    (214,255,  0)),
     Label("screen",                 130,    (  0,204,255)),
     Label("blanket",                131,    ( 20,  0,255)),
     Label("sculpture",              132,    (255,255,  0)),
     Label("hood",                   133,    (  0,153,255)),
     Label("sconce",                 134,    (  0, 41,255)),
     Label("vase",                   135,    (  0,255,204)),
     Label("traffic light",          136,    ( 41,  0,255)),
     Label("tray",                   137,    ( 41,255,  0)),
     Label("ashcan",                 138,    (173,  0,255)),
     Label("fan",                    139,    (  0,245,255)),
     Label("pier",                   140,    ( 71,  0,255)),
     Label("crt screen",             141,    (122,  0,255)),
     Label("plate",                  142,    (  0,255,184)),
     Label("monitor",                143,    (  0, 92,255)),
     Label("bulletin board",         144,    (184,255,  0)),
     Label("shower",                 145,    (  0,133,255)),
     Label("radiator",               146,    (255,214,  0)),
     Label("glass",                  147,    ( 25,194,194)),
     Label("clock",                  148,    (102,255,  0)),
     Label("flag",                   149,    ( 92,  0,255)),
     Label("unlabeled",              255,    (  0,  0,  0))
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

label_info = {
    'voc':VOC_labels,
    'ade2k':ADE2K_labels
}
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

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("Procesing VOC labels:")
    from torchvision.datasets import VOCSegmentation
    import os
    import tqdm
    root = "/data/CH/data/PASCAL/VOCdevkit/VOC2012/SegmentationClass"
    for file in tqdm.tqdm(os.listdir(root), total=len(os.listdir(root))):
        path = os.path.join(root, file)
        img = Image.open(path).convert("P")
        indexed = rgb2index(img)
        newimg = Image.fromarray(indexed.astype(np.uint8))
        newimg.save(os.path.join("/data/CH/data/PASCAL/VOCdevkit/VOC2012/SegmentationClassIndex", file))
    
    root = "/data/CH/data/PASCAL/VOCdevkit/VOC2012/SegmentationClassIndex"
    total = set()
    for file in tqdm.tqdm(os.listdir(root), total=len(os.listdir(root))):
        path = os.path.join(root, file)
        img = np.array(Image.open(path))
        total = total | set(np.unique(img).tolist())
    print(total)
