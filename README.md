
# Semantic Segmentation ToolBox with Pytorch
A clean pipline for Semantic Segmentation Task

## Model
- [x]  VGG-based FCN
- [x]  resNet
- [x]  nonlocal

**TODO**
1. more backbone net

## Dataset 
- [x] VOC2012
- [x] ADE2K
- [x] CityScapes

> At the root of the dataset, there should be a txt file named train.txt or val.txt



## features

1. model could defined by yaml file
2. data visiualization via tensorboard and visdom
3. data pre-processing for both images and target
    - RandomPadding
    - RandomCropPad
    - Scale
    - CenterCrop
    - Resize
    - RandomCrop
    - RandomSizedCrop
    - RandomHorizontalFlip
    
## result

TODO..