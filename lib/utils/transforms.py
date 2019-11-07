
import torch
import random
import numbers
import math
import collections

from PIL import ImageOps, Image, ImageFilter
import numpy as np


class RandomPadding:
    def __init__(self, target_Size):
        self.target_Size = target_Size

    def __call__(self, imgmap):
        img,lbl = imgmap
        left,right,top,down = 0, 0, 0, 0
        if isinstance(img, Image.Image):
            shape = img.size
        elif isinstance(img, np.ndarray):
            shape = img.shape
            try:
                img = Image.fromarray(img)
            except e:
                raise Exception(e)
        else:
            raise ValueError("Not support type {}".format(type(img)))
        left = random.randint(0, self.target_Size[0]-shape[0])
        top  = random.randint(0, self.target_Size[1]-shape[1])
        right = self.target_Size[0]-shape[0] - left
        down  = self.target_Size[1]-shape[1] - top
        return ImageOps.expand(img, border=(left, top, right, down), fill=0), \
               ImageOps.expand(lbl, border=(left, top, right, down), fill=0)

class RandomCropPad:
    '''
    '''
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        elif isinstance(size, collections.Iterable) and len(size)==2:
            self.size = size
        else:
            raise ValueError("size must be number or Iterable with size =2")
    
    def __call__(self, imgmap):
        shape = imgmap[0].size
        if shape[0] >= self.size[0] and shape[1] >= self.size[1]:
            rc = RandomCrop(self.size)
            return rc(imgmap)
        if shape[0] > self.size[0]:
            rc = RandomCrop((self.size[0], shape[1]))
            imgmap = rc(imgmap)
            rp = RandomPadding(self.size)
            return rp(imgmap)
        if shape[1] > self.size[1]:
            rc = RandomCrop((shape[0], self.size[1]))
            imgmap = rc(imgmap)
            rp = RandomPadding(self.size)
            return rp(imgmap)
        else:
            rp = RandomPadding(self.size)
            return rp(imgmap)

class Scale:
    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap): 
        if isinstance(self.size, int):
            w, h = imgmap[0].size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                for idx in range(len(imgmap)):
                    if isinstance(imgmap[idx], Image.Image):
                        imgmap[idx] = imgmap[idx].resize((ow, oh), self.interpolation)
                    else:
                        raise NotImplementedError("only PIL Image supported")
                return imgmap
            else:
                oh = self.size
                ow = int(self.size * w / h)
                for idx in range(len(imgmap)):
                    imgmap[idx] = imgmap[idx].resize((ow, oh), self.interpolation)
                return imgmap
        else:
            for idx in range(len(imgmap)):
                imgmap[idx] = imgmap[idx].resize(self.size, self.interpolation)
            return imgmap

class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        w, h = imgmap[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        for idx in range(len(imgmap)):
            if isinstance(imgmap[idx], Image.Image):
                imgmap[idx] = imgmap[idx].crop((x1, y1, x1 + tw, y1 + th))
            else:
                raise NotImplementedError("only PIL Image supported")
        return imgmap

class Resize:

    def __init__(self, size, interpolation=Image.NEAREST):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        elif isinstance(size, collections.Iterable) and len(size)==2:
            self.size = size
        else:
            raise ValueError("size must be number or Iterable with size =2")
        self.interpolation = interpolation

    def __call__(self, imgmap):
        for i in range(2):
            if isinstance(imgmap[i], Image.Image):
                imgmap[i] = imgmap[i].resize(self.size, self.interpolation)
            else:
                raise NotImplementedError("Only support PIL Image")
        return imgmap

class RandomCropWithPOS:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        # print(self.size)
        w, h = imgmap[0].size
        if self.size is not None:
            th, tw = self.size
            if w <= tw and h <= th:
                return imgmap
            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
            for idx in range(len(imgmap)):
                if isinstance(imgmap[idx], np.ndarray):
                    if len(imgmap[idx].shape) == 2:
                        imgmap[idx] = imgmap[idx][x1:x1 + tw, y1:y1 + th]
                    elif len(imgmap[idx].shape) == 3:
                        imgmap[idx] = imgmap[idx][:,x1:x1 + tw, y1:y1 + th]
                    else:
                        print(imgmap[idx])
                        print(imgmap[idx].shape)
                        raise NotImplementedError
                elif isinstance(imgmap[idx], Image.Image):
                    imgmap[idx] = imgmap[idx].crop((x1, y1, x1 + tw, y1 + th))
                else:
                    print(type(imgmap[idx]))
                    raise NotImplementedError
            pos = [x1,y1, x1+tw, y1+th]
            return imgmap, pos
        else:
            return imgmap

class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, imgmap):
        imgmap = list(imgmap)
        w, h = imgmap[0].size
        if self.size is not None:
            th, tw = self.size
            if w <= tw and h <= th:
                return imgmap
            else:
                x1 = random.randint(0, w - tw) if w > tw else 0
                y1 = random.randint(0, h - th) if h > th else 0
            for idx in range(len(imgmap)):
                if isinstance(imgmap[idx], Image.Image):
                    imgmap[idx] = imgmap[idx].crop((x1, y1, x1 + tw, y1 + th))
                else:
                    raise NotImplementedError("only PIL Image supported")
            return imgmap
        else:
            return imgmap

class RandomSizedCrop:
    def __init__(self, size, interpolation=Image.NEAREST):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2
            self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img, *_ = imgmap
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.5, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                for idx in range(len(imgmap)):
                    imgmap[idx] = imgmap[idx].crop((x1, y1, x1 + w, y1 + h))

                for item in imgmap:
                    assert item.size == (w, h)

                for idx in range(len(imgmap)):
                    if isinstance(imgmap[idx], Image.Image):
                        imgmap[idx] = imgmap[idx].resize(self.size, self.interpolation)
                    else:
                        raise NotImplementedError("Only support PIL Image")
                return imgmap

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(imgmap))

class RandomHorizontalFlip:
    def __init__(self, ratio=0.5):
        self.ratio = 0.5
    def __call__(self, imgmap):
        if random.uniform(0,1) < self.ratio:
            for idx in range(len(imgmap)):
                if isinstance(imgmap[idx], Image.Image):
                    imgmap[idx] = imgmap[idx].transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    raise NotImplementedError("Only support PIL Image")
                    
        return imgmap

class RandomVerticalFlip:
    def __init__(self, ratio=0.5):
        self.ratio = 0.5
    def __call__(self, imgmap):
        if random.uniform(0, 1) < self.ratio:
            for idx in range(len(imgmap)):
                if isinstance(imgmap[idx], Image.Image):
                    imgmap[idx] = imgmap[idx].transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    raise NotImplementedError("Only support PIL Image")
        return imgmap
  
class RandomRotation:

    def __call__(self, imgmap, degree=10):
        # img, target = imgmap
        deg = np.random.randint(-degree, degree, 1)[0]
        for idx in range(len(imgmap)):
            if isinstance(imgmap[idx], Image.Image):
                imgmap[idx] = imgmap[idx].rotate(deg)
            else:
                raise NotImplementedError("Only support PIL Image")
        return imgmap

class RandomGaussBlur:
        
    def __call__(self, imgmap, r=1.0):
        # img, target = imgmap
        if isinstance(imgmap[idx], Image.Image):
            imgmap[0].filter(ImageFilter.GaussianBlur(r))
        else:
            raise NotImplementedError("Only support PIL Image")
        return imgmap
        
class Scale_Fixed:
    def __init__(self, scale=4):
        self.scale = scale
    def __call__(self, imgmap):
        scale = self.scale
        img, *_ = imgmap
       
        if isinstance(img, Image.Image):
            new_size = [int(img.size[0]*scale), int(img.size[1]*scale)]
        elif isinstance(img, np.ndarray):
            new_size = (int(img.shape[0]*scale), int(img.shape[1]*scale))
        
        for idx in range(len(imgmap)):
            if isinstance(imgmap[idx], Image.Image):
                imgmap[idx] = imgmap[idx].resize(new_size, Image.NEAREST)
            else:
                raise NotImplementedError("Only support PIL Image")
        return imgmap

class RandomResize:
    def __call__(self, imgmap, scale=(1, 1.2)):
        img, *_ = imgmap
        assert imgmap[0].size == imgmap[1].size, "{0} and {1}".format(imgmap[0].size, imgmap[1].size)
        scale_rand = np.random.randint(int(scale[0]*10), int(scale[1]*10))/10
        new_size = (int(img.size[0]*scale_rand), int(img.size[1]*scale_rand))
        
        for idx in range(len(imgmap)):
            if isinstance(imgmap[idx], Image.Image):
                imgmap[idx] = imgmap[idx].resize(new_size)
            else:
                raise NotImplementedError("Only support PIL Image")
        return imgmap

class ToTensor:
    def __call__(self, imgmap):
        img,lbl = imgmap
        if isinstance(img, Image.Image):
            img = np.array(img)
            lbl = np.array(lbl)
        if len(img.shape) == 2:
            img = img[np.newaxis,:,:]
            img = np.concatenate([img,img,img], axis=0)
        if img.shape[-1] == 3:
            img = img.transpose(2,0,1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

class ValToTensor:
    def __call__(self, imgmap):
        img,lbl, others = imgmap
        if isinstance(img, Image.Image):
            img = np.array(img)
            lbl = np.array(lbl)
        if len(img.shape) == 2:
            img = img[np.newaxis,:,:]
            img = np.concatenate([img,img,img], axis=0)
        if img.shape[-1] == 3:
            img = img.transpose(2,0,1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        others = torch.from_numpy(np.array(others)).long()
        return img, lbl, others

class ValPadding:
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, imgmap):
        img,lbl = imgmap

        left,right,top,down = 0, 0, 0, 0
        if isinstance(img, Image.Image):
            shape = img.size
        elif isinstance(img, np.ndarray):
            shape = img.shape
            try:
                img = Image.fromarray(img)
            except e:
                raise Exception(e)
        else:
            raise ValueError("Not support type {}".format(type(img)))
        row = shape[0] % self.scale
        row = 0 if row == 0 else self.scale - row #+ 1
        col = shape[1] % self.scale
        col = 0 if col == 0 else self.scale - col # + 1

        left = random.randint(0, row)
        top  = random.randint(0, top)
        right = row - left
        down  = col - top
        return ImageOps.expand(img, border=(left, top, right, down), fill=0), \
               ImageOps.expand(lbl, border=(left, top, right, down), fill=0), \
               (left, right, top, down)
