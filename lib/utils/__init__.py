
import random
import numpy as np

random.seed(1123)

def reMaskLabel(originLbl, ratio, ignore_index):
    shape = originLbl.shape
    data = []
    a = list(range(0, shape[-1]*shape[-2]))
    threshold = shape[-1] * shape[-2] * ratio
    random.shuffle(a)
    b = np.array(a).reshape(shape[-2], shape[-1])
    data = [b]*shape[0]
    # for _ in range(shape[0]):
    #     random.shuffle(a)
    #     data.append(b)
    mask = np.array(data) > threshold
    originLbl[mask] = ignore_index
    return originLbl.copy()

def reMaskLabelNumPix(originLbl, ratio, ignore_index):
    shape = originLbl.shape
    data = []
    a = list(range(0, shape[1]*shape[2]))
    threshold = shape[1] * shape[2] * ratio
    for _ in range(shape[0]):
        random.shuffle(a)
        b = np.array(a).reshape(shape[1], shape[2])
        data.append(b)
    mask = np.array(data) > threshold
    originLbl[mask] = ignore_index
    return originLbl.copy()