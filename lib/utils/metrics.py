#
# metrics.py
# @author bulbasaur
# @description 
# @created 2019-10-29T20:34:37.569Z+08:00
# @last-modified 2019-11-23
#
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask_ture = (label_true >= 0) & (label_true < n_class)
    mask_pred = (label_pred >= 0) & (label_pred < n_class)
    mask = np.logical_and(mask_ture, mask_pred)
    hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean I0U
        - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def IoU_per_class(pred, trues, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_i_mask = pred == cls
        true_i_mask = trues == cls
        intersection = np.logical_and(pred_i_mask, true_i_mask).sum()  # Cast to long to prevent overflows
        union        = np.logical_or(pred_i_mask, true_i_mask).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / max(union, 1))
    return ious


def cls_score(preds, trues, n_class, threshold=0.8):
    trues = _get_batch_label_vector(trues, n_class)
    preds = (preds > threshold).astype(np.uint8)

    TP = ((preds == 1) & (trues == 1)).sum()
    TN = ((preds == 0) & (trues == 0)).sum()
    FN = ((preds == 0) & (trues == 1)).sum()
    FP = ((preds == 1) & (trues == 0)).sum()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return {"precise":p, "recall":r, "F1":F1, "acc":acc}


def _get_batch_label_vector(target, nclass):
    
    batch = target.shape[0]
    tvect = np.zeros((batch, nclass))
    for i in range(batch):
        hist = np.histogram(target[i], bins=nclass, range=(0,nclass-1))
        vect = hist[0]>0
        tvect[i] = vect
    return tvect.astype(np.uint8)