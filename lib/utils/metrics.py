#
# metrics.py
# @author bulbasaur
# @description 
# @created 2019-10-29T20:34:37.569Z+08:00
# @last-modified 2019-11-05T14:44:25.346Z+08:00
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
    acc = np.diag(hist).sum() / (hist.sum()+1e-9)
    acc_cls = np.diag(hist) / (hist.sum(axis=1))
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    
    freq = hist.sum(axis=1) /( hist.sum()+1e-9)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def IoU_per_class(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_i_mask = pred == cls
        true_i_mask = target == cls
        intersection = np.logical_and(pred_i_mask, true_i_mask).sum()  # Cast to long to prevent overflows
        union        = np.logical_or(pred_i_mask, true_i_mask).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / max(union, 1))
    return ious
