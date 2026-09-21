import numpy as np

def iou_score(pred, target, eps=1e-7):
    p=pred.astype(bool); t=target.astype(bool); inter=np.logical_and(p,t).sum(); union=np.logical_or(p,t).sum(); return (inter+eps)/(union+eps)

def dice_score(pred, target, eps=1e-7):
    p=pred.astype(bool); t=target.astype(bool); inter=np.logical_and(p,t).sum(); return (2*inter+eps)/(p.sum()+t.sum()+eps)
