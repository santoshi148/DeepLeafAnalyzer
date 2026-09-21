import numpy as np


def bbox_to_mask(shape, bbox):
    h,w=shape[:2]; m=np.zeros((h,w),dtype=np.uint8)
    if bbox is None: return m
    x1,y1,x2,y2=[int(v) for v in bbox]
    x1=max(0,min(w-1,x1)); x2=max(0,min(w,x2)); y1=max(0,min(h-1,y1)); y2=max(0,min(h,y2))
    if x2>x1 and y2>y1: m[y1:y2,x1:x2]=1
    return m


def fuse_roi(image, unet_mask, bbox=None, min_intersection_ratio=0.05, fallback='unet'):
    mask=(unet_mask>0).astype(np.uint8)
    boxmask=bbox_to_mask(mask.shape,bbox)
    inter=mask & boxmask
    denom=max(mask.sum(),1)
    ratio=inter.sum()/denom
    if bbox is not None and ratio>=min_intersection_ratio:
        final=inter; strategy='intersection'
    elif fallback=='unet' and mask.sum()>0:
        final=mask; strategy='unet'
    elif bbox is not None and boxmask.sum()>0:
        final=boxmask; strategy='bbox'
    else:
        final=np.ones_like(mask); strategy='full_image'
    ys,xs=np.where(final>0)
    if len(xs)==0:
        return image.copy(), final, 'full_image'
    x1,x2=xs.min(),xs.max()+1; y1,y2=ys.min(),ys.max()+1
    crop=image[y1:y2,x1:x2].copy()
    return crop, final, strategy
