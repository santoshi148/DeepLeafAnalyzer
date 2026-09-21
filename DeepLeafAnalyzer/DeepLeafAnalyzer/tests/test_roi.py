import numpy as np
from src.roi.roi_fusion import fuse_roi

def test_roi_fusion():
    img=np.ones((100,100,3),dtype=np.uint8)*255
    mask=np.zeros((100,100),dtype=np.uint8); mask[20:80,20:80]=1
    crop, final, strategy=fuse_roi(img,mask,[30,30,70,70])
    assert crop.size>0 and final.sum()>0 and strategy=='intersection'
