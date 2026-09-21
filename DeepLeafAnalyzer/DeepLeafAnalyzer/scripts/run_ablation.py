#!/usr/bin/env python
import argparse,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.hybrid_leaf_disease_net import HybridLeafDiseaseNet

def main():
    p=argparse.ArgumentParser(); p.add_argument('--num-classes',type=int,default=3); p.add_argument('--image-size',type=int,default=224); a=p.parse_args()
    configs={'full':{},'no_fpn':{'use_fpn':False},'no_aspp':{'use_aspp':False},'no_self_attention':{'use_self_attention':False},'no_csam':{'use_csam':False},'no_lstm':{'use_lstm':False}}
    x=torch.randn(1,3,a.image_size,a.image_size)
    for name,kw in configs.items():
        m=HybridLeafDiseaseNet(a.num_classes,channels=(16,32,64,128),fpn_channels=64,heads=4,lstm_hidden=64,**kw); y=m(x); print(name, tuple(y.shape), sum(v.numel() for v in m.parameters()))
if __name__=='__main__': main()
