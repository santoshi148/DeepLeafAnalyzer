#!/usr/bin/env python
import argparse,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torchvision import models

def build(name,n):
    if name=='vgg16': m=models.vgg16(weights=None); m.classifier[-1]=torch.nn.Linear(m.classifier[-1].in_features,n)
    elif name=='resnet50': m=models.resnet50(weights=None); m.fc=torch.nn.Linear(m.fc.in_features,n)
    elif name=='mobilenet_v2': m=models.mobilenet_v2(weights=None); m.classifier[-1]=torch.nn.Linear(m.classifier[-1].in_features,n)
    elif name=='efficientnet_b0': m=models.efficientnet_b0(weights=None); m.classifier[-1]=torch.nn.Linear(m.classifier[-1].in_features,n)
    else: raise ValueError(name)
    return m

def main():
    p=argparse.ArgumentParser(); p.add_argument('--num-classes',type=int,default=3); p.add_argument('--image-size',type=int,default=224); a=p.parse_args(); x=torch.randn(1,3,a.image_size,a.image_size)
    for n in ['vgg16','resnet50','mobilenet_v2','efficientnet_b0']:
        m=build(n,a.num_classes); print(n,tuple(m(x).shape),sum(p.numel() for p in m.parameters()))
if __name__=='__main__': main()
