#!/usr/bin/env python
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from src.models.unet import UNet
from src.evaluation.efficiency import count_parameters

x=torch.randn(2,3,224,224)
seg=UNet(base=16)(x)
model=HybridLeafDiseaseNet(num_classes=5, channels=(16,32,64,128), fpn_channels=64, heads=4, lstm_hidden=64)
logits=model(x)
print('UNet output:',tuple(seg.shape))
print('Classifier output:',tuple(logits.shape))
print('Classifier parameters:',count_parameters(model))
assert seg.shape==(2,1,224,224)
assert logits.shape==(2,5)
print('SMOKE TEST PASSED')
