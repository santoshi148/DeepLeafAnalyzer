import torch
from src.models.unet import UNet
from src.models.hybrid_leaf_disease_net import HybridLeafDiseaseNet

def test_unet_shape():
    x=torch.randn(1,3,128,128); y=UNet(base=8)(x); assert y.shape==(1,1,128,128)

def test_classifier_shape():
    x=torch.randn(1,3,128,128); m=HybridLeafDiseaseNet(4,channels=(8,16,32,64),fpn_channels=32,heads=4,lstm_hidden=32); y=m(x); assert y.shape==(1,4)
