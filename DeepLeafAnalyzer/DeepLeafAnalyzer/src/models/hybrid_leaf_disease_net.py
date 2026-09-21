import torch
import torch.nn as nn
from .cnn_backbone import CNNBackbone
from .fpn import FPN
from .aspp import ASPP
from .self_attention import SpatialSelfAttention
from .csam import CSAM
from .feature_lstm import FeatureSequenceLSTM


class HybridLeafDiseaseNet(nn.Module):
    def __init__(self, num_classes, channels=(64,128,256,512), fpn_channels=256, heads=4, lstm_hidden=256, dropout=0.3,
                 use_fpn=True, use_aspp=True, use_self_attention=True, use_csam=True, use_lstm=True):
        super().__init__()
        self.flags=dict(use_fpn=use_fpn,use_aspp=use_aspp,use_self_attention=use_self_attention,use_csam=use_csam,use_lstm=use_lstm)
        self.backbone=CNNBackbone(channels)
        self.fpn=FPN(channels,fpn_channels)
        self.c5proj=nn.Conv2d(channels[-1],fpn_channels,1)
        self.aspp=ASPP(fpn_channels,fpn_channels)
        self.attn=SpatialSelfAttention(fpn_channels,heads)
        self.csam=CSAM(fpn_channels)
        self.lstm=FeatureSequenceLSTM(fpn_channels,lstm_hidden)
        self.pool=nn.AdaptiveAvgPool2d(1)
        feat_dim=lstm_hidden if use_lstm else fpn_channels
        self.classifier=nn.Sequential(nn.Linear(feat_dim,256),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(256,num_classes))
    def forward_features(self,x):
        feats=self.backbone(x)
        if self.flags['use_fpn']:
            x,_=self.fpn(feats)
        else:
            x=self.c5proj(feats[-1])
        if self.flags['use_aspp']: x=self.aspp(x)
        if self.flags['use_self_attention']: x=self.attn(x)
        if self.flags['use_csam']: x=self.csam(x)
        return x
    def forward(self,x):
        f=self.forward_features(x)
        if self.flags['use_lstm']:
            z=self.lstm(f)
        else:
            z=self.pool(f).flatten(1)
        return self.classifier(z)
