import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn_aspp import TinyBackbone, FPN, ASPP, ConvBNReLU
from .attention import CSAM, ViTSelfAttention2D

class HybridLeafDiseaseNet(nn.Module):
    def __init__(self, num_classes=38, feature_ch=128, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.backbone = TinyBackbone(widths=(32,64,128,256))
        self.fpn = FPN((32,64,128,256), feature_ch)
        self.aspp = ASPP(feature_ch, feature_ch)
        self.csam = CSAM(feature_ch)
        self.vit_attn = ViTSelfAttention2D(feature_ch, heads=4, patch_pool=4)
        self.refine = nn.Sequential(ConvBNReLU(feature_ch, feature_ch), nn.MaxPool2d(2), ConvBNReLU(feature_ch, feature_ch))
        self.lstm = nn.LSTM(input_size=feature_ch, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden*2, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    def forward_features(self, x):
        feats = self.backbone(x)
        x = self.fpn(feats)
        x = self.aspp(x)
        x = self.csam(x)
        x = self.vit_attn(x)
        x = self.refine(x)
        return x
    def forward(self, x):
        fmap = self.forward_features(x)
        b, c, h, w = fmap.shape
        seq = fmap.flatten(2).transpose(1, 2)
        _, (hn, _) = self.lstm(seq)
        feat = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.classifier(feat)
