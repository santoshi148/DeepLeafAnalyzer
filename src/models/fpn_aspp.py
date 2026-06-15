import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class TinyBackbone(nn.Module):
    def __init__(self, in_ch=3, widths=(32, 64, 128, 256)):
        super().__init__()
        self.s1 = nn.Sequential(ConvBNReLU(in_ch, widths[0]), ConvBNReLU(widths[0], widths[0]))
        self.s2 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(widths[0], widths[1]), ConvBNReLU(widths[1], widths[1]))
        self.s3 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(widths[1], widths[2]), ConvBNReLU(widths[2], widths[2]))
        self.s4 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(widths[2], widths[3]), ConvBNReLU(widths[3], widths[3]))
    def forward(self, x):
        c1 = self.s1(x); c2 = self.s2(c1); c3 = self.s3(c2); c4 = self.s4(c3)
        return [c1, c2, c3, c4]

class FPN(nn.Module):
    def __init__(self, in_channels=(32,64,128,256), out_ch=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList([ConvBNReLU(out_ch, out_ch) for _ in in_channels])
    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        p4 = lat[3]
        p3 = lat[2] + F.interpolate(p4, size=lat[2].shape[-2:], mode='nearest')
        p2 = lat[1] + F.interpolate(p3, size=lat[1].shape[-2:], mode='nearest')
        p1 = lat[0] + F.interpolate(p2, size=lat[0].shape[-2:], mode='nearest')
        return self.smooth[0](p1)

class ASPP(nn.Module):
    def __init__(self, in_ch=128, out_ch=128, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_ch, out_ch, k=1, p=0, d=1) if r == 1 else ConvBNReLU(in_ch, out_ch, k=3, p=r, d=r)
            for r in rates
        ])
        self.project = ConvBNReLU(out_ch * len(rates), out_ch, k=1, p=0)
    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))
