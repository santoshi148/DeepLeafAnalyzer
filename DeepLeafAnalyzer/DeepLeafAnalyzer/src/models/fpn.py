import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels=(64,128,256,512), out_channels=256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c,out_channels,1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels,out_channels,3,padding=1) for _ in in_channels])
    def forward(self, feats):
        c2,c3,c4,c5 = feats
        p5=self.lat[3](c5)
        p4=self.lat[2](c4)+F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3=self.lat[1](c3)+F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2=self.lat[0](c2)+F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        ps=[self.smooth[i](p) for i,p in enumerate([p2,p3,p4,p5])]
        # Aggregate at 14x14 for stable compute cost
        target=ps[-1].shape[-2:]
        agg=sum(F.adaptive_avg_pool2d(p,target) for p in ps)/len(ps)
        return agg, ps
