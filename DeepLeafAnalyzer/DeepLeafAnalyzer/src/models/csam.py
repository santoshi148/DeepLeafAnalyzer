import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        hidden=max(c//reduction,8)
        self.mlp=nn.Sequential(nn.Conv2d(c,hidden,1,bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden,c,1,bias=False))
    def forward(self,x):
        avg=self.mlp(torch.mean(x, dim=(2,3), keepdim=True))
        mx=self.mlp(torch.amax(x, dim=(2,3), keepdim=True))
        return x*torch.sigmoid(avg+mx)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__(); self.conv=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x):
        avg=torch.mean(x,1,keepdim=True); mx=torch.amax(x,1,keepdim=True)
        return x*torch.sigmoid(self.conv(torch.cat([avg,mx],1)))

class CSAM(nn.Module):
    def __init__(self,c):
        super().__init__(); self.ca=ChannelAttention(c); self.sa=SpatialAttention()
    def forward(self,x):
        return self.sa(self.ca(x))
