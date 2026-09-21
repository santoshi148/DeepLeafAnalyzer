import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBNReLU


class ASPP(nn.Module):
    def __init__(self, cin=256, cout=256, rates=(1,6,12,18)):
        super().__init__()
        branch = cout//len(rates)
        self.branches=nn.ModuleList([ConvBNReLU(cin,branch,3,d=r) for r in rates])
        self.project=ConvBNReLU(branch*len(rates),cout,1)
    def forward(self,x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))
