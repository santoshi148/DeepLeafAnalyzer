import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, d=1):
        p = d*(k-1)//2
        super().__init__(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvBNReLU(c,c,3)
        self.conv2 = nn.Sequential(nn.Conv2d(c,c,3,1,1,bias=False), nn.BatchNorm2d(c))
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.act(x + self.conv2(self.conv1(x)))
