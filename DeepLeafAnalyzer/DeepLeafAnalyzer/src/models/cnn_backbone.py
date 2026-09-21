import torch.nn as nn
from .blocks import ConvBNReLU, ResidualBlock


class CNNBackbone(nn.Module):
    def __init__(self, channels=(64,128,256,512)):
        super().__init__()
        c1,c2,c3,c4 = channels
        self.stem = nn.Sequential(ConvBNReLU(3,c1), ResidualBlock(c1), nn.MaxPool2d(2))
        self.s2 = nn.Sequential(ConvBNReLU(c1,c2), ResidualBlock(c2), nn.MaxPool2d(2))
        self.s3 = nn.Sequential(ConvBNReLU(c2,c3), ResidualBlock(c3), nn.MaxPool2d(2))
        self.s4 = nn.Sequential(ConvBNReLU(c3,c4), ResidualBlock(c4), nn.MaxPool2d(2))
    def forward(self,x):
        c2=self.stem(x)   #112
        c3=self.s2(c2)    #56
        c4=self.s3(c3)    #28
        c5=self.s4(c4)    #14
        return c2,c3,c4,c5
