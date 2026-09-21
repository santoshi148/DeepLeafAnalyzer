import torch
import torch.nn as nn


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,dim*2), nn.GELU(), nn.Linear(dim*2,dim))
    def forward(self,x):
        b,c,h,w=x.shape
        seq=x.flatten(2).transpose(1,2)
        z=self.norm(seq)
        y,_=self.attn(z,z,z, need_weights=False)
        seq=seq+y
        seq=seq+self.ff(seq)
        return seq.transpose(1,2).reshape(b,c,h,w)
