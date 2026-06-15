import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(ch, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, ch, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=(2,3), keepdim=True)
        mx = torch.amax(x, dim=(2,3), keepdim=True)
        return x * self.sigmoid(self.mlp(avg) + self.mlp(mx))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CSAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ViTSelfAttention2D(nn.Module):
    def __init__(self, ch, heads=4, patch_pool=4):
        super().__init__()
        self.patch_pool = patch_pool
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, ch*2), nn.GELU(), nn.Linear(ch*2, ch))
    def forward(self, x):
        b, c, h, w = x.shape
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.patch_pool, self.patch_pool))
        seq = pooled.flatten(2).transpose(1, 2)
        seqn = self.norm(seq)
        attn_out, _ = self.attn(seqn, seqn, seqn)
        seq = seq + attn_out
        seq = seq + self.ffn(seq)
        map_out = seq.transpose(1, 2).reshape(b, c, self.patch_pool, self.patch_pool)
        map_out = torch.nn.functional.interpolate(map_out, size=(h, w), mode="bilinear", align_corners=False)
        return x + map_out
