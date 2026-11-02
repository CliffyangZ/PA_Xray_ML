"""Attention U-Net: U-Net with additive attention gates on the skip connections (T030)."""
import torch
import torch.nn as nn

from src.models.blocks import DoubleConv, channels, up


class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.wx = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = up(self.wg(g), x) if g.shape[-2:] != x.shape[-2:] else self.wg(g)
        return x * self.psi(self.relu(g + self.wx(x)))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        ch = channels(base_channels, depth)
        self.enc = nn.ModuleList([DoubleConv(in_channels if i == 0 else ch[i - 1], ch[i]) for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.gates = nn.ModuleList([AttentionGate(ch[i + 1], ch[i], max(ch[i] // 2, 1)) for i in range(depth)])
        self.dec = nn.ModuleList([DoubleConv(ch[i + 1] + ch[i], ch[i]) for i in range(depth)])
        self.head = nn.Conv2d(ch[0], out_channels, 1)
        self.depth = depth

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips[-1]
        for i in reversed(range(self.depth)):
            s = self.gates[i](x, skips[i])
            x = self.dec[i](torch.cat([s, up(x, s)], 1))
        return self.head(x)
