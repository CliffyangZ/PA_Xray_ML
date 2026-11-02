"""Classic U-Net (T028)."""
import torch
import torch.nn as nn

from src.models.blocks import DoubleConv, channels, up


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        ch = channels(base_channels, depth)
        self.enc = nn.ModuleList([DoubleConv(in_channels if i == 0 else ch[i - 1], ch[i]) for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)
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
            x = self.dec[i](torch.cat([skips[i], up(x, skips[i])], 1))
        return self.head(x)
