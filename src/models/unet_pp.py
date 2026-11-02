"""U-Net++ with nested dense skip pathways, no deep supervision (T029)."""
import torch
import torch.nn as nn

from src.models.blocks import DoubleConv, channels, up


class UNetPP(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        ch = channels(base_channels, depth)
        self.depth = depth
        self.pool = nn.MaxPool2d(2)
        # node X[i][j]: level i, j-th node along the skip pathway (j=0 is the encoder)
        self.nodes = nn.ModuleDict()
        for j in range(depth + 1):
            for i in range(depth + 1 - j):
                cin = (in_channels if i == 0 else ch[i - 1]) if j == 0 else ch[i] * j + ch[i + 1]
                self.nodes[f"{i}_{j}"] = DoubleConv(cin, ch[i])
        self.head = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x):
        X = {}
        for j in range(self.depth + 1):
            for i in range(self.depth + 1 - j):
                if j == 0:
                    X[(i, 0)] = self.nodes[f"{i}_0"](x if i == 0 else self.pool(X[(i - 1, 0)]))
                else:
                    prev = [X[(i, k)] for k in range(j)]
                    X[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat(prev + [up(X[(i + 1, j - 1)], prev[0])], 1))
        return self.head(X[(0, self.depth)])
