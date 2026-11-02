import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


def channels(base: int, depth: int) -> list[int]:
    return [base * 2 ** i for i in range(depth + 1)]


def up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
