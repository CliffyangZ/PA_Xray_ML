import pytest
import torch

from src.models import UNET_ARCHS, build_unet


@pytest.mark.parametrize("arch", list(UNET_ARCHS))
@pytest.mark.parametrize("size", [64, 72])  # 72 is not divisible by 16: exercises size-matching upsampling
def test_forward_shape(arch, size):
    m = build_unet({"arch": arch, "base_channels": 8, "depth": 3})
    y = m(torch.randn(2, 1, size, size))
    assert y.shape == (2, 1, size, size)


def test_param_counts_logged(capsys):
    counts = {a: sum(p.numel() for p in build_unet({"arch": a, "base_channels": 8, "depth": 3}).parameters()) for a in UNET_ARCHS}
    print(counts)
    assert counts["unetpp"] > counts["unet"] and counts["attention_unet"] > counts["unet"]
