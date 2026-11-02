from src.models.attention_unet import AttentionUNet
from src.models.unet import UNet
from src.models.unet_pp import UNetPP

UNET_ARCHS = {"unet": UNet, "unetpp": UNetPP, "attention_unet": AttentionUNet}


def build_unet(cfg: dict):
    return UNET_ARCHS[cfg["arch"]](cfg.get("in_channels", 1), 1, cfg.get("base_channels", 32), cfg.get("depth", 4))
