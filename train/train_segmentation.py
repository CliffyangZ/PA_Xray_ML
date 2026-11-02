"""T036: train a segmentation / cross-task model."""
import argparse

from _cli import add_common, apply_roots

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["unet", "unetpp", "attention_unet", "yolov8s-seg", "yolo11s-seg", "yolo26s-seg", "maskrcnn"])
ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
ap.add_argument("--limit", type=int, default=None, help="use only N images per split (U-Net / Mask R-CNN smoke tests)")
ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train images (YOLO-seg smoke tests)")
add_common(ap)
a = ap.parse_args()
apply_roots(a)
from src import common  # noqa: E402

fam = common.MODEL_REGISTRY[a.model]["family"]
if fam == "unet":
    from src.training.train_unet import train_unet
    train_unet(a.model, a.seed, a.epochs, a.limit)
elif fam == "yolo_seg":
    from src.training.train_yolo import train_yolo
    train_yolo(a.model, a.seed, a.epochs, a.fraction)
else:
    from src.training.train_maskrcnn import train_maskrcnn
    train_maskrcnn(a.model, a.seed, a.epochs, a.limit)
print("done:", a.model, "seed", a.seed)
