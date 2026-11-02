"""T022: train a box-only YOLO detection model."""
import argparse

from _cli import add_common, apply_roots

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["yolov8s", "yolo11s", "yolo26s"])
ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train images (smoke tests)")
add_common(ap)
a = ap.parse_args()
apply_roots(a)
from src.training.train_yolo import train_yolo  # noqa: E402

info = train_yolo(a.model, a.seed, a.epochs, a.fraction)
print("done:", a.model, "init:", info["init"])
