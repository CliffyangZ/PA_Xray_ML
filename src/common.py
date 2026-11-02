"""Shared helpers: paths, config loading, seeding, hashing, model registry."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")

# Display order and metadata for tables (name -> family, task membership).
MODEL_REGISTRY = {
    "yolov8s": {"family": "yolo_det", "display": "YOLOv8s"},
    "yolo11s": {"family": "yolo_det", "display": "YOLO11s"},
    "yolo26s": {"family": "yolo_det", "display": "YOLO26s"},
    "unet": {"family": "unet", "display": "U-Net"},
    "unetpp": {"family": "unet", "display": "U-Net++"},
    "attention_unet": {"family": "unet", "display": "Attention U-Net"},
    "yolov8s-seg": {"family": "yolo_seg", "display": "YOLOv8s-seg"},
    "yolo11s-seg": {"family": "yolo_seg", "display": "YOLO11s-seg"},
    "yolo26s-seg": {"family": "yolo_seg", "display": "YOLO26s-seg"},
    "maskrcnn": {"family": "maskrcnn", "display": "Mask R-CNN"},
}
DET_MODELS = ["yolov8s", "yolo11s", "yolo26s"]
SEG_MODELS = ["unet", "unetpp", "attention_unet", "yolov8s-seg", "yolo11s-seg", "yolo26s-seg", "maskrcnn"]
CROSS_TASK_MODELS = ["yolov8s-seg", "yolo11s-seg", "yolo26s-seg", "maskrcnn"]
INSTANCE_FAMILIES = ("yolo_seg", "maskrcnn")


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def data_cfg() -> dict:
    return load_yaml(REPO / "configs" / "data.yaml")


def model_cfg(name: str) -> dict:
    return load_yaml(REPO / "configs" / f"{name}.yaml")


def _root(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else REPO / p


def results_root() -> Path:
    return _root("CRR_RESULTS_ROOT", data_cfg()["paths"]["results"])


def data_prep_root() -> Path:
    return _root("CRR_DATA_PREP", data_cfg()["paths"]["data_prep"])


def pretrained_dir() -> Path:
    return _root("CRR_PRETRAINED", data_cfg()["paths"]["pretrained"])


def seg_run_dir(model: str, seed: int) -> Path:
    return results_root() / "segmentation" / model / f"seed_{seed}"


def det_run_dir(model: str) -> Path:
    return results_root() / "detection" / model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def data_hash() -> str:
    m = data_prep_root() / "manifest.json"
    return read_json(m)["data_hash"] if m.exists() else "unprepared"


def source_id(stem: str) -> str:
    """'134_aug2' -> '134' (augmented copies inherit the source image id)."""
    return stem.split("_aug")[0]
