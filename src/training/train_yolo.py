"""YOLO detection / segmentation training and prediction export (T021, T035)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src import common
from src.evaluation.merge_instances import encode_mask

# Built-in augmentation is disabled: augmentation is done offline for every model (FR-002c).
AUG_OFF = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
               scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, erasing=0.0, close_mosaic=0,
               augmentations=[])  # [] also disables ultralytics' default Albumentations (Blur/CLAHE/...)


def _task(cfg: dict) -> str:
    return "segment" if cfg["family"] == "yolo_seg" else "detect"


def run_dir_for(model_name: str, seed: int) -> Path:
    cfg = common.model_cfg(model_name)
    return common.seg_run_dir(model_name, seed) if cfg["family"] == "yolo_seg" else common.det_run_dir(model_name)


def data_yaml_for(cfg: dict) -> Path:
    sub = "segmentation" if cfg["family"] == "yolo_seg" else "detection"
    return common.data_prep_root() / sub / "data.yaml"


def _load_model(cfg: dict):
    """Resolve pretrained weights: local file -> ultralytics download -> detection weights into -seg arch."""
    from ultralytics import YOLO

    pre = common.pretrained_dir()
    w = pre / cfg["weights"]
    if w.exists():
        return YOLO(str(w)), f"local:{w.name}"
    try:
        return YOLO(str(w)), f"downloaded:{w.name}"  # ultralytics downloads into model/pretrained/
    except Exception as e:
        fb = cfg.get("fallback_weights")
        if fb and (pre / fb).exists():
            m = YOLO(cfg["weights"].replace(".pt", ".yaml")).load(str(pre / fb))
            return m, f"fallback:{fb} loaded into {cfg['weights'].replace('.pt', '.yaml')} ({type(e).__name__})"
        raise


def train_yolo(model_name: str, seed: int, epochs: int | None = None, fraction: float = 1.0, device=None) -> dict:
    cfg = common.model_cfg(model_name)
    if epochs:
        cfg["epochs"] = epochs
    run = run_dir_for(model_name, seed)
    model, init_note = _load_model(cfg)
    t0 = time.time()
    model.train(
        data=str(data_yaml_for(cfg)), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"], seed=seed,
        deterministic=True, amp=cfg["amp"], patience=cfg["patience"], optimizer=cfg["optimizer"], fraction=fraction,
        project=str(run.parent), name=run.name, exist_ok=True, plots=False, workers=2, device=device, verbose=False,
        **AUG_OFF,
    )
    info = {"model": model_name, "family": cfg["family"], "seed": seed, "config": cfg, "config_hash": common.stable_hash(cfg),
            "data_hash": common.data_hash(), "init": init_note, "aug_off": {k: v for k, v in AUG_OFF.items() if k != "augmentations"},
            "train_fraction": fraction, "train_seconds": time.time() - t0}
    common.write_json(run / "config.json", info)
    export_predictions(model_name, seed, device=device)
    return info


def export_predictions(model_name: str, seed: int, splits=("val", "test"), device=None) -> None:
    """All boxes with score >= min_score (and RLE instance masks for -seg models) per split."""
    from ultralytics import YOLO

    cfg = common.model_cfg(model_name)
    run = run_dir_for(model_name, seed)
    seg = cfg["family"] == "yolo_seg"
    sub = "segmentation" if seg else "detection"
    min_score = common.data_cfg()["evaluation"]["min_score"]
    model = YOLO(str(run / "weights" / "best.pt"))
    for split in splits:
        out = []
        for p in sorted((common.data_prep_root() / sub / split / "images").glob("*.png")):
            r = model.predict(str(p), conf=min_score, iou=0.7, imgsz=cfg["imgsz"], max_det=50, retina_masks=seg,
                              verbose=False, device=device)[0]
            h, w = r.orig_shape
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            masks = (r.masks.data > 0.5).cpu().numpy() if (seg and r.masks is not None) else None  # bool on GPU first: saves memory
            for i in range(len(conf)):
                x1, y1, x2, y2 = xyxy[i]
                rec = {"image_id": p.stem, "category_id": 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                       "score": float(conf[i])}
                if masks is not None:
                    m = masks[i]
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    rec["segmentation"] = encode_mask(m)
                out.append(rec)
        common.write_json(run / f"predictions_{split}.json", out)
