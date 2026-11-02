"""Cross-task evaluation (T043): one set of instance predictions scored on both tasks."""
from __future__ import annotations

import cv2
import numpy as np

from src import common
from src.evaluation import detection_metrics as dm
from src.evaluation import seg_metrics
from src.evaluation.io import update_seg_summary
from src.evaluation.merge_instances import merge_instances

INSTANCE_NOTE = ("instance-segmentation model, scored on both tasks; trained/tested on the segmentation dataset, "
                 "not directly comparable with box-only rows")


def _seg_root():
    return common.data_prep_root() / "segmentation"


def evaluate_segmentation_run(model: str, seed: int) -> dict:
    """Segmentation metrics for one (model, seed). Handles U-Net family (saved masks) and instance models."""
    family = common.MODEL_REGISTRY[model]["family"]
    ev = common.data_cfg()["evaluation"]
    run = common.seg_run_dir(model, seed)
    test_dir = _seg_root() / "test"
    tau = None
    if family == "unet":
        loader = lambda stem, shape: (cv2.imread(str(run / "pred_masks" / "test" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 0)
    else:
        gt_v = dm.load_gt(_seg_root() / "val" / "gt.json")
        pv = common.read_json(run / "predictions_val.json")
        tau, _ = dm.select_tau(gt_v[2], dm.group_predictions(pv, gt_v[1], ev["min_score"]), dm.tau_grid(), ev["iou_threshold"])
        pt = common.read_json(run / "predictions_test.json")
        by_img: dict[str, list] = {}
        for r in pt:
            by_img.setdefault(str(r["image_id"]), []).append(r)
        loader = lambda stem, shape: merge_instances(by_img.get(stem, []), tau, shape)

    per_image = {}
    pm_dir = run / "pred_masks" / "test"
    pm_dir.mkdir(parents=True, exist_ok=True)
    for gp in sorted((test_dir / "masks").glob("*.png")):
        gt = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
        pred = loader(gp.stem, gt.shape)
        if family != "unet":
            cv2.imwrite(str(pm_dir / gp.name), pred.astype(np.uint8))
        per_image[gp.stem] = seg_metrics.image_metrics(pred, gt, ev["boundary_tolerance_px"])
    agg = seg_metrics.aggregate(per_image)
    cross = model in common.CROSS_TASK_MODELS
    metrics = {"model": model, "split": "test", "seed": seed, **agg, "per_image": per_image,
               "bf1_tolerance_px": ev["boundary_tolerance_px"], "tau": tau, "tau_selected_on": "val" if tau is not None else None,
               "cross_task": cross, "comparable_with_box_only": not cross,
               "config_hash": common.stable_hash(common.model_cfg(model)), "data_hash": common.data_hash()}
    common.write_json(run / "metrics.json", metrics)
    update_seg_summary(model)
    return metrics


def evaluate_detection_cross_task(model: str, seed: int) -> dict:
    """Score the boxes of an instance model on the segmentation dataset with the detection metrics."""
    run = common.seg_run_dir(model, seed)
    seg = _seg_root()
    m = dm.evaluate_detection(seg / "val" / "gt.json", common.read_json(run / "predictions_val.json"),
                              seg / "test" / "gt.json", common.read_json(run / "predictions_test.json"))
    m.update({"model": model, "seed": seed, "cross_task": True, "comparable_with_box_only": False, "note": INSTANCE_NOTE,
              "config_hash": common.stable_hash(common.model_cfg(model)), "data_hash": common.data_hash()})
    out = common.det_run_dir(model)
    common.write_json(out / "metrics.json", m)
    common.write_json(out / "val_threshold.json", {"tau": m["tau"], "rule": "max F1 at IoU 0.5", "split": "val"})
    return m
