"""Evaluation entry points used by train/evaluate.py (T024, T037)."""
from __future__ import annotations

from src import common
from src.evaluation import cross_task
from src.evaluation import detection_metrics as dm


def evaluate_detection_model(model: str, seed: int = 42, repredict: bool = False) -> dict:
    """Detection-task evaluation. Box-only YOLO models use the detection dataset; instance models are cross-task."""
    fam = common.MODEL_REGISTRY[model]["family"]
    if fam in common.INSTANCE_FAMILIES:
        _ensure_predictions(model, seed, repredict)
        return cross_task.evaluate_detection_cross_task(model, seed)
    if fam != "yolo_det":
        raise ValueError(f"{model} has no detection output")
    run = common.det_run_dir(model)
    _ensure_predictions(model, seed, repredict)
    det = common.data_prep_root() / "detection"
    m = dm.evaluate_detection(det / "val" / "gt.json", common.read_json(run / "predictions_val.json"),
                              det / "test" / "gt.json", common.read_json(run / "predictions_test.json"))
    m.update({"model": model, "seed": seed, "cross_task": False, "comparable_with_box_only": True,
              "config_hash": common.stable_hash(common.model_cfg(model)), "data_hash": common.data_hash()})
    common.write_json(run / "metrics.json", m)
    common.write_json(run / "val_threshold.json", {"tau": m["tau"], "rule": "max F1 at IoU 0.5", "split": "val",
                                                   "val_f1": m["val_f1_at_tau"]})
    return m


def evaluate_segmentation_model(model: str, seed: int, repredict: bool = False) -> dict:
    fam = common.MODEL_REGISTRY[model]["family"]
    if fam == "yolo_det":
        raise ValueError(f"{model} is a detection-only model")
    _ensure_predictions(model, seed, repredict)
    m = cross_task.evaluate_segmentation_run(model, seed)
    if fam in common.INSTANCE_FAMILIES and seed == common.data_cfg()["seeds"][0]:
        cross_task.evaluate_detection_cross_task(model, seed)  # detection rows come from the first seed
    return m


def _ensure_predictions(model: str, seed: int, force: bool) -> None:
    fam = common.MODEL_REGISTRY[model]["family"]
    if fam == "yolo_det":
        run = common.det_run_dir(model)
    else:
        run = common.seg_run_dir(model, seed)
    if fam == "unet":
        if force or not (run / "pred_masks" / "test").exists():
            from src.training.train_unet import export_masks
            export_masks(model, seed)
        return
    if force or not ((run / "predictions_val.json").exists() and (run / "predictions_test.json").exists()):
        if fam == "maskrcnn":
            from src.training.train_maskrcnn import export_predictions
        else:
            from src.training.train_yolo import export_predictions
        export_predictions(model, seed)
