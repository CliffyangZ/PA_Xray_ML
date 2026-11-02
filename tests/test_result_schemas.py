"""Schema tests (T027, T040) using synthetic prediction files in a temp results root."""
import json

import cv2
import numpy as np
import pytest

from src import common
from src.evaluation.merge_instances import encode_mask


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("CRR_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("CRR_DATA_PREP", str(tmp_path / "dp"))
    return tmp_path


def _make_gt(split_dir, n=4, seg=False):
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "masks").mkdir(parents=True)
    images, anns = [], []
    for i in range(1, n + 1):
        cv2.imwrite(str(split_dir / "images" / f"{i}.png"), np.zeros((64, 64), np.uint8))
        m = np.zeros((64, 64), np.uint8)
        m[10:30, 10:30] = 1
        cv2.imwrite(str(split_dir / "masks" / f"{i}.png"), m)
        images.append({"id": i, "file_name": f"{i}.png", "width": 64, "height": 64})
        anns.append({"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0})
    common.write_json(split_dir / "gt.json", {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "tooth"}]})


def _preds(n=4, with_mask=False):
    out = []
    m = np.zeros((64, 64), np.uint8)
    m[10:30, 10:30] = 1
    for i in range(1, n + 1):
        r = {"image_id": str(i), "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}
        if with_mask:
            r["segmentation"] = encode_mask(m)
        out.append(r)
        out.append({"image_id": str(i), "category_id": 1, "bbox": [40, 40, 10, 10], "score": 0.01,
                    **({"segmentation": encode_mask(np.zeros((64, 64)))} if with_mask else {})})
    return out


def test_detection_metrics_schema(roots):
    from src.evaluation.run_eval import evaluate_detection_model

    for s in ("val", "test"):
        _make_gt(common.data_prep_root() / "detection" / s)
    run = common.det_run_dir("yolov8s")
    for s in ("val", "test"):
        common.write_json(run / f"predictions_{s}.json", _preds())
    m = evaluate_detection_model("yolov8s")
    for k in ("tau", "ap_50_95", "ap_50", "precision", "recall", "f1", "split", "tau_selected_on", "cross_task", "config_hash", "data_hash"):
        assert k in m
    for k in ("precision", "recall", "f1"):
        assert set(m[k]) == {"value", "ci95"} and len(m[k]["ci95"]) == 2
    assert m["split"] == "test" and m["tau_selected_on"] == "val" and m["cross_task"] is False
    assert m["f1"]["value"] == 1.0 and m["ap_50"] > 0.99
    assert json.loads((run / "val_threshold.json").read_text())["split"] == "val"


def test_segmentation_cross_task_schema_and_summary(roots):
    from src.evaluation.run_eval import evaluate_detection_model, evaluate_segmentation_model

    for s in ("val", "test"):
        _make_gt(common.data_prep_root() / "segmentation" / s)
    for seed in (42, 43):
        run = common.seg_run_dir("maskrcnn", seed)
        for s in ("val", "test"):
            common.write_json(run / f"predictions_{s}.json", _preds(with_mask=True))
        m = evaluate_segmentation_model("maskrcnn", seed)
    for k in ("dice", "iou", "precision", "recall", "hd95_px", "boundary_f1", "n_excluded_empty", "bf1_tolerance_px", "cross_task"):
        assert k in m
    assert m["cross_task"] is True and m["comparable_with_box_only"] is False and m["dice"] > 0.99
    summ = common.read_json(common.results_root() / "segmentation" / "maskrcnn" / "summary.json")
    assert summ["n_seeds"] == 2 and set(summ["dice"]) == {"mean", "std"}
    det = common.read_json(common.det_run_dir("maskrcnn") / "metrics.json")  # cross-task detection row (first seed)
    assert det["cross_task"] is True and det["tau_selected_on"] == "val"
    # tables carry the dagger marker and footnote
    from src.reporting import build_tables as bt

    t = bt.write_tables()
    assert t["detection"][0][0].endswith(bt.DAGGER) and t["segmentation"][0][0].endswith(bt.DAGGER)
    assert "not directly comparable" in (common.results_root() / "detection" / "detection_table.md").read_text()


def test_unet_eval_uses_saved_masks(roots):
    from src.evaluation.run_eval import evaluate_segmentation_model

    _make_gt(common.data_prep_root() / "segmentation" / "test")
    pm = common.seg_run_dir("unet", 42) / "pred_masks" / "test"
    pm.mkdir(parents=True)
    for p in (common.data_prep_root() / "segmentation" / "test" / "masks").glob("*.png"):
        cv2.imwrite(str(pm / p.name), cv2.imread(str(p), 0))
    m = evaluate_segmentation_model("unet", 42)
    assert m["dice"] == 1.0 and m["cross_task"] is False and m["tau"] is None
