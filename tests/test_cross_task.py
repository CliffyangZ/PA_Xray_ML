import numpy as np

from src import common
from src.evaluation.merge_instances import encode_mask, instances_to_boxes, merge_instances


def test_merge_keeps_union_and_boxes_filtered_by_tau():
    a = np.zeros((40, 40), np.uint8); a[5:15, 5:15] = 1
    b = np.zeros((40, 40), np.uint8); b[10:25, 10:25] = 1
    ins = [{"score": 0.8, "bbox": [5, 5, 10, 10], "segmentation": encode_mask(a)},
           {"score": 0.6, "bbox": [10, 10, 15, 15], "segmentation": encode_mask(b)}]
    assert merge_instances(ins, 0.5, (40, 40)).sum() == ((a | b) > 0).sum()
    assert len(instances_to_boxes(ins, 0.7)) == 1


def test_registry_flags_cross_task_models():
    assert set(common.CROSS_TASK_MODELS) == {"yolov8s-seg", "yolo11s-seg", "yolo26s-seg", "maskrcnn"}
    assert all(common.MODEL_REGISTRY[m]["family"] in common.INSTANCE_FAMILIES for m in common.CROSS_TASK_MODELS)
