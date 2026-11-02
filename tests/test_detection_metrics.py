import numpy as np

from src.evaluation import detection_metrics as dm
from src.evaluation.bootstrap import bootstrap_ci


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]])
    assert dm.iou_xywh(a, a)[0, 0] == 1.0
    assert dm.iou_xywh(a, np.array([[20, 20, 5, 5]]))[0, 0] == 0.0
    assert abs(dm.iou_xywh(a, np.array([[5, 0, 10, 10]]))[0, 0] - 50 / 150) < 1e-9


def test_match_counts_threshold_and_one_to_one():
    gt = [[0, 0, 10, 10], [50, 50, 10, 10]]
    preds = [[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 10, 10]]  # second pred duplicates first gt
    scores = [0.9, 0.8, 0.3]
    assert dm.match_counts(gt, preds, scores, tau=0.1) == (2, 1, 0)
    assert dm.match_counts(gt, preds, scores, tau=0.5) == (1, 1, 1)  # third pred filtered out
    assert dm.match_counts([], preds, scores, tau=0.1) == (0, 3, 0)


def test_prf_pooled():
    p, r, f = dm.prf(np.array([[2, 1, 0], [1, 0, 1]]))
    assert abs(p - 3 / 4) < 1e-9 and abs(r - 3 / 4) < 1e-9 and abs(f - 0.75) < 1e-9
    assert dm.prf(np.zeros((2, 3))) == (0.0, 0.0, 0.0)


def test_select_tau_prefers_f1_and_largest_tie():
    gt = {1: np.array([[0, 0, 10, 10]])}
    pred = {1: {"boxes": np.array([[0, 0, 10, 10], [100, 100, 5, 5]]), "scores": np.array([0.9, 0.2])}}
    grid = np.array([0.1, 0.3, 0.5, 0.95])
    tau, f1 = dm.select_tau(gt, pred, grid)
    assert f1 == 1.0 and tau == 0.5  # 0.3 and 0.5 tie on F1 (removing the FP); the largest wins; 0.95 loses the TP


def test_bootstrap_is_reproducible_and_contains_estimate():
    counts = np.array([[1, 0, 0]] * 8 + [[0, 1, 1]] * 2, float)
    f = lambda c: dm.prf(c)[2]
    a = bootstrap_ci(counts, f, n=200, seed=1)
    b = bootstrap_ci(counts, f, n=200, seed=1)
    assert a == b and a[0] <= f(counts) <= a[1]


def test_coco_ap_perfect(tmp_path):
    from src import common

    gt = {"images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}],
          "categories": [{"id": 1, "name": "tooth"}]}
    common.write_json(tmp_path / "gt.json", gt)
    ap, ap50 = dm.coco_ap(tmp_path / "gt.json", [{"image_id": "a", "bbox": [10, 10, 30, 30], "score": 0.9}], {"a": 1})
    assert ap > 0.99 and ap50 > 0.99
    assert dm.coco_ap(tmp_path / "gt.json", [], {"a": 1}) == (0.0, 0.0)
