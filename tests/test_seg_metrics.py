import numpy as np

from src.evaluation import seg_metrics as sm
from src.evaluation.merge_instances import encode_mask, merge_instances


def square(n=64, y=10, x=10, s=20):
    m = np.zeros((n, n), np.uint8)
    m[y:y + s, x:x + s] = 1
    return m


def test_perfect_prediction():
    g = square()
    r = sm.image_metrics(g, g)
    assert r["dice"] == r["iou"] == r["precision"] == r["recall"] == 1.0
    assert r["hd95_px"] == 0.0 and r["boundary_f1"] == 1.0


def test_known_overlap_values():
    g, p = square(), square(x=20)  # shifted by half: inter=200, each 400
    r = sm.overlap_metrics(p, g)
    assert abs(r["dice"] - 0.5) < 1e-9 and abs(r["iou"] - 1 / 3) < 1e-9
    assert abs(r["precision"] - 0.5) < 1e-9 and abs(r["recall"] - 0.5) < 1e-9


def test_hd95_shift_matches_pixels():
    g, p = square(), square(x=15)
    assert 4.0 <= sm.hd95(p, g) <= 6.0  # 5-px horizontal shift


def test_empty_rules():
    g, e = square(), np.zeros((64, 64), np.uint8)
    assert sm.overlap_metrics(e, e)["dice"] == 1.0
    r = sm.overlap_metrics(e, g)
    assert r["dice"] == 0.0 and r["precision"] == 0.0 and r["recall"] == 0.0
    assert np.isnan(sm.hd95(e, g)) and np.isnan(sm.boundary_f1(e, g))


def test_boundary_f1_tolerance():
    g, p = square(), square(x=11)  # 1-px shift
    assert sm.boundary_f1(p, g, tol_px=2) > 0.9
    assert sm.boundary_f1(square(x=30), g, tol_px=2) < 0.5


def test_aggregate_excludes_nan_and_counts():
    a = sm.image_metrics(square(), square())
    b = sm.image_metrics(np.zeros((64, 64), np.uint8), square())
    agg = sm.aggregate({"a": a, "b": b})
    assert agg["n_excluded_empty"] == 1 and agg["hd95_px"] == 0.0 and abs(agg["dice"] - 0.5) < 1e-9


def test_merge_preserves_union_area_and_threshold():
    a, b = square(s=20), square(y=20, x=20, s=20)  # overlapping instances
    ins = [{"score": 0.9, "segmentation": encode_mask(a)}, {"score": 0.4, "segmentation": encode_mask(b)}]
    full = merge_instances(ins, 0.1, (64, 64))
    assert full.sum() == ((a | b) > 0).sum()
    assert merge_instances(ins, 0.5, (64, 64)).sum() == a.sum()
