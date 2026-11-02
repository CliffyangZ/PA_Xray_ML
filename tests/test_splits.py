import cv2
import numpy as np
import pytest

from src.data import augment, convert, splits
from src.data.prepare import apply_clahe
from src.common import data_cfg


def _touch(root, split, stem):
    d = root / split / "images"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{stem}.png"), np.zeros((8, 8), np.uint8))


def test_clean_split_passes(tmp_path):
    for s, stems in {"train": ["1", "1_aug0", "2"], "val": ["3"], "test": ["4"]}.items():
        for st in stems:
            _touch(tmp_path, s, st)
    assert splits.check_prepared(tmp_path)["ok"]


def test_leakage_detected(tmp_path):
    _touch(tmp_path, "train", "1_aug0")
    _touch(tmp_path, "test", "1")
    with pytest.raises(AssertionError, match="leakage"):
        splits.check_prepared(tmp_path)


def test_augmented_copy_outside_train_detected(tmp_path):
    _touch(tmp_path, "train", "1")
    _touch(tmp_path, "val", "2_aug0")
    with pytest.raises(AssertionError, match="augmented"):
        splits.check_prepared(tmp_path)


def test_clahe_changes_contrast_keeps_shape():
    img = (np.random.default_rng(0).normal(100, 5, (64, 64))).clip(0, 255).astype(np.uint8)
    out = apply_clahe(img, 2.0, (8, 8))
    assert out.shape == img.shape and out.dtype == np.uint8 and out.std() > img.std()


def test_augmentation_is_deterministic_and_flips_only_horizontally():
    cfg = data_cfg()
    img = np.zeros((100, 100), np.uint8)
    img[40:60, 10:30] = 255
    mask = np.zeros_like(img)
    mask[40:60, 10:30] = 1
    a = augment.augment_segmentation(img, [mask], cfg, "k")
    b = augment.augment_segmentation(img, [mask], cfg, "k")
    assert len(a) == cfg["augmentation"]["copies"]
    assert all((x[0] == y[0]).all() for x, y in zip(a, b))
    # tooth stays in the same vertical band up to rotation (<= 15 deg): vertical flip would move it
    for _, ms in a:
        for m in ms:
            ys, _ = np.where(m > 0)
            assert 25 < ys.mean() < 75


def test_detection_boxes_follow_flip():
    cfg = data_cfg()
    cfg["augmentation"].update(rotation_p=0.0, gaussian_noise_p=0.0, hflip_p=1.0, copies=1)
    img = np.zeros((100, 100), np.uint8)
    (aimg, boxes), = augment.augment_detection(img, [(0, 0.2, 0.5, 0.2, 0.2)], cfg, "k")
    assert abs(boxes[0][1] - 0.8) < 1e-6


def test_polygon_roundtrip_and_yolo_seg_label(tmp_path):
    m = np.zeros((50, 60), np.uint8)
    m[10:30, 20:40] = 1
    polys = convert.mask_to_polygons(m)
    back = convert.polygons_to_mask(polys, 50, 60)
    assert (back & m).sum() / m.sum() > 0.85
    f = tmp_path / "a.txt"
    convert.write_yolo_seg_labels(f, [m])
    vals = f.read_text().split()
    assert vals[0] == "0" and all(0 <= float(v) <= 1 for v in vals[1:])
    assert convert.mask_to_bbox(m) == [20.0, 10.0, 20.0, 20.0]
