"""Segmentation metrics shared by every model (T013)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _bin(m) -> np.ndarray:
    return np.asarray(m) > 0


def confusion(pred, gt) -> tuple[int, int, int, int]:
    p, g = _bin(pred), _bin(gt)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    return tp, fp, fn, int(p.size - tp - fp - fn)


def overlap_metrics(pred, gt) -> dict:
    """Dice, IoU, Precision, Recall for one image. Both empty -> all 1.0; one empty -> 0 where undefined."""
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    return {
        "dice": 2 * tp / (2 * tp + fp + fn),
        "iou": tp / (tp + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def boundary(mask) -> np.ndarray:
    m = _bin(mask)
    return m & ~ndi.binary_erosion(m, border_value=0)


def _surface_distances(pred, gt) -> tuple[np.ndarray, np.ndarray]:
    bp, bg = boundary(pred), boundary(gt)
    dg = ndi.distance_transform_edt(~bg)
    dp = ndi.distance_transform_edt(~bp)
    return dg[bp], dp[bg]  # pred->gt, gt->pred


def hd95(pred, gt) -> float:
    """Symmetric 95th-percentile Hausdorff distance in pixels; NaN if either mask is empty."""
    if not _bin(pred).any() or not _bin(gt).any():
        return float("nan")
    d_pg, d_gp = _surface_distances(pred, gt)
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def boundary_f1(pred, gt, tol_px: float = 2.0) -> float:
    """Boundary F1 with a pixel tolerance; NaN if either mask is empty."""
    if not _bin(pred).any() or not _bin(gt).any():
        return float("nan")
    d_pg, d_gp = _surface_distances(pred, gt)
    prec = float((d_pg <= tol_px).mean())
    rec = float((d_gp <= tol_px).mean())
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def image_metrics(pred, gt, tol_px: float = 2.0) -> dict:
    m = overlap_metrics(pred, gt)
    m["hd95_px"] = hd95(pred, gt)
    m["boundary_f1"] = boundary_f1(pred, gt, tol_px)
    return m


def aggregate(per_image: dict[str, dict]) -> dict:
    """Mean over images; NaN HD95/BF1 images are excluded and counted (n_excluded_empty)."""
    keys = ["dice", "iou", "precision", "recall", "hd95_px", "boundary_f1"]
    out = {}
    for k in keys:
        v = np.asarray([d[k] for d in per_image.values()], float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    out["n_excluded_empty"] = int(sum(1 for d in per_image.values() if not np.isfinite(d["hd95_px"])))
    out["n_images"] = len(per_image)
    return out
