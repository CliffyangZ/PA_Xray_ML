"""Detection metrics shared by every model (T011): COCO AP, P/R/F1 at IoU 0.5, tau selection."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

from src import common
from src.evaluation.bootstrap import bootstrap_ci


# ------------------------------------------------------------------ geometry
def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) boxes in xywh."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_counts(gt_boxes, pred_boxes, scores, tau: float, iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy score-ordered one-to-one matching; returns (tp, fp, fn) for one image at threshold tau."""
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float)
    keep = scores >= tau
    pred_boxes, scores = pred_boxes[keep], scores[keep]
    order = np.argsort(-scores, kind="stable")
    pred_boxes = pred_boxes[order]
    ious = iou_xywh(pred_boxes, gt_boxes)
    used = set()
    tp = 0
    for i in range(len(pred_boxes)):
        if ious.shape[1] == 0:
            break
        cand = [(ious[i, j], j) for j in range(ious.shape[1]) if j not in used and ious[i, j] >= iou_thr]
        if cand:
            used.add(max(cand)[1])
            tp += 1
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


def prf(counts: np.ndarray) -> tuple[float, float, float]:
    """Pooled (micro) precision, recall, F1 from (n_images, 3) [tp, fp, fn] rows."""
    tp, fp, fn = np.asarray(counts, float).reshape(-1, 3).sum(0)
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return float(p), float(r), float(f)


# ------------------------------------------------------------------ data access
def load_gt(gt_path: Path) -> tuple[dict, dict[str, int], dict[int, np.ndarray]]:
    gt = common.read_json(gt_path)
    stem2id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}
    boxes = {im["id"]: [] for im in gt["images"]}
    for a in gt["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])
    return gt, stem2id, {k: np.asarray(v, float).reshape(-1, 4) for k, v in boxes.items()}


def group_predictions(preds: list[dict], stem2id: dict[str, int], min_score: float = 0.0) -> dict[int, dict]:
    out = {i: {"boxes": [], "scores": []} for i in stem2id.values()}
    for p in preds:
        if p["score"] < min_score:
            continue
        iid = stem2id[str(p["image_id"])]
        out[iid]["boxes"].append(p["bbox"])
        out[iid]["scores"].append(p["score"])
    return {k: {"boxes": np.asarray(v["boxes"], float).reshape(-1, 4), "scores": np.asarray(v["scores"], float)} for k, v in out.items()}


def per_image_counts(gt_boxes: dict[int, np.ndarray], pred: dict[int, dict], tau: float, iou_thr: float = 0.5) -> np.ndarray:
    ids = sorted(gt_boxes)
    return np.asarray([match_counts(gt_boxes[i], pred[i]["boxes"], pred[i]["scores"], tau, iou_thr) for i in ids], float).reshape(-1, 3)


def tau_grid(cfg: dict | None = None) -> np.ndarray:
    g = (cfg or common.data_cfg())["evaluation"]["tau_grid"]
    return np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 6)


def select_tau(gt_boxes, pred, grid: np.ndarray, iou_thr: float = 0.5) -> tuple[float, float]:
    """tau maximizing F1 on the validation split; ties -> the largest tau. Returns (tau, f1)."""
    best_f, best_t = -1.0, float(grid[0])
    for t in grid:
        f = prf(per_image_counts(gt_boxes, pred, t, iou_thr))[2]
        if f >= best_f - 1e-12:
            best_f, best_t = (f if f > best_f else best_f), float(t)
    return best_t, float(best_f)


# ------------------------------------------------------------------ COCO AP
def coco_ap(gt_path: Path, preds: list[dict], stem2id: dict[str, int], min_score: float = 0.001) -> tuple[float, float]:
    """(AP@[.5:.95], AP@.5) via pycocotools. Returns zeros if there are no predictions."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dets = [{"image_id": stem2id[str(p["image_id"])], "category_id": 1, "bbox": list(map(float, p["bbox"])),
             "score": float(p["score"])} for p in preds if p["score"] >= min_score]
    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(gt_path))
        res = coco.loadRes(dets)
        ev = COCOeval(coco, res, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(max(ev.stats[0], 0.0)), float(max(ev.stats[1], 0.0))


# ------------------------------------------------------------------ full evaluation
def evaluate_detection(gt_val: Path, preds_val: list[dict], gt_test: Path, preds_test: list[dict]) -> dict:
    """tau chosen on val only, then fixed for test. Returns the metrics dict (without bookkeeping fields)."""
    cfg = common.data_cfg()["evaluation"]
    _, stem_v, gtb_v = load_gt(gt_val)
    _, stem_t, gtb_t = load_gt(gt_test)
    tau, val_f1 = select_tau(gtb_v, group_predictions(preds_val, stem_v, cfg["min_score"]), tau_grid(), cfg["iou_threshold"])
    counts = per_image_counts(gtb_t, group_predictions(preds_test, stem_t, cfg["min_score"]), tau, cfg["iou_threshold"])
    p, r, f = prf(counts)
    n, seed = cfg["bootstrap_resamples"], cfg["bootstrap_seed"]
    ci = {k: bootstrap_ci(counts, lambda c, i=i: prf(c)[i], n, seed) for i, k in enumerate(["precision", "recall", "f1"])}
    ap, ap50 = coco_ap(gt_test, preds_test, stem_t, cfg["min_score"])
    return {
        "tau": tau, "tau_selected_on": "val", "val_f1_at_tau": val_f1, "split": "test",
        "ap_50_95": ap, "ap_50": ap50,
        "precision": {"value": p, "ci95": list(ci["precision"])},
        "recall": {"value": r, "ci95": list(ci["recall"])},
        "f1": {"value": f, "ci95": list(ci["f1"])},
        "n_images": int(len(counts)), "tp_fp_fn": [int(x) for x in counts.sum(0)],
    }
