"""Result-file helpers following contracts/result-schemas.md (T015)."""
from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np

from src import common

SEG_KEYS = ["dice", "iou", "precision", "recall", "hd95_px", "boundary_f1"]


def aggregate_seeds(per_seed: list[dict]) -> dict:
    """mean/std (population-free: ddof=1 if n>1) over seed-level metrics dicts."""
    out = {"n_seeds": len(per_seed), "seeds": [m["seed"] for m in per_seed]}
    for k in SEG_KEYS:
        v = np.asarray([m[k] for m in per_seed], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (e.g. empty predictions) -> NaN
            out[k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else None}
    return out


def update_seg_summary(model: str) -> Path | None:
    root = common.results_root() / "segmentation" / model
    files = sorted(root.glob("seed_*/metrics.json"))
    if not files:
        return None
    per_seed = [common.read_json(f) for f in files]
    summ = {"model": model, **aggregate_seeds(per_seed), "cross_task": per_seed[0].get("cross_task", False),
            "comparable_with_box_only": per_seed[0].get("comparable_with_box_only", True),
            "bf1_tolerance_px": per_seed[0].get("bf1_tolerance_px")}
    common.write_json(root / "summary.json", summ)
    return root / "summary.json"
