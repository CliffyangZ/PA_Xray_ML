"""Instance -> foreground merge and box extraction for cross-task models (T014)."""
from __future__ import annotations

import numpy as np


def decode_rle(rle: dict) -> np.ndarray:
    from pycocotools import mask as mu

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode()
    return mu.decode(r).astype(np.uint8)


def encode_mask(mask: np.ndarray) -> dict:
    from pycocotools import mask as mu

    r = mu.encode(np.asfortranarray((np.asarray(mask) > 0).astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def merge_instances(instances: list[dict], tau: float, shape: tuple[int, int]) -> np.ndarray:
    """Union of instance masks with score >= tau into one uint8 (0/1) foreground mask of `shape`."""
    out = np.zeros(shape, np.uint8)
    for ins in instances:
        if ins["score"] >= tau and "segmentation" in ins:
            out |= decode_rle(ins["segmentation"])
    return out


def instances_to_boxes(instances: list[dict], tau: float) -> list[dict]:
    return [{"bbox": i["bbox"], "score": i["score"]} for i in instances if i["score"] >= tau]
