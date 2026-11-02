"""Percentile bootstrap over test images (T012)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(per_image: np.ndarray, stat_fn: Callable[[np.ndarray], float], n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """95% CI (default) of `stat_fn` over resampled image rows. `per_image` is (n_images, k)."""
    per_image = np.asarray(per_image)
    rng = np.random.default_rng(seed)
    N = len(per_image)
    if N == 0:
        return (float("nan"), float("nan"))
    stats = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        stats[i] = stat_fn(per_image[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
