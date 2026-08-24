"""Bootstrap confidence intervals for coverage / width / paired deltas (Phase-2 statistical rigor)."""
from __future__ import annotations
import numpy as np


def bootstrap_ci(values, B=2000, level=0.95, seed=0, stat=np.mean):
    """Percentile bootstrap CI for a 1-D sample statistic (default: mean)."""
    v = np.asarray(values, float)
    if v.size == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(B, v.size))
    boot = stat(v[idx], axis=1)
    a = (1 - level) / 2
    return {"estimate": float(stat(v)), "lo": float(np.quantile(boot, a)),
            "hi": float(np.quantile(boot, 1 - a)), "n": int(v.size)}


def paired_delta_ci(a, b, B=2000, level=0.95, seed=0):
    """Percentile bootstrap CI for mean(a) - mean(b) with paired resampling (same indices)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    assert a.shape == b.shape, "paired arrays must match"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lv = (1 - level) / 2
    return {"delta": float(a.mean() - b.mean()), "lo": float(np.quantile(boot, lv)),
            "hi": float(np.quantile(boot, 1 - lv))}
