"""Interval baselines for benchmarking against split-conformal (Phase-2 rigor).

All operate on already-computed predictions so they are model-shared with ConformalGen:
  * uncalibrated quantile regression  — raw q_lo/q_hi heads, no conformal correction
  * parametric Gaussian               — mu +/- z * sigma
  * standard split-conformal          — absolute-residual conformal (reference)
Each returns coverage and mean width on a test set. Off-target lives on the log1p scale; widths are
reported on the count scale via expm1.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

from .scores import conformal_quantile


def _cov_width(y, lo, hi, count_scale=False):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); y = np.asarray(y, float)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    if count_scale:
        width = float(np.mean(np.expm1(hi) - np.expm1(np.maximum(lo, 0.0))))
    else:
        width = float(np.mean(hi - lo))
    return cov, width


def uncalibrated_qr(y_te, qlo_te, qhi_te, count_scale=False):
    """Raw quantile-head interval [q_lo, q_hi] with no conformal calibration."""
    cov, width = _cov_width(y_te, qlo_te, qhi_te, count_scale)
    return {"method": "uncalibrated_QR", "coverage": cov, "width": width}


def gaussian_interval(y_te, mu_te, sigma_te, alpha, count_scale=False):
    """Parametric Gaussian interval mu +/- z_{1-alpha/2} * sigma."""
    z = float(norm.ppf(1 - alpha / 2))
    lo, hi = mu_te - z * sigma_te, mu_te + z * sigma_te
    cov, width = _cov_width(y_te, lo, hi, count_scale)
    return {"method": "parametric_Gaussian", "coverage": cov, "width": width}


def standard_split_cp(y_cal, mu_cal, y_te, mu_te, alpha, count_scale=False):
    """Absolute-residual split-conformal interval mu +/- q_hat."""
    q = conformal_quantile(np.abs(np.asarray(y_cal, float) - np.asarray(mu_cal, float)), alpha)
    lo, hi = mu_te - q, mu_te + q
    cov, width = _cov_width(y_te, lo, hi, count_scale)
    return {"method": "standard_split_CP", "coverage": cov, "width": width, "q": float(q)}
