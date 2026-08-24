"""Split-conformal calibration engine for ConformalGen (RFC-001 §3.2, §3.4, §3.5).

Three calibrators, all operating on precomputed non-conformity scores (from `src.scores`), so they
are agnostic to the score family (absolute, signed directional, CQR, or the multi-objective
inf-norm score):

  * SplitConformal    — standard split-conformal (marginal 1-alpha coverage).
  * MondrianConformal — group-conditional calibration (per-chromosome / per-stratum).
  * WeightedConformal — covariate-shift-robust calibration via likelihood-ratio weights.

Coverage is checked uniformly through `covered(test_scores) := test_score <= q_hat`, which holds for
ANY conformal score. Directional / joint bound builders are provided for reporting interval widths.
"""
from __future__ import annotations
import numpy as np

from .scores import conformal_quantile


# ─────────────────────────────────────────────────────────────────────────────
# Weighted quantile (Tibshirani et al., 2019) — includes the test-point mass at +inf
# ─────────────────────────────────────────────────────────────────────────────
def weighted_quantile(scores, weights, alpha, test_weight=1.0):
    scores = np.asarray(scores, float); weights = np.asarray(weights, float)
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    total = w.sum() + float(test_weight)
    if total <= 0:
        return np.inf
    cum = np.cumsum(w) / total                       # normalized cumulative mass over cal points
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    return float(s[idx]) if idx < len(s) else np.inf  # +inf when the test mass carries the tail


# ─────────────────────────────────────────────────────────────────────────────
class SplitConformal:
    """Standard split-conformal (RFC-001 §3.2)."""
    def __init__(self, alpha: float):
        self.alpha = float(alpha); self.q = None

    def calibrate(self, cal_scores):
        self.q = conformal_quantile(cal_scores, self.alpha)
        return self

    def covered(self, test_scores):
        return np.asarray(test_scores, float) <= self.q

    # --- bound builders (for reporting widths) ---
    @staticmethod
    def interval_abs(mu, q):      return np.asarray(mu, float) - q, np.asarray(mu, float) + q
    @staticmethod
    def lower_bound(mu, q):       return np.asarray(mu, float) - q          # e.g. L_eff
    @staticmethod
    def upper_bound(mu, q):       return np.asarray(mu, float) + q          # e.g. U_off (on fit scale)
    @staticmethod
    def interval_cqr(q_lo, q_hi, q):
        return np.asarray(q_lo, float) - q, np.asarray(q_hi, float) + q


# ─────────────────────────────────────────────────────────────────────────────
class MondrianConformal:
    """Group-conditional (Mondrian) conformal (RFC-001 §3.5): a separate quantile per group."""
    def __init__(self, alpha: float):
        self.alpha = float(alpha); self.q_by_group = {}; self.q_pooled = None

    def calibrate(self, cal_scores, cal_groups):
        cal_scores = np.asarray(cal_scores, float); cal_groups = np.asarray(cal_groups)
        self.q_pooled = conformal_quantile(cal_scores, self.alpha)
        for g in np.unique(cal_groups):
            self.q_by_group[g] = conformal_quantile(cal_scores[cal_groups == g], self.alpha)
        return self

    def q_for(self, groups):
        return np.array([self.q_by_group.get(g, self.q_pooled) for g in np.asarray(groups)], float)

    def covered(self, test_scores, test_groups):
        return np.asarray(test_scores, float) <= self.q_for(test_groups)


# ─────────────────────────────────────────────────────────────────────────────
class WeightedConformal:
    """Covariate-shift-robust conformal (RFC-001 §3.5) via per-calibration likelihood-ratio weights.
    `test_weights[j]` = w(x_test_j) = dP_test/dP_cal(x_test_j), typically from a domain classifier."""
    def __init__(self, alpha: float):
        self.alpha = float(alpha); self.cal_scores = None; self.cal_weights = None

    def calibrate(self, cal_scores, cal_weights):
        self.cal_scores = np.asarray(cal_scores, float)
        self.cal_weights = np.asarray(cal_weights, float)
        return self

    def q_for(self, test_weights):
        tw = np.asarray(test_weights, float)
        return np.array([weighted_quantile(self.cal_scores, self.cal_weights, self.alpha, w) for w in tw])

    def covered(self, test_scores, test_weights):
        return np.asarray(test_scores, float) <= self.q_for(test_weights)
