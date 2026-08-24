"""Non-conformity scores for ConformalGen (RFC-001 §3.1) + calibration helpers.

Pure, model-agnostic array functions: given predictions and labels they return per-example
non-conformity scores. A score is combined with the split-conformal quantile (Section 3.2) in
Phase 3 to form prediction sets with finite-sample (1-alpha) coverage.

Scale convention: each score's inputs must be on a mutually consistent scale. For the heavy-tailed
off-target objective the caller may pass log1p-space predictions/labels (the off predictor is fit
there); a monotone transform preserves quantiles, so coverage is unaffected.
"""
from __future__ import annotations
import numpy as np


# --- RFC score (a): absolute residual (two-sided, per objective) ---------------
def score_abs(y, mu):
    return np.abs(np.asarray(y, float) - np.asarray(mu, float))


# --- RFC score (b): signed directional scores --------------------------------
def score_signed_lower(y, mu):
    """For a guaranteed LOWER bound (e.g. efficacy >= L): s = mu - y."""
    return np.asarray(mu, float) - np.asarray(y, float)


def score_signed_upper(y, mu):
    """For a guaranteed UPPER bound (e.g. off-target <= U): s = y - mu."""
    return np.asarray(y, float) - np.asarray(mu, float)


# --- RFC score (c): conformalized quantile regression (CQR) -------------------
def score_cqr(y, q_lo, q_hi):
    """s = max(q_lo - y, y - q_hi); <=0 inside [q_lo, q_hi], >0 outside."""
    y = np.asarray(y, float)
    return np.maximum(np.asarray(q_lo, float) - y, y - np.asarray(q_hi, float))


# --- RFC score (d): multi-objective standardized inf-norm score ---------------
def score_mo(y_eff, mu_eff, sigma_eff, y_off, mu_off, sigma_off):
    """Joint directional score for simultaneous (lower-eff, upper-off) bounds:
        s = max( (mu_eff - y_eff)/sigma_eff ,  (y_off - mu_off)/sigma_off ).
    A single calibrated quantile then yields a hyper-rectangle covering the vector at 1-alpha."""
    eff_term = (np.asarray(mu_eff, float) - np.asarray(y_eff, float)) / np.asarray(sigma_eff, float)
    off_term = (np.asarray(y_off, float) - np.asarray(mu_off, float)) / np.asarray(sigma_off, float)
    return np.maximum(eff_term, off_term)


# --- split-conformal quantile (RFC §3.2) -------------------------------------
def conformal_quantile(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score; +inf if that rank exceeds n
    (i.e. n too small for the requested level). Under exchangeability this q gives
    P[s_test <= q] >= 1-alpha."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return np.inf if k > n else float(s[k - 1])


# --- pinball / quantile loss (for evaluating quantile heads) ------------------
def pinball_loss(y, q_pred, tau):
    y = np.asarray(y, float); q = np.asarray(q_pred, float)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))
