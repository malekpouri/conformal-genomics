"""Base property predictors for ConformalGen (RFC-001 §3, Phase 2).

Lightweight, CPU-only (0 GB VRAM) gradient-boosted regressors:
  * point predictors  mu_eff(x), mu_off(x)                     (RFC scores a, b, d)
  * quantile heads    q_{alpha/2}(x), q_{1-alpha/2}(x)          (RFC score c: CQR)
  * conditional scale sigma_eff(x), sigma_off(x)               (RFC score d normaliser)

Off-target risk is a heavy-tailed count; its point/quantile predictors are fit on the
log1p scale and mapped back with expm1 (a monotone transform, so quantiles are preserved).
Conformal coverage is model-agnostic: predictor choice affects interval *width*, not validity.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .featurize import featurize

RNG = 42
_HGB = dict(max_iter=300, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0, random_state=RNG)


def _point(loss="squared_error"):
    return HistGradientBoostingRegressor(loss=loss, **_HGB)


def _quantile(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **_HGB)


@dataclass
class ObjectivePredictor:
    """Point + (per-alpha) quantile heads + conditional scale for one objective."""
    name: str
    log_scale: bool = False                 # fit on log1p (for the off-target count)
    alphas: tuple = (0.10, 0.05)
    mu: object = None
    scale: object = None                    # predicts |residual| (conditional MAD-ish)
    q_lo: dict = field(default_factory=dict)  # alpha -> model for level alpha/2
    q_hi: dict = field(default_factory=dict)  # alpha -> model for level 1-alpha/2

    # --- transforms ---
    def _fwd(self, y):  return np.log1p(y) if self.log_scale else np.asarray(y, float)
    def _inv(self, z):  return np.expm1(z) if self.log_scale else z

    def fit(self, seqs, y):
        X = featurize(seqs); z = self._fwd(y)
        self.mu = _point().fit(X, z)
        resid = np.abs(z - self.mu.predict(X))
        self.scale = _point().fit(X, resid)   # conditional scale on the (fwd) scale
        for a in self.alphas:
            self.q_lo[a] = _quantile(a / 2).fit(X, z)
            self.q_hi[a] = _quantile(1 - a / 2).fit(X, z)
        return self

    # --- prediction (all returned on the ORIGINAL property scale) ---
    def predict_mu(self, seqs):
        return self._inv(self.mu.predict(featurize(seqs)))

    def predict_quantiles(self, seqs, alpha):
        X = featurize(seqs)
        lo = self._inv(self.q_lo[alpha].predict(X))
        hi = self._inv(self.q_hi[alpha].predict(X))
        return lo, hi

    def predict_scale(self, seqs, floor=1e-6):
        # scale is fit on the fwd space; keep it there (used to standardise fwd-space residuals).
        s = self.scale.predict(featurize(seqs))
        return np.maximum(s, floor)


@dataclass
class PropertyPredictors:
    """Bundle of both objectives (efficacy, off-target)."""
    eff: ObjectivePredictor = None
    off: ObjectivePredictor = None

    def fit(self, seqs, y_eff, y_off, alphas=(0.10, 0.05)):
        self.eff = ObjectivePredictor("eff", log_scale=False, alphas=alphas).fit(seqs, y_eff)
        self.off = ObjectivePredictor("off", log_scale=True, alphas=alphas).fit(seqs, y_off)
        return self
