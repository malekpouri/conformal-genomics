#!/usr/bin/env python3
"""
ConformalGen — Phase 2 tests for non-conformity scores and quantile behaviour.

Pure-function checks on dummy batches (always run) plus integration checks on D_tr using the fitted
predictors if `models/predictors.joblib` is present. CPU-only, memory-light.

Run:  python tests/test_scores.py   |   pytest tests/test_scores.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.scores import (score_abs, score_signed_lower, score_signed_upper, score_cqr,   # noqa: E402
                        score_mo, conformal_quantile, pinball_loss)


# ---------------- pure score-function checks (dummy batches) ------------------
def test_score_values_and_finiteness():
    y = np.array([1.0, 2.0, 3.0]); mu = np.array([1.5, 1.0, 5.0])
    assert np.allclose(score_abs(y, mu), [0.5, 1.0, 2.0])
    assert np.allclose(score_signed_lower(y, mu), mu - y)             # mu - y
    assert np.allclose(score_signed_upper(y, mu), y - mu)            # y - mu
    for s in (score_abs(y, mu), score_signed_lower(y, mu), score_signed_upper(y, mu)):
        assert np.all(np.isfinite(s))


def test_cqr_sign_semantics():
    q_lo = np.array([0.0, 0.0, 0.0]); q_hi = np.array([10.0, 10.0, 10.0])
    y = np.array([5.0, -3.0, 13.0])                                  # inside, below, above
    s = score_cqr(y, q_lo, q_hi)
    assert s[0] <= 0.0                      # inside interval -> non-positive
    assert np.isclose(s[1], 3.0)            # 3 below lower -> +3
    assert np.isclose(s[2], 3.0)            # 3 above upper -> +3
    assert np.all(np.isfinite(s))


def test_score_mo_is_max_of_standardized_terms():
    s = score_mo(y_eff=np.array([40.0]), mu_eff=np.array([50.0]), sigma_eff=np.array([5.0]),
                 y_off=np.array([2.0]), mu_off=np.array([1.0]), sigma_off=np.array([2.0]))
    # eff term = (50-40)/5 = 2.0 ; off term = (2-1)/2 = 0.5 ; max = 2.0
    assert np.isclose(s[0], 2.0) and np.all(np.isfinite(s))


def test_conformal_quantile_rank_and_monotonicity():
    scores = np.arange(1, 101, dtype=float)          # 1..100
    # smaller alpha -> larger (or equal) quantile
    q90 = conformal_quantile(scores, 0.10)
    q95 = conformal_quantile(scores, 0.05)
    assert q95 >= q90
    # exact rank: k = ceil((n+1)(1-alpha)); n=100, alpha=0.1 -> k=ceil(90.9)=91 -> 91st smallest = 91
    assert q90 == 91.0
    # n too small for the level -> +inf
    assert conformal_quantile(np.array([1.0, 2.0]), 0.01) == np.inf


def test_pinball_loss_known_value():
    y = np.array([0.0, 10.0]); q = np.array([5.0, 5.0]); tau = 0.5
    # 0.5*|error| averaged = 0.5*(5+5)/2 = 2.5
    assert np.isclose(pinball_loss(y, q, tau), 2.5)


# ---------------- integration checks on D_tr with fitted predictors ----------
def _load_artifacts():
    import joblib, pandas as pd
    ckpt = PROJECT / "models" / "predictors.joblib"
    if not ckpt.exists():
        return None
    pp = joblib.load(ckpt)
    tr = pd.read_csv(PROJECT / "data" / "splits" / "train.csv")
    return pp, tr


def test_fitted_scores_finite_and_quantiles_monotone():
    art = _load_artifacts()
    if art is None:
        print("  (skip: models/predictors.joblib not found — run scripts/02_fit_predictors.py)")
        return
    pp, tr = art
    seq = tr["seq"].tolist(); y_eff = tr["y_eff"].to_numpy(float); y_off = tr["off_id"].to_numpy(float)
    mu_eff = pp.eff.predict_mu(seq)
    assert np.all(np.isfinite(score_abs(y_eff, mu_eff)))
    assert np.all(np.isfinite(score_signed_lower(y_eff, mu_eff)))
    # monotonicity of the fitted quantile heads: q_lo <= q_hi everywhere (no crossing)
    for a in (0.10, 0.05):
        lo_e, hi_e = pp.eff.predict_quantiles(seq, a)
        lo_o, hi_o = pp.off.predict_quantiles(seq, a)
        assert np.mean(lo_e > hi_e) == 0.0, f"efficacy quantile crossing at alpha={a}"
        assert np.mean(lo_o > hi_o) == 0.0, f"off-target quantile crossing at alpha={a}"
        assert np.all(np.isfinite(score_cqr(y_eff, lo_e, hi_e)))


def _main():
    tests = [test_score_values_and_finiteness, test_cqr_sign_semantics,
             test_score_mo_is_max_of_standardized_terms, test_conformal_quantile_rank_and_monotonicity,
             test_pinball_loss_known_value, test_fitted_scores_finite_and_quantiles_monotone]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed (CPU-only; 0 GB VRAM).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
