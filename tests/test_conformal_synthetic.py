#!/usr/bin/env python3
"""
ConformalGen — Phase 3 theoretical sanity on synthetic exchangeable data.

For a fixed predictor, split-conformal coverage over the randomness of the calibration/test split
must satisfy  1-alpha <= E[coverage] <= 1-alpha + 1/(n_cal+1)  (RFC-001 §3.2). We verify this by
Monte-Carlo: 1,000 resplits, checking the MEAN empirical coverage lands in the theoretical band
(with a small MC tolerance) for alpha in {0.05, 0.10}, in 1-D (absolute residual) and 2-D
(multi-objective inf-norm). We also check that Mondrian restores per-group coverage under
group-dependent noise, and that weighted conformal restores coverage under covariate shift.

Pure NumPy, CPU-only, no GPU.  Run: python tests/test_conformal_synthetic.py  |  pytest ...
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.scores import conformal_quantile, score_mo                      # noqa: E402
from src.conformal import SplitConformal, MondrianConformal, WeightedConformal, weighted_quantile  # noqa: E402

N_CAL, N_TEST, TRIALS = 200, 1000, 1000
MC_TOL = 0.01
RNG = np.random.default_rng(0)


def _band(alpha, n=N_CAL):
    return 1.0 - alpha, 1.0 - alpha + 1.0 / (n + 1)


def test_coverage_band_1d_absolute():
    """1-D absolute-residual score; fixed predictor => residual == noise."""
    for alpha in (0.10, 0.05):
        lo, hi = _band(alpha)
        cov = []
        for _ in range(TRIALS):
            noise = RNG.standard_normal(N_CAL + N_TEST)      # exchangeable residuals
            s = np.abs(noise)
            q = conformal_quantile(s[:N_CAL], alpha)
            cov.append(np.mean(s[N_CAL:] <= q))
        m = float(np.mean(cov))
        assert m >= lo - MC_TOL, f"1D under-coverage: {m:.4f} < {lo:.4f} (alpha={alpha})"
        assert m <= hi + MC_TOL, f"1D over-coverage beyond band: {m:.4f} > {hi:.4f} (alpha={alpha})"
        print(f"  1D abs   alpha={alpha}: mean coverage={m:.4f}  band=[{lo:.4f},{hi:.4f}]")


def test_coverage_band_2d_joint_mo():
    """2-D joint inf-norm score over two standardized directional residuals."""
    for alpha in (0.10, 0.05):
        lo, hi = _band(alpha)
        cov = []
        for _ in range(TRIALS):
            # correlated standardized directional residuals a=(mu_eff-y_eff)/sig, b=(y_off-mu_off)/sig
            z = RNG.standard_normal((N_CAL + N_TEST, 2))
            z[:, 1] = 0.5 * z[:, 0] + np.sqrt(1 - 0.25) * z[:, 1]     # rho=0.5
            s = np.maximum(z[:, 0], z[:, 1])                          # = score_mo with sigma=1
            q = conformal_quantile(s[:N_CAL], alpha)
            cov.append(np.mean(s[N_CAL:] <= q))                       # joint: a<=q AND b<=q
        m = float(np.mean(cov))
        assert m >= lo - MC_TOL, f"2D under-coverage: {m:.4f} < {lo:.4f} (alpha={alpha})"
        assert m <= hi + MC_TOL, f"2D over-coverage beyond band: {m:.4f} > {hi:.4f} (alpha={alpha})"
        print(f"  2D mo    alpha={alpha}: mean coverage={m:.4f}  band=[{lo:.4f},{hi:.4f}]")


def test_score_mo_matches_manual():
    """score_mo(sigma=1) equals max of the two directional residuals used above."""
    y_eff = np.array([1.0, 2.0]); mu_eff = np.array([2.0, 2.0])
    y_off = np.array([3.0, 1.0]); mu_off = np.array([1.0, 1.0])
    s = score_mo(y_eff, mu_eff, np.ones(2), y_off, mu_off, np.ones(2))
    assert np.allclose(s, np.maximum(mu_eff - y_eff, y_off - mu_off))


def test_mondrian_restores_group_coverage():
    """Under group-dependent noise scale, pooled split under-covers the wide group; Mondrian fixes it."""
    alpha = 0.10; scales = {"A": 1.0, "B": 4.0}; per_group = 400
    pooled_worst, mondrian_worst = [], []
    for _ in range(200):
        groups, scores = [], []
        for g, sc in scales.items():
            groups += [g] * (2 * per_group); scores += list(np.abs(RNG.standard_normal(2 * per_group) * sc))
        groups = np.array(groups); scores = np.array(scores)
        idx = RNG.permutation(len(scores)); half = len(scores) // 2
        cal, te = idx[:half], idx[half:]
        # pooled
        sc_p = SplitConformal(alpha).calibrate(scores[cal])
        cov_p = sc_p.covered(scores[te])
        pooled_worst.append(min(np.mean(cov_p[groups[te] == g]) for g in scales))
        # mondrian
        mc = MondrianConformal(alpha).calibrate(scores[cal], groups[cal])
        cov_m = mc.covered(scores[te], groups[te])
        mondrian_worst.append(min(np.mean(cov_m[groups[te] == g]) for g in scales))
    pw, mw = float(np.mean(pooled_worst)), float(np.mean(mondrian_worst))
    assert mw >= 1 - alpha - 0.02, f"Mondrian worst-group coverage too low: {mw:.4f}"
    assert mw > pw, f"Mondrian ({mw:.4f}) should beat pooled worst-group ({pw:.4f})"
    print(f"  Mondrian worst-group={mw:.4f} vs pooled worst-group={pw:.4f} (target>={1-alpha})")


def test_weighted_restores_coverage_under_shift():
    """Covariate shift: test residuals larger than calibration; weighted conformal restores coverage."""
    alpha = 0.10
    unweighted_cov, weighted_cov = [], []
    for _ in range(200):
        # cal covariate x~N(0,1); test x~N(1,1) (shift). residual scale grows with x: sigma(x)=1+0.5*|x|
        x_cal = RNG.standard_normal(N_CAL); x_te = RNG.standard_normal(N_TEST) + 1.0
        s_cal = np.abs(RNG.standard_normal(N_CAL) * (1 + 0.5 * np.abs(x_cal)))
        s_te = np.abs(RNG.standard_normal(N_TEST) * (1 + 0.5 * np.abs(x_te)))
        # unweighted
        q = conformal_quantile(s_cal, alpha)
        unweighted_cov.append(np.mean(s_te <= q))
        # weighted: likelihood ratio w(x)=dP_te/dP_cal for N(1,1)/N(0,1) = exp(x-0.5)
        w_cal = np.exp(x_cal - 0.5)
        wc = WeightedConformal(alpha).calibrate(s_cal, w_cal)
        w_te = np.exp(x_te - 0.5)
        weighted_cov.append(np.mean(wc.covered(s_te, w_te)))
    uc, wcv = float(np.mean(unweighted_cov)), float(np.mean(weighted_cov))
    assert uc < 1 - alpha, f"expected plain split to under-cover under shift, got {uc:.4f}"
    assert wcv >= 1 - alpha - 0.03, f"weighted conformal failed to restore coverage: {wcv:.4f}"
    print(f"  Weighted coverage={wcv:.4f} vs unweighted (shifted)={uc:.4f} (target>={1-alpha})")


def test_small_n_gives_infinite_quantile():
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.01) == np.inf
    assert weighted_quantile(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0.01, test_weight=1.0) == np.inf


def _main():
    tests = [test_coverage_band_1d_absolute, test_coverage_band_2d_joint_mo, test_score_mo_matches_manual,
             test_mondrian_restores_group_coverage, test_weighted_restores_coverage_under_shift,
             test_small_n_gives_infinite_quantile]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed ({TRIALS} MC resplits; CPU-only, 0 GB VRAM).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
