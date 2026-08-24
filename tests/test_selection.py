#!/usr/bin/env python3
"""
ConformalGen — tests for cfBH conformal selection (Jin & Candes 2023; rigorous, no null pre-filtering).

Verifies: cfBH p-values are super-uniform for null candidates; BH is monotone in q; and Monte-Carlo
mean FAR <= q on exchangeable synthetic data using the entire calibration set. CPU-only, 0 GB VRAM.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.selection import conformal_pvalues, bh_select                       # noqa: E402

RNG = np.random.default_rng(0)


def _draw(n):
    """Latent target T ~ N(0,1); noisy predictor T_hat = T + noise. Returns (T, T_hat)."""
    T = RNG.normal(0, 1, n)
    T_hat = T + RNG.normal(0, 0.7, n)
    return T, T_hat


def test_pvalues_superuniform_for_nulls():
    """For null candidates (T_j < 0), cfBH p-values satisfy P(p<=t) <= t (using full calibration)."""
    for t in (0.1, 0.2):
        fracs = []
        for _ in range(400):
            Tc, Thc = _draw(400); Tt, Tht = _draw(400)
            V_cal = Tc - Thc                       # V(X_i, T_i) over ALL calibration points
            null = Tt < 0
            p = conformal_pvalues(V_cal, -Tht[null])   # boundary score V(X_j,0) = -T_hat_j
            fracs.append(np.mean(p <= t))
        m = float(np.mean(fracs))
        assert m <= t + 0.02, f"null p-values not super-uniform at t={t}: P(p<=t)={m:.3f}"


def test_bh_monotone_in_q():
    Tc, Thc = _draw(400); Tt, Tht = _draw(600)
    p = conformal_pvalues(Tc - Thc, -Tht)
    s_small = bh_select(p, 0.05); s_big = bh_select(p, 0.20)
    assert s_big.sum() >= s_small.sum() and np.all(s_small <= s_big)


def test_far_control_exchangeable():
    """MC mean FAR <= q on exchangeable data (finite-sample cfBH guarantee)."""
    for q in (0.1, 0.2):
        fars = []
        for _ in range(400):
            Tc, Thc = _draw(300); Tt, Tht = _draw(300)
            p = conformal_pvalues(Tc - Thc, -Tht)      # test ALL candidates at boundary 0
            sel = bh_select(p, q)
            ok = Tt >= 0
            fars.append(np.mean(~ok[sel]) if sel.any() else 0.0)
        m = float(np.mean(fars))
        assert m <= q + 0.03, f"FAR control violated: mean_FAR={m:.4f} > q={q}"


def test_uses_full_calibration_not_filtered():
    """Sanity: p-values change if calibration is (incorrectly) pre-filtered to nulls only."""
    Tc, Thc = _draw(500); Tt, Tht = _draw(50)
    V_cal_full = Tc - Thc
    V_cal_nullonly = V_cal_full[Tc < 0]                # the OLD (invalid) construction
    p_full = conformal_pvalues(V_cal_full, -Tht)
    p_filt = conformal_pvalues(V_cal_nullonly, -Tht)
    assert not np.allclose(p_full, p_filt)             # they genuinely differ
    assert np.all((p_full > 0) & (p_full <= 1))


def _main():
    tests = [test_pvalues_superuniform_for_nulls, test_bh_monotone_in_q,
             test_far_control_exchangeable, test_uses_full_calibration_not_filtered]
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
