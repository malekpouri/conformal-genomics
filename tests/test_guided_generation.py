#!/usr/bin/env python3
"""
ConformalGen — Phase 4 unit tests for conformal-guided generation (RFC-001 §3.3).

Checks: acceptance filter logic (bounds vs point), yield in [0,1], finite/non-NaN outputs,
conformal conservativeness (conformal yield <= point yield), and pool-sampling invariants.
CPU-only, 0 GB VRAM.  Run: python tests/test_guided_generation.py  |  pytest ...
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.predictors import PropertyPredictors                        # noqa: E402
from src.guided_generation import ConformalGuidedGenerator, ReservoirGenerator, MODES  # noqa: E402

DATA = PROJECT / "data"
ALPHAS = (0.10, 0.05)


def _load(split):
    d = pd.read_csv(DATA / "splits" / f"{split}.csv")
    return d["seq"].tolist(), d["y_eff"].to_numpy(float), d["off_id"].to_numpy(float)


def _generator():
    import joblib
    ck = PROJECT / "models" / "predictors.joblib"
    pp = joblib.load(ck) if ck.exists() else PropertyPredictors().fit(*_load("train"), alphas=ALPHAS)
    cs, ce, co = _load("calibration")
    return ConformalGuidedGenerator(pp).calibrate(cs, ce, co, ALPHAS), _load("test")


GG, (TE_SEQ, TE_EFF, TE_OFF) = _generator()


def test_accept_mask_shape_and_dtype():
    for mode in MODES:
        m = GG.accept(TE_SEQ, 50.0, 12.0, 0.10, mode)
        assert m.shape == (len(TE_SEQ),)
        assert m.dtype == bool


def test_yield_in_unit_interval_and_finite():
    for a in ALPHAS:
        for mode in MODES:
            r = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 50.0, 12.0, a, mode)
            assert 0.0 <= r["yield"] <= 1.0
            assert np.isfinite(r["yield"])
            for k in ("design_precision", "post_selection_bound_coverage"):
                assert r[k] is None or (np.isfinite(r[k]) and 0.0 <= r[k] <= 1.0)


def test_conformal_yield_not_above_point():
    """Conformal bounds are conservative (L_eff<=mu_eff, U_off>=mu_off) => yield <= point yield."""
    for a in ALPHAS:
        y_point = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 50.0, 12.0, a, "point")["yield"]
        for mode in ("conformal_directional", "conformal_joint"):
            y_c = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 50.0, 12.0, a, mode)["yield"]
            assert y_c <= y_point + 1e-9, f"{mode} yield {y_c} exceeds point {y_point} (alpha={a})"


def test_tighter_alpha_not_more_permissive():
    """Smaller alpha => wider bounds => yield(0.05) <= yield(0.10) for conformal modes."""
    for mode in ("conformal_directional", "conformal_joint"):
        y10 = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 50.0, 12.0, 0.10, mode)["yield"]
        y05 = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 50.0, 12.0, 0.05, mode)["yield"]
        assert y05 <= y10 + 1e-9


def test_monotonic_thresholds():
    """Raising tau_eff or lowering tau_off cannot increase yield."""
    base = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 45.0, 20.0, 0.10, "conformal_directional")["yield"]
    harder_eff = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 60.0, 20.0, 0.10, "conformal_directional")["yield"]
    harder_off = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 45.0, 5.0, 0.10, "conformal_directional")["yield"]
    assert harder_eff <= base + 1e-9 and harder_off <= base + 1e-9


def test_bound_coverage_lowerbounds_precision():
    """By construction accepted+covered => design-satisfying, so bound_coverage <= design_precision."""
    for a in ALPHAS:
        for mode in ("conformal_directional", "conformal_joint"):
            r = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 50.0, 12.0, a, mode)
            if r["n_accepted"] > 0:
                assert r["post_selection_bound_coverage"] <= r["design_precision"] + 1e-9


def test_impossible_threshold_zero_yield():
    r = GG.evaluate(TE_SEQ, TE_EFF, TE_OFF, 200.0, 12.0, 0.10, "conformal_directional")
    assert r["yield"] == 0.0 and r["n_accepted"] == 0
    assert r["design_precision"] is None and r["post_selection_bound_coverage"] is None


def test_generator_interface_and_sampling_invariants():
    for tilt in (0.0, 1.2):
        gen = ReservoirGenerator(TE_SEQ, TE_EFF, TE_OFF, tilt=tilt)
        s, ye, yo = gen.sample(500, seed=1)
        assert len(s) == len(ye) == len(yo) == 500
        w = gen.weights()
        assert np.isclose(w.sum(), 1.0) and np.all(w >= 0)
        assert np.all(np.isfinite(ye)) and np.all(np.isfinite(yo))
    # raw pool = full reservoir, no resampling
    sf, yef, yof = ReservoirGenerator(TE_SEQ, TE_EFF, TE_OFF).full()
    assert len(sf) == len(TE_SEQ)
    # aligned prior should raise mean efficacy and lower mean off-target vs unaligned prior
    _, ye_u, yo_u = ReservoirGenerator(TE_SEQ, TE_EFF, TE_OFF, tilt=0.0).sample(4000, seed=2)
    _, ye_a, yo_a = ReservoirGenerator(TE_SEQ, TE_EFF, TE_OFF, tilt=1.2).sample(4000, seed=2)
    assert ye_a.mean() > ye_u.mean() and yo_a.mean() < yo_u.mean()


def test_run_accepts_pluggable_generator():
    """The wrapper filters ANY GenericSequenceGenerator via .run()."""
    gen = ReservoirGenerator(TE_SEQ, TE_EFF, TE_OFF, tilt=1.2)
    r = GG.run(gen, 500, 45.0, 20.0, 0.10, "conformal_directional", seed=3)
    assert 0.0 <= r["yield"] <= 1.0 and r["n_candidates"] == 500


def _main():
    tests = [test_accept_mask_shape_and_dtype, test_yield_in_unit_interval_and_finite,
             test_conformal_yield_not_above_point, test_tighter_alpha_not_more_permissive,
             test_monotonic_thresholds, test_bound_coverage_lowerbounds_precision,
             test_impossible_threshold_zero_yield, test_generator_interface_and_sampling_invariants,
             test_run_accepts_pluggable_generator]
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
