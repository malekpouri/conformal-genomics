#!/usr/bin/env python3
"""
ConformalGen — Phase-2 tests for principled density-ratio estimation (R1 Thm-4 clause 4)
and bootstrap statistics. CPU-only, 0 GB VRAM.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.density_ratio import seq_features, estimate_density_ratio          # noqa: E402
from src.stats_utils import bootstrap_ci, paired_delta_ci                    # noqa: E402

RNG = np.random.default_rng(0)
NUC = "ACGT"


def _rand_seqs(n, gc_bias=0.5):
    p = np.array([(1 - gc_bias) / 2, gc_bias / 2, gc_bias / 2, (1 - gc_bias) / 2])  # A,C,G,T
    return ["".join(RNG.choice(list(NUC), size=20, p=p)) for _ in range(n)]


def test_seq_features_shape_and_ranges():
    X = seq_features(_rand_seqs(50))
    assert X.shape == (50, 82)
    assert np.all(np.isfinite(X))
    assert np.all((X[:, 0] >= 0) & (X[:, 0] <= 1))                # GC fraction
    assert np.allclose(X[:, 1:17].sum(axis=1), 1.0, atol=1e-6)    # 2-mer freqs sum to 1
    assert np.all((X[:, 81] >= 0) & (X[:, 81] <= 1))             # complexity fraction


def test_no_shift_weights_near_one_and_hard_to_classify():
    cal = _rand_seqs(400, 0.5); te = _rand_seqs(400, 0.5)         # same distribution
    d = estimate_density_ratio(cal, te, C=1.0)
    assert np.all(d["w"] > 0)
    assert d["ess"] > 0.5 * len(cal)                              # weights not degenerate
    assert d["auc_proxy"] < 0.65                                  # classifier near chance under no shift


def test_shift_detected_and_weights_vary():
    cal = _rand_seqs(400, 0.35); te = _rand_seqs(400, 0.75)       # strong GC shift
    d = estimate_density_ratio(cal, te, C=1.0)
    assert d["auc_proxy"] > 0.7                                   # classifier separates the pools
    assert d["ess"] < len(cal)                                    # weights concentrate
    assert d["w"].std() > 0


def test_bootstrap_ci_covers_true_mean():
    x = RNG.normal(0.9, 0.05, 500)
    ci = bootstrap_ci(x)
    assert ci["lo"] <= x.mean() <= ci["hi"] and ci["lo"] < ci["hi"]
    d = paired_delta_ci(x + 0.02, x)
    assert d["lo"] <= 0.02 <= d["hi"]


def _main():
    tests = [test_seq_features_shape_and_ranges, test_no_shift_weights_near_one_and_hard_to_classify,
             test_shift_detected_and_weights_vary, test_bootstrap_ci_covers_true_mean]
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
