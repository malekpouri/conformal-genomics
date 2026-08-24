#!/usr/bin/env python3
"""
ConformalGen — Phase 5 end-to-end benchmark reproducibility & sanity.

Verifies the RQ benchmark suite is deterministic (fixed seeds) and that its headline claims hold:
RQ1 coverage ~ nominal for every score; RQ3 plain conformal is valid-but-inflated on chr22 while
weighted conformal recovers coverage with tighter width; the consolidated JSON has the expected keys.
CPU-only, 0 GB VRAM.  Run: python tests/test_benchmark_repro.py  |  pytest ...
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import importlib.util                                                        # noqa: E402

spec = importlib.util.spec_from_file_location("rqbench", PROJECT / "scripts" / "05_run_rq_benchmarks.py")
rqb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rqb)
PP = rqb._predictors()
ALPHAS = (0.10, 0.05)


def test_rq1_coverage_near_nominal_and_deterministic():
    r1 = rqb.rq1(PP); r2 = rqb.rq1(PP)
    assert r1 == r2, "RQ1 not deterministic under fixed seed"
    for a in ALPHAS:
        for fam, r in r1[f"alpha={a}"].items():
            assert abs(r["signed_dev_from_nominal"]) <= 0.02, f"{fam} dev {r['signed_dev_from_nominal']} (a={a})"
            assert r["sd"] < 0.05
            assert np.isfinite(r["mean_coverage"])


def test_rq2_mondrian_restores_tail_coverage():
    r = rqb.rq2(PP)
    for a in ALPHAS:
        m = r[f"alpha={a}"]["mondrian_offtarget"]["by_stratum"]
        assert "high" in m
        # Mondrian should not worsen the heavy-tail (high) stratum coverage vs pooled
        assert m["high"]["cov_mondrian"] >= m["high"]["cov_pooled"] - 1e-9
        for name, st in m.items():
            assert 0.0 <= st["cov_pooled"] <= 1.0 and 0.0 <= st["cov_mondrian"] <= 1.0
            assert st["meanU_pooled"] > 0 and st["meanU_mondrian"] > 0


def test_rq3_plain_valid_weighted_more_efficient():
    r = rqb.rq3()
    for a in ALPHAS:
        d = r[f"alpha={a}"]; nom = d["nominal"]
        plain = d["a_plain_on_chr22"]; wtd = d["c_weighted_on_chr22"]
        assert plain["coverage"] >= nom - 0.02, "plain conformal should be ~valid/conservative on chr22"
        assert wtd["coverage"] >= nom - 0.03, "weighted conformal should recover coverage"
        assert wtd["meanU_count"] <= plain["meanU_count"] + 1e-9, "weighted should be no wider than plain"
        b = d["b_mondrian_in_distribution"]
        assert b["worst_chrom_cov_mondrian"] >= b["worst_chrom_cov_pooled"] - 0.02


def test_consolidated_json_present_and_keyed():
    p = PROJECT / "results" / "json" / "rq_benchmarks.json"
    assert p.exists(), "run scripts/05 first"
    d = json.loads(p.read_text())
    for k in ("RQ1_marginal_coverage", "RQ2_efficiency_width", "RQ3_ood_transfer", "guided_pareto"):
        assert k in d


def test_figures_exist():
    figs = ["fig1_framework_schematic", "fig2_rq1_coverage_validity", "fig3_rq2_efficiency_width",
            "fig4_rq3_ood_transfer", "fig5_guided_pareto", "fig7_selection_sensitivity"]
    for f in figs:
        p = PROJECT / "figures" / f"{f}.png"
        assert p.exists() and p.stat().st_size > 10_000, f"missing/empty figure {f}"


def _main():
    tests = [test_rq1_coverage_near_nominal_and_deterministic, test_rq2_mondrian_restores_tail_coverage,
             test_rq3_plain_valid_weighted_more_efficient, test_consolidated_json_present_and_keyed,
             test_figures_exist]
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
