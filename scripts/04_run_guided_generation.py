#!/usr/bin/env python3
"""
ConformalGen — Phase 4: conformal-guided generation & acceptance policy (RFC-001 §3.3).

A UNIVERSAL, model-agnostic guarantee-aware filter for generative genomic design. The generator is a
pluggable GenericSequenceGenerator; we demonstrate general-purpose validity on standard baseline
pools:
  * raw_pool     — the raw candidate reservoir as-is (no generator prior).
  * sft_prior    — a standard unaligned generative prior (uniform draw).
  * aligned_prior— a generic aligned prior (softmax-tilted toward high-efficacy / low-off-target).
Any live generator (SFT, LLM, diffusion, aligned policy) is a drop-in replacement — nothing below is
specific to a particular model. Candidate reservoirs carry REAL oracle labels (held-out D_te guides),
so post-selection coverage is genuine.

Predictors fit on D_tr, calibrated on D_cal. For each pool, alpha in {0.10, 0.05}, and acceptance
mode (point / conformal_directional / conformal_joint) we report acceptance yield, design precision,
and post-selection bound-coverage, plus a (tau_eff, tau_off) sweep for yield-vs-coverage trade-offs.

CPU-only, 0 GB VRAM. Output: results/json/phase4_guided_generation.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.predictors import PropertyPredictors                             # noqa: E402
from src.guided_generation import ConformalGuidedGenerator, ReservoirGenerator, MODES  # noqa: E402

DATA = PROJECT / "data"
ALPHAS = (0.10, 0.05)
N_GEN = 2000
TILT = 1.2
TAU_EFF_HEADLINE, TAU_OFF_HEADLINE = 50.0, 12.0     # RFC candidate thresholds
TAU_EFF_GRID = [40.0, 45.0, 50.0, 55.0, 60.0]
TAU_OFF_GRID = [5.0, 10.0, 15.0, 20.0, 25.0]
SEED = 42


def _load(split):
    d = pd.read_csv(DATA / "splits" / f"{split}.csv")
    return d["seq"].tolist(), d["y_eff"].to_numpy(float), d["off_id"].to_numpy(float)


def _predictors():
    import joblib
    ck = PROJECT / "models" / "predictors.joblib"
    if ck.exists():
        return joblib.load(ck)
    seq, eff, off = _load("train")
    return PropertyPredictors().fit(seq, eff, off, alphas=ALPHAS)


def main():
    pp = _predictors()
    cal_seq, cal_eff, cal_off = _load("calibration")
    te_seq, te_eff, te_off = _load("test")

    gg = ConformalGuidedGenerator(pp).calibrate(cal_seq, cal_eff, cal_off, ALPHAS)

    # pluggable generators over the held-out reservoir (real oracle labels)
    reservoir = ReservoirGenerator(te_seq, te_eff, te_off, tilt=0.0, name="sft_prior")
    aligned = ReservoirGenerator(te_seq, te_eff, te_off, tilt=TILT, name="aligned_prior")

    pools = {}
    pools["raw_pool"] = reservoir.full()                                          # every candidate as-is
    pools["sft_prior"] = reservoir.sample(N_GEN, seed=SEED)                       # standard unaligned prior
    pools["aligned_prior"] = aligned.sample(N_GEN, seed=SEED)                     # generic aligned prior

    def summ(pool):
        _, ye, yo = pool
        return {"n": len(ye), "eff_mean": round(float(ye.mean()), 3),
                "eff_pct_ge_50": round(float(np.mean(ye >= 50) * 100), 1),
                "off_mean": round(float(yo.mean()), 3), "off_median": round(float(np.median(yo)), 1)}

    results = {"config": {"framework": "universal model-agnostic conformal wrapper",
                          "n_gen": N_GEN, "aligned_tilt": TILT, "seed": SEED,
                          "tau_eff_headline": TAU_EFF_HEADLINE, "tau_off_headline": TAU_OFF_HEADLINE,
                          "tau_eff_grid": TAU_EFF_GRID, "tau_off_grid": TAU_OFF_GRID, "alphas": list(ALPHAS)},
               "pool_summary": {k: summ(v) for k, v in pools.items()},
               "headline": {}, "sweep": {}}

    # headline @ (tau_eff=50, tau_off=12)
    for name, (s, ye, yo) in pools.items():
        results["headline"][name] = {}
        for a in ALPHAS:
            results["headline"][name][f"alpha={a}"] = {
                mode: gg.evaluate(s, ye, yo, TAU_EFF_HEADLINE, TAU_OFF_HEADLINE, a, mode) for mode in MODES}

    # (tau_eff, tau_off) sweep — point & conformal_directional, for trade-off curves
    for name, (s, ye, yo) in pools.items():
        results["sweep"][name] = {}
        for a in ALPHAS:
            rows = []
            for te_ in TAU_EFF_GRID:
                for to_ in TAU_OFF_GRID:
                    for mode in ("point", "conformal_directional"):
                        r = gg.evaluate(s, ye, yo, te_, to_, a, mode)
                        rows.append({"tau_eff": te_, "tau_off": to_, "mode": mode, "yield": r["yield"],
                                     "design_precision": r["design_precision"],
                                     "post_selection_bound_coverage": r["post_selection_bound_coverage"]})
            results["sweep"][name][f"alpha={a}"] = rows

    results["note"] = ("Model-agnostic: the generator is a pluggable GenericSequenceGenerator; pools "
                       "here (raw_pool / sft_prior / aligned_prior) are standard reference baselines "
                       "over labelled reservoir guides (real oracle labels). post_selection_bound_"
                       "coverage = P(y_eff>=L and y_off<=U | accepted); it lower-bounds design_precision "
                       "by construction. Tilted priors induce covariate shift vs D_cal (weighted "
                       "conformal, Phase 5, addresses residual coverage drift).")
    (PROJECT / "results" / "json").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results" / "json" / "phase4_guided_generation.json").write_text(json.dumps(results, indent=2))

    # ── console report ──
    print("Candidate pools (pluggable generators over labelled reservoir; real oracle labels):")
    for k, v in results["pool_summary"].items():
        print(f"  {k:<14} n={v['n']:>4}  eff_mean={v['eff_mean']:>6}  %eff>=50={v['eff_pct_ge_50']:>5}  "
              f"off_mean={v['off_mean']:>7}  off_med={v['off_median']}")
    print(f"\nHeadline acceptance @ tau_eff={TAU_EFF_HEADLINE}, tau_off={TAU_OFF_HEADLINE}:")
    print(f"{'pool':<14}{'alpha':>6}  {'mode':<22}{'yield':>8}{'precision':>11}{'bound_cov':>11}")
    for name in pools:
        for a in ALPHAS:
            for mode in MODES:
                r = results["headline"][name][f"alpha={a}"][mode]
                prec = "n/a" if r["design_precision"] is None else f"{r['design_precision']:.3f}"
                cov = "n/a" if r["post_selection_bound_coverage"] is None else f"{r['post_selection_bound_coverage']:.3f}"
                print(f"{name:<14}{a:>6}  {mode:<22}{r['yield']:>8.3f}{prec:>11}{cov:>11}")
    print("\n[done] -> results/json/phase4_guided_generation.json")


if __name__ == "__main__":
    main()
