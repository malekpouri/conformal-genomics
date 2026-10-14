#!/usr/bin/env python3
"""
Referee audit A6 — cfBH vs a calibration-tuned threshold AT MATCHED YIELD, and the cohort size a
certified deployment actually needs (roadmap M4, and the constructive replacement for the
"~80 validated guides" protocol invalidated by R1).

Two defects are addressed.

(1) M4 -- unmatched comparison.  final_referee_closure.json compares cfBH (mean yield 0.297) with a
    calibration-tuned threshold (mean yield 0.545) at q = 0.10 and reports that the tuned rule
    violates the target on 36% of deployments.  Part of that violation is simply that it accepts
    1.8x as many candidates.  Here the tuned threshold is additionally evaluated at cfBH's realised
    yield, which is the comparison the manuscript's own "identical selections at matched yield"
    argument implies.

(2) The deployment protocol.  The published cohort-size sweep (power 0 -> 0.58 as n goes 10 -> 80)
    was computed over pseudo-guides, i.e. by adding alignment representations of the same 10 real
    guides, so it does not describe what happens as real validated guides accumulate.  This script
    replaces it with a design calculation anchored on the measured quantities: the empirical
    per-real-guide burden distribution (10 CIRCLE-seq guides) and the leak-free guide-level oracle
    fidelity, sweeping cohort size n and oracle fidelity rho.

Everything here is clearly labelled: the burden distribution and fidelity are measured, the cohort
sizes beyond 10 are a simulation of what such a cohort would yield.

CPU-only.  Output: audit/results/a6_matched_yield_cohort.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.selection import conformal_pvalues, bh_select                       # noqa: E402

PROTO = 20
QS = (0.10, 0.20)
TAU_ABS = 250.0          # pre-registered absolute tolerance: validated CIRCLE-seq cleavage sites
N_GRID = (10, 20, 40, 60, 80, 120, 200, 400)
RHO_GRID = (0.80, 0.90, 0.95, 0.99)
REPS = 400
SEED = 0


def observed_burdens():
    """Per-REAL-guide validated burden on the CIRCLE-seq cohort (grouped on sgRNA_type)."""
    path = sorted(DS.glob("I_1_CIRCLE_seq*csv"))[0]
    df = pd.read_csv(path, usecols=["sgRNA_seq", "off_seq", "label", "sgRNA_type"])
    df = df[(df["sgRNA_seq"].str.replace("_", "", regex=False).str.len() >= PROTO) &
            (df["off_seq"].str.len() >= PROTO)]
    b = (df.assign(hit=(df["label"] > 0).astype(int), g=df["sgRNA_type"].str.strip())
           .groupby("g")["hit"].sum().to_numpy(float))
    return b


def simulate_cohort(n, rho, logmu, logsd, rng):
    """Draw n guides with log-normal true burden and a predictor of rank fidelity ~rho."""
    z1 = rng.standard_normal(n)
    # Gaussian copula: Spearman rho ~= (6/pi) arcsin(r/2) -> invert for the latent correlation
    r = 2.0 * np.sin(np.pi * rho / 6.0)
    z2 = r * z1 + np.sqrt(max(1.0 - r * r, 0.0)) * rng.standard_normal(n)
    true_b = np.expm1(logmu + logsd * z1)
    pred_b = np.expm1(logmu + logsd * z2)
    return np.maximum(true_b, 0.0), np.maximum(pred_b, 0.0)


def tuned_threshold_far(T_true_cal, T_hat_cal, T_hat_te, ok_te, q, target_yield=None):
    """Calibration-tuned cutoff on T_hat.

    Default: the most permissive cutoff whose empirical FAR on the calibration split is <= q
    (the manuscript's "fair baseline").  With `target_yield`, the cutoff is instead set to accept
    exactly that fraction of the test pool -- the matched-yield comparison.
    """
    if target_yield is not None:
        k = int(round(target_yield * len(T_hat_te)))
        if k <= 0:
            return 0.0, 0.0, 0
        cut = np.sort(T_hat_te)[::-1][k - 1]
        sel = T_hat_te >= cut
    else:
        ok_cal = T_true_cal >= 0
        order = np.argsort(-T_hat_cal)
        best = None
        for i in range(1, len(order) + 1):
            far = float(np.mean(~ok_cal[order[:i]]))
            if far <= q:
                best = T_hat_cal[order[i - 1]]
        if best is None:
            return 0.0, 0.0, 0
        sel = T_hat_te >= best
    ns = int(sel.sum())
    far = float(np.mean(~ok_te[sel])) if ns else 0.0
    return far, ns / len(T_hat_te), ns


def run_cell(n, rho, q, logmu, logsd, rng):
    n_cal = int(round(0.66 * n)); m = n - n_cal
    cf_far, cf_pow, cf_yield = [], [], []
    tt_far, tt_pow, tt_yield = [], [], []
    my_far, my_yield = [], []
    for _ in range(REPS):
        true_b, pred_b = simulate_cohort(n, rho, logmu, logsd, rng)
        s = float(np.log1p(true_b).std()) or 1.0
        ztau = np.log1p(TAU_ABS)
        T_true = (ztau - np.log1p(true_b)) / s
        T_hat = (ztau - np.log1p(pred_b)) / s
        ok = true_b <= TAU_ABS
        perm = rng.permutation(n); ci, ti = perm[:n_cal], perm[n_cal:]
        # cfBH
        p = conformal_pvalues(T_true[ci] - T_hat[ci], -T_hat[ti])
        sel = bh_select(p, q); ns = int(sel.sum()); nok = int(ok[ti].sum())
        cf_far.append(float(np.mean(~ok[ti][sel])) if ns else 0.0)
        cf_pow.append(float(np.sum(ok[ti] & sel) / nok) if nok else np.nan)
        cf_yield.append(ns / m)
        # tuned threshold, its own operating point
        f, yy, _ = tuned_threshold_far(T_true[ci], T_hat[ci], T_hat[ti], ok[ti], q)
        tt_far.append(f); tt_yield.append(yy)
        tt_pow.append(float(np.sum(ok[ti] & (T_hat[ti] >= np.sort(T_hat[ti])[::-1][max(int(round(yy*m))-1, 0)])) / nok)
                      if (nok and yy > 0) else 0.0)
        # tuned threshold forced to cfBH's realised yield  (M4)
        f2, y2, _ = tuned_threshold_far(T_true[ci], T_hat[ci], T_hat[ti], ok[ti], q,
                                        target_yield=ns / m)
        my_far.append(f2); my_yield.append(y2)
    f = lambda a: round(float(np.nanmean(a)), 4)
    p95 = lambda a: round(float(np.nanpercentile(a, 95)), 4)
    viol = lambda a: round(float(np.mean(np.asarray(a) > q + 1e-12)), 3)
    return {
        "n_cal": n_cal, "m_test": m,
        "selection_feasible_by_construction": bool(1.0 / (n_cal + 1) <= q / m),
        "cfbh": {"mean_FAR": f(cf_far), "p95_FAR": p95(cf_far), "P_FAR_gt_q": viol(cf_far),
                 "mean_power": f(cf_pow), "mean_yield": f(cf_yield)},
        "tuned_threshold_own_operating_point": {
            "mean_FAR": f(tt_far), "p95_FAR": p95(tt_far), "P_FAR_gt_q": viol(tt_far),
            "mean_power": f(tt_pow), "mean_yield": f(tt_yield)},
        "tuned_threshold_at_cfbh_yield": {
            "mean_FAR": f(my_far), "p95_FAR": p95(my_far), "P_FAR_gt_q": viol(my_far),
            "mean_yield": f(my_yield)},
    }


def main():
    b = observed_burdens()
    lb = np.log1p(b)
    logmu, logsd = float(lb.mean()), float(lb.std(ddof=1))
    print(f"[a6] measured per-REAL-guide burden (n={len(b)}): median {np.median(b):.0f}, "
          f"range {b.min():.0f}-{b.max():.0f}; log1p mean {logmu:.2f} sd {logsd:.2f}")
    print(f"[a6] pre-registered tolerance tau = {TAU_ABS:.0f} validated sites "
          f"({np.mean(b <= TAU_ABS):.0%} of the observed cohort is OK)")

    # measured leak-free fidelity, if audit A1 has been run
    a1 = HERE / "results" / "a1_regrouped.json"
    measured_rho = None
    if a1.exists():
        j = json.loads(a1.read_text())
        measured_rho = j["C_burden_fidelity_per_real_guide"]["ml_oracle_leakfree"]["rho_spearman"]
        print(f"[a6] measured leak-free guide-level oracle fidelity rho = {measured_rho}")

    rng = np.random.default_rng(SEED)
    out = {"measured": {"n_real_guides": int(len(b)),
                        "burden_median": float(np.median(b)),
                        "burden_min": float(b.min()), "burden_max": float(b.max()),
                        "log1p_mean": round(logmu, 3), "log1p_sd": round(logsd, 3),
                        "tau_absolute": TAU_ABS,
                        "ok_fraction_observed": round(float(np.mean(b <= TAU_ABS)), 3),
                        "leakfree_oracle_rho": measured_rho},
           "design_note": ("burden distribution and oracle fidelity are MEASURED on the 10-guide "
                           "CIRCLE-seq cohort; cohort sizes above 10 are simulated from that "
                           "fitted distribution, so the sweep is a design calculation, not data"),
           "sweep": {}}

    rhos = list(RHO_GRID) + ([measured_rho] if measured_rho and measured_rho not in RHO_GRID else [])
    for q in QS:
        for rho in rhos:
            for n in N_GRID:
                key = f"q={q}|rho={rho}|n={n}"
                out["sweep"][key] = run_cell(n, float(rho), q, logmu, logsd, rng)
        # smallest cohort at which cfBH reaches non-trivial power, per fidelity
        for rho in rhos:
            ns = [n for n in N_GRID
                  if out["sweep"][f"q={q}|rho={rho}|n={n}"]["cfbh"]["mean_power"] >= 0.5]
            out.setdefault("min_cohort_for_power_0.5", {})[f"q={q}|rho={rho}"] = (ns[0] if ns else None)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "a6_matched_yield_cohort.json").write_text(json.dumps(out, indent=2))

    print("\n[a6] cfBH vs tuned threshold (rho = measured or 0.95), q = 0.10")
    rho_show = measured_rho or 0.95
    for n in N_GRID:
        c = out["sweep"][f"q=0.1|rho={rho_show}|n={n}"]
        print(f"     n={n:4d} (cal {c['n_cal']:3d}/test {c['m_test']:3d}) feasible={str(c['selection_feasible_by_construction']):5s} | "
              f"cfBH FAR {c['cfbh']['mean_FAR']:.3f} pow {c['cfbh']['mean_power']:.3f} yield {c['cfbh']['mean_yield']:.3f} | "
              f"tuned FAR {c['tuned_threshold_own_operating_point']['mean_FAR']:.3f} "
              f"P(>q) {c['tuned_threshold_own_operating_point']['P_FAR_gt_q']:.2f} "
              f"yield {c['tuned_threshold_own_operating_point']['mean_yield']:.3f} | "
              f"tuned@cfBH-yield FAR {c['tuned_threshold_at_cfbh_yield']['mean_FAR']:.3f} "
              f"P(>q) {c['tuned_threshold_at_cfbh_yield']['P_FAR_gt_q']:.2f}")
    print("\n[a6] min cohort for cfBH power >= 0.5:", out.get("min_cohort_for_power_0.5"))
    print("[a6] -> audit/results/a6_matched_yield_cohort.json")


if __name__ == "__main__":
    main()
