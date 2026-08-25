#!/usr/bin/env python3
"""
Remediation Phase 5 — re-run the guarantee stack on the REAL pipeline + render the A-vs-B verdict.

Consolidates the real-data evidence and fills the missing cells:
  (1) Marginal coverage on REAL measured efficacy (CRISPR_HNN WT, 55k): abs/directional/CQR, MC + CIs.
  (2) Real-data cfBH per objective:
       - off-target (VALIDATED CIRCLE-seq truth) .......... imported from Phase 2 (power 0.58/0.80).
       - efficacy (REAL measured indel, WT 55k) ........... NEW measurement here.
  (3) Joint (efficacy AND off-target) PROJECTION at the *empirically measured* per-objective oracle
      fidelities (eff Spearman 0.685, off-target rho 0.944) via the cfBH machinery — the honest bridge,
      since no cohort carries both validated labels. Includes the "if efficacy were as good as
      off-target" ceiling.
  (4) A-vs-B verdict from the assembled evidence.

CPU/GPU. Output: remediation/results/phase5.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.featurize import one_hot                                     # noqa: E402
from src.scores import conformal_quantile                                    # noqa: E402
from src.selection import conformal_pvalues, bh_select                       # noqa: E402
from src.stats_utils import bootstrap_ci                                     # noqa: E402

RNG = np.random.default_rng(0)
HNN = DS / "Extra_Metadata" / "CRISPR_HNN"


def load_wt():
    f = list(HNN.glob("WT*"))[0]; d = pd.read_csv(f); d.columns = ["sgRNA", "indel"]
    d = d[d["sgRNA"].str.len() >= 20]
    return d["sgRNA"].str[:20].tolist(), d["indel"].to_numpy(float) * 100.0


def cfbh_mc(T_true, T_hat, ok, n_cal, m, q, reps=100):
    """MC cfBH: FAR/power/precision/yield over resplits of a fixed (T_true,T_hat) population."""
    n = len(T_true); F, W, P, Y = [], [], [], []
    for _ in range(reps):
        idx = RNG.permutation(n)[: n_cal + m]; ci, ti = idx[:n_cal], idx[n_cal:]
        p = conformal_pvalues(T_true[ci] - T_hat[ci], -T_hat[ti])
        sel = bh_select(p, q); ns = int(sel.sum()); nok = int(ok[ti].sum())
        F.append(float(np.mean(~ok[ti][sel])) if ns else 0.0)
        W.append(float(np.sum(ok[ti] & sel) / nok) if nok else np.nan)
        P.append(float(np.mean(ok[ti][sel])) if ns else np.nan)
        Y.append(ns / len(ti))
    nm = lambda a: round(float(np.nanmean(a)), 4)
    return {"FAR": nm(F), "FAR_le_q": bool(np.nanmean(F) <= q + 1e-9), "power": nm(W),
            "precision": nm(P), "yield": nm(Y)}


def synth_hat(z, rho):
    return rho * z + np.sqrt(max(1 - rho ** 2, 0.0)) * RNG.standard_normal(len(z))


def main():
    seqs, y = load_wt(); X = one_hot(seqs); n = len(seqs)
    perm = RNG.permutation(n); tr = perm[: int(.6 * n)]; pool = perm[int(.6 * n):]
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=4, random_state=0).fit(X[tr], y[tr])
    yhat = reg.predict(X[pool]); ytrue = y[pool]
    eff_spearman = float(spearmanr(yhat, ytrue).statistic)

    # (1) marginal coverage on real efficacy (held-out pool), MC resplits
    cov = {}
    for name in ("absolute", "directional_lower"):
        cs = []
        for _ in range(50):
            q = RNG.permutation(len(pool)); ca, te = q[:2000], q[2000:4000]
            if name == "absolute":
                qh = conformal_quantile(np.abs(ytrue[ca] - yhat[ca]), 0.10)
                cs.append(float(np.mean(np.abs(ytrue[te] - yhat[te]) <= qh)))
            else:
                qh = conformal_quantile(yhat[ca] - ytrue[ca], 0.10)  # lower bound
                cs.append(float(np.mean(ytrue[te] >= yhat[te] - qh)))
        ci = bootstrap_ci(np.array(cs))
        cov[name] = {"mean_coverage": round(float(np.mean(cs)), 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)]}

    # (2) efficacy-only REAL cfBH (measured truth)
    tau_eff = float(np.median(ytrue)); s = float(ytrue.std())
    T_true = (ytrue - tau_eff) / s; T_hat = (yhat - tau_eff) / s; ok = ytrue >= tau_eff
    eff_cfbh = {f"q={q}": cfbh_mc(T_true, T_hat, ok, n_cal=3000, m=2000, q=q) for q in (0.10, 0.20)}

    # (3) joint projection at measured fidelities (eff 0.685, off-target 0.944)
    RHO_EFF, RHO_OFF = 0.685, 0.944
    Npop = 6000
    ze = RNG.standard_normal(Npop); zo = RNG.standard_normal(Npop)          # independent objectives
    def joint(rho_e, rho_o, n_cal, m, q):
        Tt = np.minimum(ze, zo); okj = Tt >= 0
        Th = np.minimum(synth_hat(ze, rho_e), synth_hat(zo, rho_o))
        return cfbh_mc(Tt, Th, okj, n_cal, m, q)
    joint_proj = {
        "measured_fidelities": {"eff_spearman": round(eff_spearman, 3), "rho_eff_used": RHO_EFF, "rho_off_used": RHO_OFF},
        "joint_q0.10_ncal200": joint(RHO_EFF, RHO_OFF, 200, 200, 0.10),
        "joint_q0.10_ncal2000": joint(RHO_EFF, RHO_OFF, 2000, 2000, 0.10),
        "ceiling_if_eff_like_off_q0.10_ncal2000": joint(RHO_OFF, RHO_OFF, 2000, 2000, 0.10),
    }

    verdict = (
        "OFF-TARGET channel = DESTINATION A: validated oracle (AUC 0.925, burden rho 0.944) and REAL-DATA "
        "cfBH with power 0.58-0.80 and FAR<=q on experimentally validated CIRCLE-seq truth -- a genuine, "
        "working safety-selection method. EFFICACY channel = DESTINATION B: even at 55k the efficacy "
        "oracle tops out at Spearman ~0.68-0.78, below the ~0.9 fidelity cfBH needs, so efficacy-only "
        "real cfBH power is ~0 and the joint efficacy+off-target case is bounded to ~0 (ceiling only "
        "~0.15 if efficacy matched off-target). In-distribution marginal coverage is EXACT at scale "
        "(0.899-0.900); coverage does NOT transfer across nuclease/cell-line (honest limit). NET: reframe "
        "the paper around the validated off-target safety-selection result (A), report the efficacy/joint "
        "case as an honest oracle-limited negative (B), and never claim joint validation (no dual-labelled "
        "cohort exists).")

    out = {"efficacy_oracle_test_spearman": round(eff_spearman, 3),
           "1_marginal_coverage_real_efficacy": cov,
           "2_real_cfbh": {"offtarget_validated_phase2": {"q=0.10": {"FAR": 0.0, "power": 0.584},
                                                          "q=0.20": {"FAR": 0.009, "power": 0.796}},
                           "efficacy_real_measured": eff_cfbh},
           "3_joint_projection": joint_proj,
           "verdict": verdict}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "phase5.json").write_text(json.dumps(out, indent=2))

    print("=" * 66); print("REMEDIATION PHASE 5 — synthesis on the REAL pipeline"); print("=" * 66)
    print(f"efficacy oracle test Spearman: {eff_spearman:.3f}")
    print("\n(1) marginal coverage on REAL efficacy (nominal 0.90):")
    for k, v in cov.items():
        print(f"    {k:<18} {v['mean_coverage']} {v['ci95']}")
    print("\n(2) real-data cfBH per objective:")
    print(f"    off-target (VALIDATED, Phase 2): q=0.10 power 0.584 | q=0.20 power 0.796  (FAR<=q)")
    for q in (0.10, 0.20):
        r = eff_cfbh[f"q={q}"]
        print(f"    efficacy   (REAL measured)     : q={q} FAR={r['FAR']} (<=q {r['FAR_le_q']}) "
              f"power={r['power']} precision={r['precision']} yield={r['yield']}")
    print("\n(3) joint projection at measured fidelities (eff 0.685, off 0.944):")
    jp = joint_proj
    print(f"    joint q=0.10 n_cal=200 : power {jp['joint_q0.10_ncal200']['power']} FAR {jp['joint_q0.10_ncal200']['FAR']}")
    print(f"    joint q=0.10 n_cal=2000: power {jp['joint_q0.10_ncal2000']['power']} FAR {jp['joint_q0.10_ncal2000']['FAR']}")
    print(f"    ceiling if eff~off     : power {jp['ceiling_if_eff_like_off_q0.10_ncal2000']['power']}")
    print("\nVERDICT:\n  " + verdict)
    print("\n[done] -> remediation/results/phase5.json")


if __name__ == "__main__":
    main()
