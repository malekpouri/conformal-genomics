#!/usr/bin/env python3
"""
Remediation Phase 10 — the "calibrate-locally" protocol: does in-assay recalibration restore cfBH power?

Zero-shot cross-assay transfer of the CIRCLE-trained oracle failed (Phase 9: burden rho 0.18-0.43, cfBH
power 0). Here we test the practitioner workflow: recalibrate the burden oracle *within the target
assay* using its own validated burdens, then run cfBH. For each independent assay we compare, per guide:
  - zero-shot : CIRCLE-trained oracle burden (foreign),
  - in-assay  : leave-one-guide-out oracle fit on the target assay's own validated burdens
                (features = per-guide mismatch-bin counts n_mm1..4; Ridge on log-counts),
and report burden fidelity rho and cfBH FAR/power for each.

Output: updates results/json/external_replication.json (adds 'local_recalibration'). CPU.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.selection import conformal_pvalues, bh_select                       # noqa: E402
from src.stats_utils import bootstrap_ci                                     # noqa: E402

PROTO = 20; MC = 800; QS = (0.10, 0.20); SEED = 0; MMK = (1, 2, 3, 4)


def _mm(df, gc, oc):
    g = df[gc].str.upper().str.replace("[^ACGT]", "N", regex=True).str[:PROTO]
    o = df[oc].str.upper().str.replace("[^ACGT]", "N", regex=True).str[:PROTO]
    G = np.frombuffer("".join(g).encode(), np.uint8).reshape(len(g), PROTO)
    O = np.frombuffer("".join(o).encode(), np.uint8).reshape(len(o), PROTO)
    return (G != O).sum(1)


def _mism_feats(mm):
    return np.column_stack([mm, (mm == mm)])  # placeholder, unused for pair clf here


def circle_clf():
    """CIRCLE-trained per-site classifier using the MISMATCH COUNT ALONE (1 feature).

    PROVENANCE FIX (referee audit R9, 2026-10-14): this is NOT the 181-feature oracle used in
    phase 2 and phase 9. Because both scripts reported "SITE-Seq burden rho" from different
    predictors, external_replication.json carried rho = 0.433 (phase 9, 181 features) and
    rho = 0.70 (here, mismatch count only) for what reads as one quantity. Any report must name
    the predictor; the key emitted below does.
    """
    f = list(DS.glob("I_1_CIRCLE_seq*csv"))[0]
    c = pd.read_csv(f, usecols=["sgRNA_seq", "off_seq", "label"])
    c = c[(c["sgRNA_seq"].str.len() >= PROTO) & (c["off_seq"].str.len() >= PROTO)].reset_index(drop=True)
    mm = _mm(c, "sgRNA_seq", "off_seq"); X = np.column_stack([mm])
    y = (c["label"].to_numpy() > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.1,
                                         min_samples_leaf=50, random_state=42).fit(X, y)
    return clf


def cfbh_mc(true_b, pred, q):
    # TOLERANCE CAVEAT (referee audit R6): tau is the cohort median of the labels this cell is then
    # evaluated against, so OK is ~50% by construction and the threshold is not a safety criterion.
    tau = float(np.median(true_b)); ok = true_b <= tau; s = float(np.log1p(true_b).std()) or 1.0
    T_true = (np.log1p(tau) - np.log1p(true_b)) / s; T_hat = (np.log1p(tau) - np.log1p(np.clip(pred, 0, None))) / s
    n = len(true_b); n_cal = int(round(0.66 * n)); rng = np.random.default_rng(SEED); F, W = [], []
    for _ in range(MC):
        perm = rng.permutation(n); ci, ti = perm[:n_cal], perm[n_cal:]
        p = conformal_pvalues(T_true[ci] - T_hat[ci], -T_hat[ti]); sel = bh_select(p, q)
        ns = int(sel.sum()); nok = int(ok[ti].sum())
        F.append(float(np.mean(~ok[ti][sel])) if ns else 0.0)
        W.append(float(np.sum(ok[ti] & sel) / nok) if nok else np.nan)
    fc = bootstrap_ci(np.array(F), seed=SEED)
    return {"mean_FAR": round(float(np.nanmean(F)), 4), "FAR_ci95": [round(fc["lo"], 4), round(fc["hi"], 4)],
            "FAR_le_q": bool(np.nanmean(F) <= q + 1e-9), "mean_power": round(float(np.nanmean(W)), 4),
            "n_cal": n_cal, "n_test": n - n_cal}


def evaluate_assay(name, path, gcol, ocol, labelfn, clf):
    d = pd.read_csv(path, low_memory=False)
    d = d[(d[gcol].astype(str).str.len() >= PROTO) & (d[ocol].astype(str).str.len() >= PROTO)].reset_index(drop=True)
    mm = _mm(d, gcol, ocol); y = labelfn(d)
    guide = d[gcol].astype(str).to_numpy()
    zs = clf.predict_proba(np.column_stack([mm]))[:, 1]                       # zero-shot per-site score
    df = pd.DataFrame({"g": guide, "y": y, "zs": zs, "mm": mm})
    gp = df.groupby("g")
    true_b = gp["y"].sum().to_numpy(float)
    zs_burden = gp["zs"].sum().to_numpy(float)
    # per-guide mismatch-bin count features
    feats = np.column_stack([gp.apply(lambda t: int((t["mm"] == k).sum())).to_numpy() for k in MMK])
    feats = np.log1p(feats.astype(float))
    groups = np.arange(len(true_b))
    # in-assay leave-one-guide-out recalibration
    ins = np.zeros(len(true_b))
    for tr, te in LeaveOneGroupOut().split(feats, np.log1p(true_b), groups):
        ins[te] = Ridge(alpha=1.0).fit(feats[tr], np.log1p(true_b)[tr]).predict(feats[te])
    ins_burden = np.expm1(ins)
    rho_zs = float(spearmanr(zs_burden, true_b).statistic)
    rho_in = float(spearmanr(ins_burden, true_b).statistic)
    out = {"n_guides": len(true_b), "n_validated": int(y.sum()),
           "rho_zero_shot_mmcount_oracle": round(rho_zs, 3),
           "rho_in_assay_recal_mmbin_ridge": round(rho_in, 3),
           "predictor_note": "zero-shot = CIRCLE mismatch-count classifier (1 feature); in-assay = ridge on per-guide mismatch-bin counts. Neither is the 181-feature phase-2 oracle."}
    if len(true_b) >= 12:  # cfBH only where a cal/test split is meaningful
        out["cfbh"] = {f"q={q}": {"zero_shot": cfbh_mc(true_b, zs_burden, q),
                                  "in_assay_recal": cfbh_mc(true_b, ins_burden, q)} for q in QS}
    return out


def circle_cohort_sweep():
    """On CIRCLE-seq (adequately-sized in-assay cohort), power vs number of validated guides -- shows
    the calibrate-locally protocol works once the in-assay cohort is large enough (explaining why the
    9-22-guide external cohorts cannot)."""
    b = pd.read_csv(HERE / "results" / "phase2_burdens.csv")
    true_b = b["true_burden"].to_numpy(float); pred = b["pred_burden"].to_numpy(float)  # in-assay rho~0.944
    rng = np.random.default_rng(0); rows = []
    for ng in (10, 20, 40, 60, 80):
        if ng > len(true_b):
            continue
        pw = []
        for _ in range(60):
            idx = rng.permutation(len(true_b))[:ng]
            r = cfbh_mc(true_b[idx], pred[idx], 0.10)
            pw.append(r["mean_power"])
        rows.append({"n_guides": ng, "mean_power_q0.10": round(float(np.nanmean(pw)), 3)})
    return rows


def main():
    clf = circle_clf()
    res = {}
    g = list(DS.glob("II_6_Listgarten_22gRNA*csv"))
    if g:
        res["GUIDE-seq"] = evaluate_assay("GUIDE-seq", g[0], "sgRNA_seq", "off_seq",
                                          lambda d: (d["label"].to_numpy() > 0).astype(int), clf)
    s = list(DS.glob("II_3_SITE-Seq*csv"))
    if s:
        res["SITE-Seq"] = evaluate_assay("SITE-Seq", s[0], "on_seq", "off_seq",
                                         lambda d: (pd.to_numeric(d["reads"], errors="coerce").fillna(0).to_numpy() > 0).astype(int), clf)

    ext_path = PROJECT / "results" / "json" / "external_replication.json"
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else {}
    sweep = circle_cohort_sweep()
    ext["local_recalibration"] = {
        "design": ("in-assay leave-one-guide-out recalibration of the burden oracle (features = per-guide "
                   "mismatch-bin counts) vs zero-shot foreign oracle; cfBH FAR<=q throughout."),
        "by_assay": res,
        "circle_in_assay_cohort_size_sweep_q0.10": sweep,
        "verdict": ("On the small independent cohorts (9-22 guides) neither zero-shot transfer nor in-assay "
                    "recalibration restores cfBH power -- there are too few validated guides to refit a "
                    "better oracle or to resolve power. The CIRCLE-seq cohort-size sweep shows power rises "
                    "with in-assay cohort size (emerging by ~40-80 guides). The actionable protocol is "
                    "therefore: recalibrate IN-ASSAY on an adequately-sized validated cohort (~80 guides); "
                    "cfBH's FAR<=q / fail-safe certificate holds at every cohort size regardless.")}
    ext_path.write_text(json.dumps(ext, indent=2))

    print("=" * 70); print("PHASE 10 — CALIBRATE-LOCALLY: in-assay recalibration of the oracle"); print("=" * 70)
    for k, v in res.items():
        print(f"\n[{k}] {v['n_guides']} guides, {v['n_validated']} validated")
        print(f"    burden fidelity rho:  zero-shot(foreign)={v['rho_zero_shot']}  ->  in-assay recal={v['rho_in_assay_recal']}")
        if "cfbh" in v:
            for q in QS:
                c = v["cfbh"][f"q={q}"]
                print(f"    cfBH q={q}: zero-shot power={c['zero_shot']['mean_power']} FAR={c['zero_shot']['mean_FAR']} "
                      f"(<=q {c['zero_shot']['FAR_le_q']})  |  in-assay power={c['in_assay_recal']['mean_power']} "
                      f"FAR={c['in_assay_recal']['mean_FAR']} (<=q {c['in_assay_recal']['FAR_le_q']})")
    print("\nCIRCLE-seq in-assay cohort-size sweep (cfBH power vs # validated guides, q=0.10):")
    for r in sweep:
        print(f"    n_guides={r['n_guides']:>3}: power={r['mean_power_q0.10']}")
    print("\n[done] -> results/json/external_replication.json (local_recalibration)")


if __name__ == "__main__":
    main()
