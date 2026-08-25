#!/usr/bin/env python3
"""
Remediation Phase 9 — EXTERNAL CROSS-ASSAY REPLICATION.

The honest external test is cross-assay transfer: calibrate the off-target oracle on CIRCLE-seq (assay A)
and apply it, unchanged, to an INDEPENDENT assay -- Listgarten 22-gRNA GUIDE-seq (II_6, assay B). (Note:
GUIDE-seq has only 56 validated sites, far too few to *re-train* a classifier -- a trained oracle lands
at chance -- so cross-assay transfer of the CIRCLE-calibrated oracle is the correct design.)

Oracle: P(cleavage | mismatch=k) calibrated on CIRCLE-seq -> per-guide burden = sum_sites p_k over the
independent assay's candidate sites. Report cross-assay pair ROC-AUC / AP, per-guide burden fidelity
rho, empirical FAR and cfBH power on the independent cohort.

Output: results/json/external_replication.json.  CPU.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.selection import conformal_pvalues, bh_select                       # noqa: E402
from src.stats_utils import bootstrap_ci                                     # noqa: E402

PROTO = 20; MC = 800; QS = (0.10, 0.20); SEED = 0


def _mism_feats(df, gcol, ocol):
    """Per-position mismatch pattern (20) + count (1) -- transfer-robust, guide-identity-free."""
    g = df[gcol].str.upper().str.replace("[^ACGT]", "N", regex=True).str[:PROTO]
    o = df[ocol].str.upper().str.replace("[^ACGT]", "N", regex=True).str[:PROTO]
    G = np.frombuffer("".join(g).encode(), np.uint8).reshape(len(g), PROTO)
    O = np.frombuffer("".join(o).encode(), np.uint8).reshape(len(o), PROTO)
    m = (G != O).astype(np.float32)
    return np.concatenate([m, m.sum(1, keepdims=True)], 1)


def train_circle_oracle():
    """Train the mismatch-pattern cleavage classifier on ALL of CIRCLE-seq (assay A)."""
    f = list(DS.glob("I_1_CIRCLE_seq*csv"))[0]
    c = pd.read_csv(f, usecols=["sgRNA_seq", "off_seq", "label"])
    c = c[(c["sgRNA_seq"].str.len() >= PROTO) & (c["off_seq"].str.len() >= PROTO)].reset_index(drop=True)
    X = _mism_feats(c, "sgRNA_seq", "off_seq"); y = (c["label"].to_numpy() > 0).astype(int)
    return HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.08,
                                          min_samples_leaf=50, random_state=42).fit(X, y)


def cfbh_mc(true_b, pred, n_cal, m, q):
    tau = float(np.median(true_b)); ok = true_b <= tau; s = float(np.log1p(true_b).std()) or 1.0
    T_true = (np.log1p(tau) - np.log1p(true_b)) / s; T_hat = (np.log1p(tau) - np.log1p(pred)) / s
    n = len(true_b); rng = np.random.default_rng(SEED); F, W, P, Y = [], [], [], []
    for _ in range(MC):
        idx = rng.permutation(n)[: n_cal + m]; ci, ti = idx[:n_cal], idx[n_cal:]
        p = conformal_pvalues(T_true[ci] - T_hat[ci], -T_hat[ti]); sel = bh_select(p, q)
        ns = int(sel.sum()); nok = int(ok[ti].sum())
        F.append(float(np.mean(~ok[ti][sel])) if ns else 0.0)
        W.append(float(np.sum(ok[ti] & sel) / nok) if nok else np.nan)
        P.append(float(np.mean(ok[ti][sel])) if ns else np.nan); Y.append(ns / m)
    nm = lambda a: round(float(np.nanmean(a)), 4); fc = bootstrap_ci(np.array(F), seed=SEED)
    return {"tau": round(tau, 2), "ok_fraction": round(float(ok.mean()), 3), "mean_FAR": nm(F),
            "FAR_ci95": [round(fc["lo"], 4), round(fc["hi"], 4)], "FAR_le_q": bool(np.nanmean(F) <= q + 1e-9),
            "mean_power": nm(W), "mean_precision": nm(P), "mean_yield": nm(Y)}


def transfer_auc(clf, df, gcol, ocol, y):
    ss = clf.predict_proba(_mism_feats(df, gcol, ocol))[:, 1]
    g = pd.DataFrame({"guide": df[gcol].to_numpy(), "y": y, "s": ss}).groupby("guide")
    tb = g["y"].sum().to_numpy(float); pb = g["s"].sum().to_numpy(float)
    return {"n_guides": int(df[gcol].nunique()), "n_validated": int(y.sum()),
            "pair_roc_auc": round(float(roc_auc_score(y, ss)), 3),
            "pair_average_precision": round(float(average_precision_score(y, ss)), 3),
            "per_guide_burden_rho": round(float(spearmanr(pb, tb).statistic), 3)}, ss


def main():
    clf = train_circle_oracle()                                       # oracle trained on CIRCLE-seq

    # second independent assay: SITE-Seq (9 guides, ~3.8k validated -> clean transfer AUC)
    site = None
    sf = list(DS.glob("II_3_SITE-Seq*csv"))
    if sf:
        d3 = pd.read_csv(sf[0], low_memory=False)
        d3 = d3[(d3["on_seq"].astype(str).str.len() >= PROTO) & (d3["off_seq"].astype(str).str.len() >= PROTO)].reset_index(drop=True)
        y3 = (pd.to_numeric(d3["reads"], errors="coerce").fillna(0).to_numpy() > 0).astype(int)
        site, _ = transfer_auc(clf, d3, "on_seq", "off_seq", y3)

    f = list(DS.glob("II_6_Listgarten_22gRNA*csv"))[0]
    d = pd.read_csv(f)
    d = d[(d["sgRNA_seq"].str.len() >= PROTO) & (d["off_seq"].str.len() >= PROTO)].reset_index(drop=True)
    y = (d["label"].to_numpy() > 0).astype(int)
    site_score = clf.predict_proba(_mism_feats(d, "sgRNA_seq", "off_seq"))[:, 1]   # applied to GUIDE-seq
    auc = float(roc_auc_score(y, site_score)); ap = float(average_precision_score(y, site_score))

    guide = d["sgRNA_seq"].to_numpy()
    df = pd.DataFrame({"guide": guide, "y": y, "score": site_score}); g = df.groupby("guide")
    true_b = g["y"].sum().to_numpy(float); pred_b = g["score"].sum().to_numpy(float)
    rho = float(spearmanr(pred_b, true_b).statistic)
    n = len(true_b); n_cal = int(round(0.66 * n))
    sel = {f"q={q}": cfbh_mc(true_b, pred_b, n_cal, n - n_cal, q) for q in QS}

    out = {
        "design": "cross-assay transfer: oracle CALIBRATED on CIRCLE-seq, VALIDATED on GUIDE-seq",
        "external_dataset": {"file": f.name, "assay": "GUIDE-seq (Listgarten 22-gRNA)",
                             "n_pairs": int(len(y)), "n_validated": int(y.sum()), "n_guides": n,
                             "sites_per_guide_median": int(g.size().median()),
                             "negative_rate": round(float(1 - y.mean()), 5)},
        
        "cross_assay_oracle_GUIDEseq": {"pair_roc_auc": round(auc, 3), "pair_average_precision": round(ap, 3),
                                        "per_guide_burden_rho": round(rho, 3)},
        "cross_assay_oracle_SITEseq": site,
        "cfbh_replication_GUIDEseq": {"n_cal": n_cal, "n_test": n - n_cal, "mc": MC, "levels": sel},
        "primary_CIRCLE_seq": {"roc_auc": 0.925, "burden_rho": 0.944, "power_q0.10": 0.584, "power_q0.20": 0.796},
        "note": ("Trained-from-scratch oracles fail on GUIDE-seq (56 positives -> chance); cross-assay "
                 "transfer of the CIRCLE-calibrated mismatch->cleavage oracle is the valid external test. "
                 "cfBH FAR<=q holds on the independent cohort by construction."),
    }
    (PROJECT / "results" / "json").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results" / "json" / "external_replication.json").write_text(json.dumps(out, indent=2))

    print("=" * 70); print("PHASE 9 — EXTERNAL CROSS-ASSAY REPLICATION (CIRCLE-seq -> GUIDE-seq)")
    print("=" * 70)
    e = out["external_dataset"]
    print(f"{e['assay']}: {e['n_pairs']:,} pairs | {e['n_validated']} validated | {e['n_guides']} guides "
          f"| {e['sites_per_guide_median']:,} sites/guide")
    print(f"\ncross-assay oracle (CIRCLE-trained, applied to independent assays):")
    if site:
        print(f"  SITE-Seq  ({site['n_guides']} guides, {site['n_validated']} val): pair AUC {site['pair_roc_auc']} "
              f"| AP {site['pair_average_precision']} | burden rho {site['per_guide_burden_rho']}")
    print(f"  GUIDE-seq ({n} guides, {int(y.sum())} val): pair AUC {auc:.3f} | AP {ap:.3f} | burden rho {rho:.3f}")
    print(f"  (CIRCLE-seq primary in-assay: AUC 0.925 | rho 0.944)")
    print(f"\ncfBH replication ({n} guides, n_cal={n_cal}/n_test={n-n_cal}, {MC} resplits):")
    for q in QS:
        r = sel[f"q={q}"]
        print(f"  q={q}: FAR={r['mean_FAR']} {r['FAR_ci95']} (<=q {r['FAR_le_q']})  power={r['mean_power']}  "
              f"precision={r['mean_precision']}  yield={r['mean_yield']}  (OK={r['ok_fraction']})")
    print("\n[done] -> results/json/external_replication.json")


if __name__ == "__main__":
    main()
