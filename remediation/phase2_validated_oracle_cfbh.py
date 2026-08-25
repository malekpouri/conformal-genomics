#!/usr/bin/env python3
"""
Remediation Phase 2 — (A) richer validated off-target oracle, (B) REAL-DATA cfBH on validated truth.

Phase 1 showed a validated-grounded oracle reaches ROC-AUC 0.856 / burden rho 0.910 and extrapolated
cfBH power ~0.16. Phase 2:
  (A) Enrich the pair-level oracle (guide+off one-hot + mismatch pattern) and re-measure AUC / burden rho
      under guide-disjoint CV — assess the achievable ceiling.
  (B) MEASURE (not extrapolate) real-data cfBH power: run off-target-only conformal selection on the 80
      CIRCLE-seq guides against VALIDATED burden truth, using the out-of-fold predicted burden as the
      fixed predictor T_hat. Report FAR / power / precision / yield over MC guide resplits.

This is the honest real-data test the Phase-1 curve only implied. CPU-only.
Output: remediation/results/phase2.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
NOTEBOOK = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.selection import conformal_pvalues, bh_select                       # noqa: E402

PROTO = 20
QS = (0.10, 0.20)
SEL_MC = 300
SEED = 0
_LUT = np.full(256, -1, int)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i


def _find_circle():
    for p in (NOTEBOOK.parent, NOTEBOOK):
        hits = list(p.glob("I_1_CIRCLE_seq*csv"))
        if hits:
            return hits[0]
    return None


def _charmat(series):
    s = series.str.replace("_", "", regex=False).str.upper().str[:PROTO]
    return np.frombuffer("".join(s.tolist()).encode(), dtype=np.uint8).reshape(len(s), PROTO)


def _onehot(cm):
    idx = _LUT[cm]; N = cm.shape[0]
    oh = np.zeros((N, PROTO, 4), np.float32)
    r, c = np.where(idx >= 0)
    oh[r, c, idx[idx >= 0]] = 1.0
    return oh.reshape(N, PROTO * 4)


def main():
    path = _find_circle()
    df = pd.read_csv(path, usecols=["sgRNA_seq", "off_seq", "label"])
    df = df[(df["sgRNA_seq"].str.replace("_", "", regex=False).str.len() >= PROTO) &
            (df["off_seq"].str.len() >= PROTO)].reset_index(drop=True)
    G = _charmat(df["sgRNA_seq"]); O = _charmat(df["off_seq"])
    mism = (G != O).astype(np.float32); mm = mism.sum(1, keepdims=True)
    X = np.concatenate([mism, _onehot(G), _onehot(O), mm], axis=1).astype(np.float32)   # 20+80+80+1
    y = (df["label"].to_numpy() > 0).astype(int)
    guide = df["sgRNA_seq"].str.replace("_", "", regex=False).to_numpy()
    guides = np.unique(guide)

    # ---- (A) richer oracle, guide-disjoint CV -> OOF probs ----
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=guide):
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                             min_samples_leaf=50, random_state=42)
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof)); ap = float(average_precision_score(y, oof))

    d = pd.DataFrame({"guide": guide, "y": y, "oof": oof})
    g = d.groupby("guide")
    true_burden = g["y"].sum().reindex(guides).to_numpy(float)          # validated off-target count
    pred_burden = g["oof"].sum().reindex(guides).to_numpy(float)        # sum of predicted cleavage prob
    rho = float(spearmanr(pred_burden, true_burden).statistic)
    pear = float(np.corrcoef(pred_burden, true_burden)[0, 1])
    # cache per-guide burdens so the cfBH experiment can be iterated without re-training the oracle
    pd.DataFrame({"guide": guides, "true_burden": true_burden, "pred_burden": pred_burden}).to_csv(
        HERE / "results" / "phase2_burdens.csv", index=False)

    # ---- (B) REAL-DATA cfBH off-target-only selection (validated truth) ----
    # design criterion OK(guide) := validated_burden <= tau_off  (select provably-safe guides).
    # Target on the LOG1P scale (as the main pipeline models off-target) so the heavy tail does not
    # inflate the standardization and crush the predictor's dynamic range.
    tau = float(np.median(true_burden))
    zt, zp, ztau = np.log1p(true_burden), np.log1p(pred_burden), np.log1p(tau)
    s = float(zt.std()) or 1.0
    T_true = (ztau - zt) / s                                           # OK <=> burden<=tau <=> T_true>=0
    T_hat = (ztau - zp) / s                                            # fixed predictor (OOF-based)
    ok = true_burden <= tau
    n = len(guides); n_cal = int(round(0.66 * n)); rng = np.random.default_rng(SEED)

    sel = {}
    for q in QS:
        F, W, P, Y, NS = [], [], [], [], []
        for _ in range(SEL_MC):
            perm = rng.permutation(n); ci, ti = perm[:n_cal], perm[n_cal:]
            V_cal = T_true[ci] - T_hat[ci]
            p = conformal_pvalues(V_cal, -T_hat[ti])                   # boundary V(X_j,0) = -T_hat
            m = bh_select(p, q); ns = int(m.sum()); nok = int(ok[ti].sum())
            F.append(float(np.mean(~ok[ti][m])) if ns else 0.0)
            W.append(float(np.sum(ok[ti] & m) / nok) if nok else np.nan)
            P.append(float(np.mean(ok[ti][m])) if ns else np.nan)
            Y.append(ns / len(ti)); NS.append(ns)
        fin = lambda a: float(np.nanmean(a))
        sel[f"q={q}"] = {"mean_FAR": round(fin(F), 4), "FAR_le_q": bool(fin(F) <= q + 1e-9),
                         "mean_power": round(fin(W), 4), "mean_precision": round(fin(P), 4),
                         "mean_yield": round(fin(Y), 4), "mean_n_selected": round(float(np.mean(NS)), 2)}

    out = {
        "richer_oracle": {"features": "mismatch(20)+guide_onehot(80)+off_onehot(80)+count(1)=181",
                          "roc_auc": round(auc, 3), "average_precision": round(ap, 3),
                          "phase1_mmonly_roc_auc": 0.856},
        "per_guide_burden_fidelity": {"rho_spearman": round(rho, 3), "pearson": round(pear, 3),
                                      "phase1_rho": 0.910, "n_guides": int(n)},
        "real_data_cfbh_offtarget_only": {"tau_off_validated": round(tau, 2),
                                          "ok_fraction": round(float(ok.mean()), 3),
                                          "n_cal": n_cal, "n_test": n - n_cal, "mc": SEL_MC, "levels": sel},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "phase2.json").write_text(json.dumps(out, indent=2))

    print("=" * 66); print("REMEDIATION PHASE 2 — validated oracle + REAL-DATA cfBH"); print("=" * 66)
    print(f"(A) Richer off-target oracle (181 feats, guide-disjoint CV):")
    print(f"    ROC-AUC = {auc:.3f}  (Phase-1 mismatch-only 0.856)   avg precision = {ap:.3f}")
    print(f"    per-guide burden fidelity: rho = {rho:.3f} (Pearson {pear:.3f})  (Phase-1 0.910)")
    print(f"\n(B) REAL-DATA cfBH, off-target-only, VALIDATED truth "
          f"(tau={tau:.1f}, OK={out['real_data_cfbh_offtarget_only']['ok_fraction']}, "
          f"n_cal={n_cal}/n_test={n-n_cal}, {SEL_MC} resplits):")
    for q in QS:
        r = sel[f"q={q}"]
        print(f"    q={q}: FAR={r['mean_FAR']:.4f} (<=q {r['FAR_le_q']})  power={r['mean_power']:.3f}  "
              f"precision={r['mean_precision']:.3f}  yield={r['mean_yield']:.3f}  n_sel={r['mean_n_selected']}")
    print("\n[done] -> remediation/results/phase2.json")


if __name__ == "__main__":
    main()
