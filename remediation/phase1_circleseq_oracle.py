#!/usr/bin/env python3
"""
Remediation Phase 1 — diagnostic: invert the off-target target to validated CIRCLE-seq cleavage.

Question this answers: is a *validated-grounded* off-target oracle materially stronger than the
hand-built MM<=3 count surrogate (ROC-AUC 0.696 in the paper), and does the implied guide-level
fidelity rho rise far enough to move cfBH real-data power off zero?

Steps:
  1. Encode each CIRCLE-seq (sgRNA, off-target) pair by its 20-nt protospacer mismatch pattern.
  2. Train a proper off-target classifier with GUIDE-DISJOINT CV (GroupKFold on sgRNA) -> out-of-fold
     P(validated cleavage). Report ROC-AUC and average precision vs the naive mismatch-count score.
  3. Per-guide VALIDATED BURDEN = # validated off-target sites; predicted burden = sum of OOF probs.
     Fidelity rho_new = Spearman(predicted, true) across guides; compare to rho_naive from MM<=3 counts.
  4. Read expected cfBH power at rho_new off the Phase-2(revision) sensitivity curve.

CPU-only. Output: remediation/results/phase1.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parents[1]                      # .../Human genomes Dataset/notebook
PROTO = 20
MM_MAX = 3
# expected cfBH power at a given oracle fidelity rho (from fig7 semisynthetic sweep, n_cal=m=200, q=0.10)
POWER_VS_RHO = {0.2: 0.000, 0.4: 0.000, 0.6: 0.000, 0.8: 0.006, 0.9: 0.112, 0.95: 0.334, 0.99: 0.707, 1.0: 0.980}


def _find_circle():
    for p in (NOTEBOOK.parent, NOTEBOOK):
        hits = list(p.glob("I_1_CIRCLE_seq*csv"))
        if hits:
            return hits[0]
    return None


def _proto(series):
    s = series.str.replace("_", "", regex=False).str.upper().str[:PROTO]
    return np.frombuffer("".join(s.tolist()).encode(), dtype=np.uint8).reshape(len(s), PROTO)


def expected_power(rho):
    xs = np.array(sorted(POWER_VS_RHO)); ys = np.array([POWER_VS_RHO[x] for x in xs])
    return float(np.interp(np.clip(rho, xs[0], xs[-1]), xs, ys))


def main():
    path = _find_circle()
    if path is None:
        print("CIRCLE-seq file not found"); return
    df = pd.read_csv(path, usecols=["sgRNA_seq", "off_seq", "label"])
    df = df[(df["sgRNA_seq"].str.replace("_", "", regex=False).str.len() >= PROTO) &
            (df["off_seq"].str.len() >= PROTO)].reset_index(drop=True)
    G = _proto(df["sgRNA_seq"]); O = _proto(df["off_seq"])
    mism = (G != O).astype(np.float32)                      # (N,20) per-position mismatch
    mm = mism.sum(1)                                        # total mismatch count
    X = np.column_stack([mism, mm])                        # 21 features
    y = (df["label"].to_numpy() > 0).astype(int)
    guide = df["sgRNA_seq"].str.replace("_", "", regex=False).to_numpy()
    n_guides = len(np.unique(guide))

    # ---- guide-disjoint CV: out-of-fold P(validated) ----
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups=guide):
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.08,
                                             min_samples_leaf=50, random_state=42)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    auc = float(roc_auc_score(y, oof)); ap = float(average_precision_score(y, oof))
    # naive baselines (paper): score = -mismatch (fewer mm => more likely off-target)
    auc_naive = float(roc_auc_score(y, -mm)); ap_naive = float(average_precision_score(y, -mm))

    # ---- per-guide validated burden fidelity ----
    d = pd.DataFrame({"guide": guide, "y": y, "oof": oof, "mm": mm})
    grp = d.groupby("guide")
    true_burden = grp["y"].sum()                                    # # validated off-target sites
    pred_burden_model = grp["oof"].sum()                            # sum of predicted cleavage prob
    pred_burden_naive = grp.apply(lambda g: int((g["mm"] <= MM_MAX).sum()))  # old surrogate: MM<=3 count
    rho_new = float(spearmanr(pred_burden_model, true_burden).statistic)
    rho_naive = float(spearmanr(pred_burden_naive, true_burden).statistic)
    r_pearson_new = float(np.corrcoef(pred_burden_model, true_burden)[0, 1])

    out = {
        "dataset": {"file": path.name, "n_pairs": int(len(y)), "n_validated": int(y.sum()),
                    "n_guides": int(n_guides), "positive_rate": round(float(y.mean()), 5)},
        "pair_classifier_guide_disjoint_CV": {
            "roc_auc": round(auc, 3), "average_precision": round(ap, 3),
            "naive_mmcount_roc_auc": round(auc_naive, 3), "naive_mmcount_ap": round(ap_naive, 3)},
        "per_guide_offtarget_burden_fidelity": {
            "rho_spearman_validated_oracle": round(rho_new, 3),
            "pearson_validated_oracle": round(r_pearson_new, 3),
            "rho_spearman_old_MM3_surrogate": round(rho_naive, 3),
            "n_guides": int(n_guides)},
        "implied_cfBH_power_at_rho_new_q0.10": round(expected_power(rho_new), 3),
        "reference_power_vs_rho_curve": POWER_VS_RHO,
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "phase1.json").write_text(json.dumps(out, indent=2))

    print("=" * 66)
    print("REMEDIATION PHASE 1 — CIRCLE-seq retargeting diagnostic")
    print("=" * 66)
    print(f"CIRCLE-seq: {out['dataset']['n_pairs']:,} pairs | {out['dataset']['n_validated']:,} validated "
          f"| {n_guides} guides | positive rate {out['dataset']['positive_rate']*100:.2f}%")
    print("\nPair-level off-target classifier (guide-disjoint 5-fold CV):")
    print(f"  trained model : ROC-AUC = {auc:.3f}   average precision = {ap:.3f}")
    print(f"  naive MM count: ROC-AUC = {auc_naive:.3f}   average precision = {ap_naive:.3f}   "
          f"(paper reported 0.696)")
    print("\nPer-guide off-target BURDEN fidelity (Spearman with validated truth):")
    print(f"  validated-grounded oracle : rho = {rho_new:.3f}  (Pearson {r_pearson_new:.3f})")
    print(f"  old MM<=3 surrogate        : rho = {rho_naive:.3f}")
    print(f"\nImplied cfBH power at rho={rho_new:.3f} (q=0.10, from fig7 curve): "
          f"{expected_power(rho_new):.3f}")
    print("\n[done] -> remediation/results/phase1.json")


if __name__ == "__main__":
    main()
