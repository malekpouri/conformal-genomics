#!/usr/bin/env python3
"""
Referee audit A5 — domain baselines for the off-target oracle, and AP-led reporting (roadmap R3).

The defect: the manuscript's only comparator is a naive mismatch count (ROC-AUC 0.696), while the
headline metric is pooled pair-level ROC-AUC on a problem with 1.26% positives.  A bioinformatics
referee will ask for (a) average precision as the lead metric, (b) within-guide stratified
discrimination, and (c) the standard domain scores.

Baselines implemented here, all evaluated leak-free (grouped on `sgRNA_type`, the real guide):
  1. mismatch count                      -- the paper's comparator
  2. MIT / Hsu-Zhang (2013) position-weighted score
  3. position-specific substitution logistic model -- a transparent, fully-specified "learned CFD":
     logistic regression on 20 positions x 16 (guide base, off base) combinations.  This is the
     comparator that answers "is the 181-feature gradient-boosted oracle actually buying anything?"
  4. the manuscript's oracle (181 features, HistGradientBoosting) for reference

CFD (Doench et al. 2016) is NOT included: its penalty matrix is distributed only as a binary
pickle (`mismatch_score.pkl` / `pam_scores.pkl` in the CRISPOR distribution) and downloading and
unpickling third-party binaries was not permitted in this environment.  See the roadmap note for
the exact command to add it.

CPU-only.  Output: audit/results/a5_domain_baselines.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]

PROTO = 20
SEED = 0
BOOT = 2000

# MIT / Hsu-Zhang (2013) position weights, index 0 = position furthest from the PAM.
# Verified 2026-10-14 against the default `weights` argument of CRISPRseek::getOfftargetScore
# (Bioconductor, Zhu lab), which attributes it to Hsu et al., Nat Biotechnol 31:827-834 (2013):
# c(0, 0, 0.014, 0, 0, 0.395, 0.317, 0, 0.389, 0.079, 0.445, 0.508, 0.613, 0.851, 0.732,
#   0.828, 0.615, 0.804, 0.685, 0.583) -- identical, zeros included.
# The three-term formula below (position product x mismatch-spread term x 1/n_mm^2, aggregated
# as 100/(100+sum) at guide level) is independently reproduced in US patents 10781444 and
# 12441995.  Hsu's own supplementary table sits behind a paywall and remains unchecked; two
# independent implementations agreeing is the strongest confirmation available here.
HSU_W = np.array([0.000, 0.000, 0.014, 0.000, 0.000, 0.395, 0.317, 0.000, 0.389, 0.079,
                  0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804, 0.685, 0.583])

_LUT = np.full(256, -1, int)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i


def _charmat(series):
    s = series.str.replace("_", "", regex=False).str.upper().str[:PROTO]
    return np.frombuffer("".join(s.tolist()).encode(), dtype=np.uint8).reshape(len(s), PROTO)


def _onehot(cm):
    idx = _LUT[cm]; N = cm.shape[0]
    oh = np.zeros((N, PROTO, 4), np.float32)
    r, c = np.where(idx >= 0)
    oh[r, c, idx[idx >= 0]] = 1.0
    return oh.reshape(N, PROTO * 4)


def mit_hsu_score(mism: np.ndarray) -> np.ndarray:
    """MIT/Hsu-Zhang single-site score from a boolean mismatch pattern (n, 20).

    S = prod_i (1 - w_i * mm_i) * 1/((19-d)/19*4 + 1) * 1/n_mm^2 , with d the mean pairwise
    distance between mismatches and n_mm the mismatch count (terms 2 and 3 are set to 1 when
    n_mm < 2, per the published definition).
    """
    n = mism.shape[0]
    pos = np.arange(PROTO)
    out = np.empty(n, float)
    for i in range(n):
        m = mism[i].astype(bool)
        k = int(m.sum())
        if k == 0:
            out[i] = 1.0; continue
        t1 = float(np.prod(1.0 - HSU_W[m]))
        if k > 1:
            p = pos[m]
            d = float((p.max() - p.min()) / (k - 1))
            t2 = 1.0 / (((19.0 - d) / 19.0) * 4.0 + 1.0)
            t3 = 1.0 / (k * k)
        else:
            t2 = t3 = 1.0
        out[i] = t1 * t2 * t3
    return out


def subst_features(G, O):
    """20 positions x 16 (guide base, off base) one-hot = 320 features (a transparent 'learned CFD')."""
    gi, oi = _LUT[G], _LUT[O]
    n = G.shape[0]
    X = np.zeros((n, PROTO * 16), np.float32)
    ok = (gi >= 0) & (oi >= 0)
    r, c = np.nonzero(ok)
    X[r, c * 16 + gi[r, c] * 4 + oi[r, c]] = 1.0
    return X


def metrics(y, score, groups, name):
    pooled_auc = float(roc_auc_score(y, score)); pooled_ap = float(average_precision_score(y, score))
    aucs, aps = [], []
    for g in np.unique(groups):
        m = groups == g
        if 0 < y[m].sum() < m.sum():
            aucs.append(roc_auc_score(y[m], score[m]))
            aps.append(average_precision_score(y[m], score[m]))
    aucs, aps = np.array(aucs), np.array(aps)
    return {"baseline": name,
            "pooled_roc_auc": round(pooled_auc, 3),
            "pooled_average_precision": round(pooled_ap, 3),
            "within_guide_roc_auc_mean": round(float(aucs.mean()), 3),
            "within_guide_roc_auc_range": [round(float(aucs.min()), 3), round(float(aucs.max()), 3)],
            "within_guide_average_precision_mean": round(float(aps.mean()), 3),
            "n_guides_scored": int(len(aucs))}


def boot_rho(a, b, seed=SEED, reps=BOOT):
    rng = np.random.default_rng(seed); n = len(a); out = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        if len(np.unique(a[i])) < 3 or len(np.unique(b[i])) < 3:
            continue
        out.append(spearmanr(a[i], b[i]).statistic)
    out = np.asarray(out, float); out = out[np.isfinite(out)]
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def main():
    path = sorted(DS.glob("I_1_CIRCLE_seq*csv"))[0]
    df = pd.read_csv(path, usecols=["sgRNA_seq", "off_seq", "label", "sgRNA_type"])
    df = df[(df["sgRNA_seq"].str.replace("_", "", regex=False).str.len() >= PROTO) &
            (df["off_seq"].str.len() >= PROTO)].reset_index(drop=True)
    G = _charmat(df["sgRNA_seq"]); O = _charmat(df["off_seq"])
    mism = (G != O).astype(np.float32); mm = mism.sum(1)
    y = (df["label"].to_numpy() > 0).astype(int)
    real = df["sgRNA_type"].str.strip().to_numpy()
    guides = np.array(sorted(np.unique(real)))
    print(f"[a5] {len(df):,} pairs, {y.sum():,} validated ({y.mean():.2%}), "
          f"{len(guides)} real guides")

    res = {"cohort": {"n_pairs": int(len(df)), "n_validated": int(y.sum()),
                      "prevalence": round(float(y.mean()), 4), "n_real_guides": int(len(guides)),
                      "grouping": "sgRNA_type (real guide)"},
           "baselines": {}, "burden_fidelity": {}}

    scores = {}
    # 1. naive mismatch count (lower = safer, so negate for "cleavage likelihood")
    scores["mismatch_count"] = -mm
    # 2. MIT / Hsu-Zhang
    print("[a5] scoring MIT/Hsu-Zhang ...")
    scores["mit_hsu_2013"] = mit_hsu_score(mism)
    # 3. position-specific substitution logistic model, leave-one-real-guide-out
    print("[a5] fitting the position-substitution logistic model (10 folds) ...")
    Xs = subst_features(G, O)
    oof_lr = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(Xs, y, groups=real):
        lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        lr.fit(Xs[tr], y[tr]); oof_lr[te] = lr.predict_proba(Xs[te])[:, 1]
    scores["position_substitution_logreg"] = oof_lr
    # 4. the manuscript's oracle, leak-free
    cache = HERE / "results" / "oof_leakfree_181feat.npy"
    if cache.exists():
        print("[a5] reusing the leak-free 181-feature OOF predictions from audit A1")
        oof_gb = np.load(cache)
        assert len(oof_gb) == len(y), "cache length mismatch - rerun audit A1"
    else:
        print("[a5] fitting the 181-feature oracle (10 folds) ...")
        X181 = np.concatenate([mism, _onehot(G), _onehot(O), mm[:, None]], axis=1).astype(np.float32)
        oof_gb = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X181, y, groups=real):
            clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                                 min_samples_leaf=50, random_state=42)
            clf.fit(X181[tr], y[tr]); oof_gb[te] = clf.predict_proba(X181[te])[:, 1]
    scores["manuscript_oracle_181feat"] = oof_gb
    # 5. mismatch-pattern-only oracle (21 features): the same learner WITHOUT the guide/off-target
    #    identity channels.  Included because the leak-free collapse of per-guide burden fidelity
    #    turns out to be a property of those identity features, not of learning per se.
    print("[a5] fitting the 21-feature mismatch-pattern oracle (10 folds) ...")
    X21 = np.concatenate([mism, mm[:, None]], axis=1).astype(np.float32)
    oof_mm = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X21, y, groups=real):
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.08,
                                             min_samples_leaf=50, random_state=42)
        clf.fit(X21[tr], y[tr]); oof_mm[te] = clf.predict_proba(X21[te])[:, 1]
    scores["mismatch_pattern_oracle_21feat"] = oof_mm

    for name, s in scores.items():
        res["baselines"][name] = metrics(y, np.asarray(s, float), real, name)
        b = res["baselines"][name]
        print(f"     {name:32s} AUC {b['pooled_roc_auc']:.3f}  AP {b['pooled_average_precision']:.3f}"
              f"  within-guide AUC {b['within_guide_roc_auc_mean']:.3f}")

    # per-real-guide burden fidelity for every baseline that yields a probability-like score
    d = pd.DataFrame({"real": real, "y": y})
    true_burden = d.groupby("real")["y"].sum().reindex(guides).to_numpy(float)
    n_sites = d.groupby("real").size().reindex(guides).to_numpy(float)
    mm3 = pd.DataFrame({"real": real, "k": mm <= 3}).groupby("real")["k"].sum() \
            .reindex(guides).to_numpy(float)
    cands = {"baseline_candidate_site_count": n_sites, "baseline_mm3_site_count": mm3}
    for name in ("mit_hsu_2013", "position_substitution_logreg", "manuscript_oracle_181feat",
                 "mismatch_pattern_oracle_21feat"):
        cands[name] = pd.DataFrame({"real": real, "s": np.asarray(scores[name], float)}) \
            .groupby("real")["s"].sum().reindex(guides).to_numpy(float)
    for name, pb in cands.items():
        res["burden_fidelity"][name] = {
            "rho_spearman": round(float(spearmanr(pb, true_burden).statistic), 3),
            "rho_ci95_guide_bootstrap": boot_rho(pb, true_burden), "n_guides": int(len(guides))}
        print(f"     burden rho  {name:32s} {res['burden_fidelity'][name]['rho_spearman']:.3f} "
              f"CI {res['burden_fidelity'][name]['rho_ci95_guide_bootstrap']}")

    res["cfd_note"] = (
        "CFD (Doench et al. 2016) is not run here: the penalty matrix ships only as a binary "
        "pickle (CRISPOR: CFD_Scoring/mismatch_score.pkl, pam_scores.pkl) and downloading/"
        "unpickling third-party binaries was blocked in this environment. To add it: "
        "`git clone https://github.com/maximilianh/crisporWebsite && "
        "python audit/a5_domain_baselines.py --cfd crisporWebsite/CFD_Scoring`.")
    res["literature_comparison_note"] = (
        "Published pair-level numbers for CFD / Elevation / CRISTA / DeepCRISPR / CRISPR-Net on "
        "these same public benchmarks (the I_*/II_* files) should be tabulated in the manuscript "
        "with their source, clearly marked as literature values rather than re-runs.")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "a5_domain_baselines.json").write_text(json.dumps(res, indent=2))
    print("\n[a5] -> audit/results/a5_domain_baselines.json")


if __name__ == "__main__":
    main()
