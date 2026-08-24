"""Principled density-ratio estimation for weighted conformal (RFC/R1 Thm 4, honesty clause 4).

Replaces the length-only heuristic with a sequence-feature domain classifier. The covariate shift most
relevant to this paper is *generative*: a generator emits sequences whose composition differs from the
calibration pool. We estimate w(x) = p_test(x)/p_cal(x) from a logistic classifier on interpretable
sequence features (GC content, 2-mer and 3-mer frequencies, sequence complexity) and expose a
sensitivity analysis over the classifier's regularization.
"""
from __future__ import annotations
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_BASES = "ACGT"
_K2 = ["".join(p) for p in itertools.product(_BASES, repeat=2)]
_K3 = ["".join(p) for p in itertools.product(_BASES, repeat=3)]
_K2I = {k: i for i, k in enumerate(_K2)}
_K3I = {k: i for i, k in enumerate(_K3)}


def seq_features(seqs):
    """Per-sequence features: GC | 16 2-mer freqs | 64 3-mer freqs | 3-mer complexity  (82-dim)."""
    out = np.zeros((len(seqs), 1 + 16 + 64 + 1), float)
    for r, s in enumerate(seqs):
        L = len(s)
        out[r, 0] = (s.count("G") + s.count("C")) / L
        d2 = np.zeros(16); d3 = np.zeros(64)
        for i in range(L - 1):
            j = _K2I.get(s[i:i + 2]);  d2[j] += 1 if j is not None else 0
        for i in range(L - 2):
            j = _K3I.get(s[i:i + 3]);  d3[j] += 1 if j is not None else 0
        n2, n3 = max(L - 1, 1), max(L - 2, 1)
        out[r, 1:17] = d2 / n2
        out[r, 17:81] = d3 / n3
        out[r, 81] = np.count_nonzero(d3) / n3          # linguistic complexity (unique 3-mer fraction)
    return out


def estimate_density_ratio(cal_seqs, test_seqs, C=1.0, seed=0):
    """Fit cal(0) vs test(1) classifier; return w for the calibration points and the fitted objects.

    w(x_cal) = [P(test|x)/P(cal|x)] * (n_cal/n_test)  — the prior-corrected likelihood ratio."""
    Xc, Xt = seq_features(cal_seqs), seq_features(test_seqs)
    X = np.vstack([Xc, Xt])
    y = np.r_[np.zeros(len(Xc)), np.ones(len(Xt))]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(scaler.transform(X), y)
    r_cal = clf.predict_proba(scaler.transform(Xc))[:, 1].clip(1e-6, 1 - 1e-6)
    w = (r_cal / (1 - r_cal)) * (len(cal_seqs) / len(test_seqs))
    ess = float(w.sum() ** 2 / np.sum(w ** 2))         # effective sample size
    return {"w": w, "clf": clf, "scaler": scaler, "ess": ess,
            "auc_proxy": float(clf.score(scaler.transform(X), y))}


def sensitivity(cal_seqs, test_seqs, Cs=(0.1, 1.0, 10.0), seed=0):
    """Report weight-distribution stability and ESS across regularization strengths."""
    rows = {}
    for C in Cs:
        d = estimate_density_ratio(cal_seqs, test_seqs, C=C, seed=seed)
        w = d["w"]
        rows[f"C={C}"] = {"ess": round(d["ess"], 1), "w_mean": round(float(w.mean()), 3),
                          "w_max": round(float(w.max()), 3), "clf_acc": round(d["auc_proxy"], 3)}
    return rows
