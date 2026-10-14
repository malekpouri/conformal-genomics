#!/usr/bin/env python3
"""
Referee audit A1 — regroup the CIRCLE-seq cohort on the REAL guide and redo everything that the
80-pseudo-guide grouping touched.  Addresses roadmap items R1a-R1e, R2, R3 (stratified), R6, R8.

The defect: remediation/phase2_validated_oracle_cfbh.py grouped on `sgRNA_seq` with "_" removed,
which yields 80 strings = 10 real protospacers x 8 per-row alignment representations
(4 PAM variants x 2 gap placements).  The dataset ships the correct key, `sgRNA_type`
(exactly 10 distinct values).  Because the model only sees the first 20 nt of the guide, the
guide-side features are bit-identical across "guide-disjoint" folds.

This script:
  (A) proves the grouping defect from the data itself;
  (B) refits the pair-level oracle under leave-one-REAL-guide-out CV (leak-free) and reports
      pooled AND within-guide-stratified ROC-AUC / average precision;
  (C) recomputes per-real-guide burden fidelity with a guide-bootstrap CI that reflects n=10,
      alongside two FREE baselines (candidate-site count, mm<=3 site count);
  (D) reruns cfBH at the real-guide level with a three-way guide-disjoint fit/calibrate/test design,
      absolute (pre-registered) tolerances instead of the cohort median of the test labels, and
      the site-count baseline as an alternative predictor;
  (E) derives and verifies the structural feasibility bound  n_cal >= m/q - 1  that governs whether
      cfBH can select anything at all -- the corrected form of the "~80 validated guides" protocol.

CPU-only.  Output: audit/results/a1_regrouped.json
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
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.selection import conformal_pvalues, bh_select                       # noqa: E402

PROTO = 20
QS = (0.10, 0.20)
SEED = 0
BOOT = 2000
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


def spearman_guide_bootstrap(a, b, reps=BOOT, seed=SEED):
    """Bootstrap a Spearman CI by resampling GUIDES (the real independent unit)."""
    rng = np.random.default_rng(seed); n = len(a); out = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        if len(np.unique(b[i])) < 3 or len(np.unique(a[i])) < 3:
            continue
        out.append(spearmanr(a[i], b[i]).statistic)
    out = np.asarray(out, float); out = out[np.isfinite(out)]
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


# ----------------------------------------------------------------------------- (A) the defect
def prove_defect(df):
    stripped = df["sgRNA_seq"].str.replace("_", "", regex=False)
    real = df["sgRNA_type"].str.strip()
    core = stripped.str.replace("-", "", regex=False).str[:PROTO]
    per_real = df.assign(_s=stripped, _r=real).groupby("_r")["_s"].nunique()
    return {
        "pipeline_group_key": "sgRNA_seq with '_' removed",
        "n_pipeline_groups": int(stripped.nunique()),
        "correct_group_key": "sgRNA_type (canonical on-target, NGG)",
        "n_real_guides": int(real.nunique()),
        "n_distinct_20mers_after_stripping_gaps": int(core.nunique()),
        "pipeline_groups_per_real_guide": sorted(per_real.unique().tolist()),
        "guide_side_features_identical_across_folds": True,
        "note": ("the 8 representations of one guide differ only in the PAM written into the "
                 "on-target string and in gap placement; feature extraction truncates to the "
                 "first 20 nt, so GroupKFold on the pipeline key trains and tests on the "
                 "identical guide feature vector"),
    }


# ----------------------------------------------------------------------------- (B) leak-free oracle
def fit_oracle(X, y, groups, tag, n_splits=None, cache=None):
    """OOF predictions under group CV.  `n_splits=None` means leave-one-group-out.

    `cache` names a .npy file; the (expensive) fits are reused when it already exists.
    """
    if cache is not None:
        cp = HERE / "results" / cache
        if cp.exists():
            oof = np.load(cp)
            print(f"     [cache] reusing {cache}")
            return oof, _oracle_metrics(y, oof, groups, tag, n_splits)
    splitter = (LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits=n_splits))
    oof = np.zeros(len(y))
    for tr, te in splitter.split(X, y, groups=groups):
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                             min_samples_leaf=50, random_state=42)
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    if cache is not None:
        (HERE / "results").mkdir(exist_ok=True)
        np.save(HERE / "results" / cache, oof)
    return oof, _oracle_metrics(y, oof, groups, tag, n_splits)


def _oracle_metrics(y, oof, groups, tag, n_splits=None):
    pooled_auc = float(roc_auc_score(y, oof)); pooled_ap = float(average_precision_score(y, oof))
    per_guide = []
    for g in np.unique(groups):
        m = groups == g
        if 0 < y[m].sum() < m.sum():
            per_guide.append((float(roc_auc_score(y[m], oof[m])),
                              float(average_precision_score(y[m], oof[m])),
                              float(y[m].mean())))
    aucs = np.array([p[0] for p in per_guide]); aps = np.array([p[1] for p in per_guide])
    return {
        "cv": tag,
        "pooled_roc_auc": round(pooled_auc, 3),
        "pooled_average_precision": round(pooled_ap, 3),
        "within_guide_roc_auc_mean": round(float(aucs.mean()), 3),
        "within_guide_roc_auc_ci95": [round(float(np.percentile(aucs, 2.5)), 3),
                                      round(float(np.percentile(aucs, 97.5)), 3)],
        "within_guide_average_precision_mean": round(float(aps.mean()), 3),
        "within_guide_average_precision_ci95": [round(float(np.percentile(aps, 2.5)), 3),
                                                round(float(np.percentile(aps, 97.5)), 3)],
        "n_folds": int(len(np.unique(groups)) if n_splits is None else n_splits),
        "positive_rate": round(float(y.mean()), 4),
    }


# ----------------------------------------------------------------------------- (D) cfBH, real guides
def cfbh_realguide(T_true, T_hat, ok, q, m_test, rng, reps=2000):
    """Exhaustive/MC guide-disjoint calibration-test splits over the REAL guides."""
    n = len(T_true); n_cal = n - m_test
    F, W, P, Y, NS, FEAS = [], [], [], [], [], []
    for _ in range(reps):
        perm = rng.permutation(n); ci, ti = perm[:n_cal], perm[n_cal:]
        V_cal = T_true[ci] - T_hat[ci]
        p = conformal_pvalues(V_cal, -T_hat[ti])
        sel = bh_select(p, q); ns = int(sel.sum()); nok = int(ok[ti].sum())
        F.append(float(np.mean(~ok[ti][sel])) if ns else 0.0)
        W.append(float(np.sum(ok[ti] & sel) / nok) if nok else np.nan)
        P.append(float(np.mean(ok[ti][sel])) if ns else np.nan)
        Y.append(ns / len(ti)); NS.append(ns)
        FEAS.append(int(np.ceil(len(ti) / (q * (n_cal + 1)))))
    f = lambda a: (round(float(np.nanmean(a)), 4) if np.any(np.isfinite(a)) else None)
    return {"n_cal": int(n_cal), "n_test": int(m_test),
            "min_attainable_pvalue": round(1.0 / (n_cal + 1), 4),
            "bh_threshold_for_smallest_p": round(q / m_test, 4),
            "k_min_candidates_at_minimal_p_for_any_selection": int(FEAS[0]),
            "selection_possible_in_principle": bool(FEAS[0] <= m_test),
            "mean_FAR": f(F), "FAR_le_q": bool((f(F) or 0.0) <= q + 1e-9),
            "mean_power": f(W), "mean_precision": f(P), "mean_yield": f(Y),
            "mean_n_selected": round(float(np.mean(NS)), 3),
            "frac_splits_with_empty_selection": round(float(np.mean(np.array(NS) == 0)), 3)}


def main():
    path = sorted(DS.glob("I_1_CIRCLE_seq*csv"))[0]
    df = pd.read_csv(path, usecols=["sgRNA_seq", "off_seq", "label", "sgRNA_type"])
    df = df[(df["sgRNA_seq"].str.replace("_", "", regex=False).str.len() >= PROTO) &
            (df["off_seq"].str.len() >= PROTO)].reset_index(drop=True)

    defect = prove_defect(df)

    G = _charmat(df["sgRNA_seq"]); O = _charmat(df["off_seq"])
    mism = (G != O).astype(np.float32); mm = mism.sum(1, keepdims=True)
    X = np.concatenate([mism, _onehot(G), _onehot(O), mm], axis=1).astype(np.float32)
    y = (df["label"].to_numpy() > 0).astype(int)
    real = df["sgRNA_type"].str.strip().to_numpy()
    pseudo = df["sgRNA_seq"].str.replace("_", "", regex=False).to_numpy()

    print(f"[a1] {len(df):,} pairs | {defect['n_pipeline_groups']} pipeline groups | "
          f"{defect['n_real_guides']} REAL guides")

    # (B) leak-free vs leaky oracle, same features, same learner
    print("[a1] fitting leak-free oracle (leave-one-real-guide-out, 10 folds)...")
    oof_clean, m_clean = fit_oracle(X, y, real, "leave-one-REAL-guide-out (n=10)",
                                    cache="oof_leakfree_181feat.npy")
    print(f"     leak-free : pooled AUC {m_clean['pooled_roc_auc']}  AP {m_clean['pooled_average_precision']}")
    print("[a1] refitting the published (leaky) grouping verbatim: GroupKFold(5) on 80 pseudo-guides ...")
    oof_leaky, m_leaky = fit_oracle(X, y, pseudo, "GroupKFold(5) on the 80 pseudo-guides "
                                   "(the published setup, verbatim)", n_splits=5,
                                   cache="oof_leaky_181feat.npy")
    print(f"     leaky     : pooled AUC {m_leaky['pooled_roc_auc']}  AP {m_leaky['pooled_average_precision']}")

    # (C) per-real-guide burden + free baselines
    d = pd.DataFrame({"real": real, "y": y, "oof": oof_clean, "oof_leaky": oof_leaky,
                      "mm": mm.ravel()})
    g = d.groupby("real")
    guides = np.array(sorted(d["real"].unique()))
    true_burden = g["y"].sum().reindex(guides).to_numpy(float)
    pred_clean = g["oof"].sum().reindex(guides).to_numpy(float)
    pred_leaky = g["oof_leaky"].sum().reindex(guides).to_numpy(float)
    n_sites = g.size().reindex(guides).to_numpy(float)
    mm3_sites = d.assign(k=d["mm"] <= 3).groupby("real")["k"].sum().reindex(guides).to_numpy(float)

    fid = {}
    for name, p in [("ml_oracle_leakfree", pred_clean), ("ml_oracle_published_leaky", pred_leaky),
                    ("baseline_candidate_site_count", n_sites),
                    ("baseline_mm3_site_count", mm3_sites)]:
        fid[name] = {"rho_spearman": round(float(spearmanr(p, true_burden).statistic), 3),
                     "rho_ci95_guide_bootstrap": spearman_guide_bootstrap(p, true_burden),
                     "n_guides": int(len(guides))}
        print(f"     burden fidelity {name:30s} rho={fid[name]['rho_spearman']:.3f} "
              f"CI {fid[name]['rho_ci95_guide_bootstrap']}")

    pd.DataFrame({"guide": guides, "true_burden": true_burden,
                  "pred_leakfree": pred_clean, "pred_leaky": pred_leaky,
                  "n_candidate_sites": n_sites, "n_mm3_sites": mm3_sites}).to_csv(
        HERE / "results" / "a1_per_real_guide_burdens.csv", index=False)

    # (D/E) cfBH at the real-guide level.  Absolute, pre-registered tolerances (R6): fixed counts of
    # validated CIRCLE-seq cleavage sites, chosen a priori and NOT read off the test labels.
    rng = np.random.default_rng(SEED)
    s = float(np.log1p(true_burden).std()) or 1.0
    cfbh = {}
    for tau in (100.0, 250.0, 500.0):
        zt, ztau = np.log1p(true_burden), np.log1p(tau)
        T_true = (ztau - zt) / s
        ok = true_burden <= tau
        row = {"tau_absolute_validated_sites": tau, "n_ok": int(ok.sum()),
               "ok_fraction": round(float(ok.mean()), 3)}
        for pname, pb in [("ml_oracle_leakfree", pred_clean),
                          ("baseline_mm3_site_count", mm3_sites)]:
            T_hat = (ztau - np.log1p(pb * (true_burden.sum() / max(pb.sum(), 1e-9)))) / s \
                if pname.startswith("baseline") else (ztau - np.log1p(pb)) / s
            for q in QS:
                for m_test in (3, 4):
                    k = f"{pname}|q={q}|m_test={m_test}"
                    row[k] = cfbh_realguide(T_true, T_hat, ok, q, m_test, rng)
        cfbh[f"tau={int(tau)}"] = row

    # the structural bound: BH can reject the smallest p only if 1/(n_cal+1) <= q/m
    feas = {f"q={q}": {f"m_test={m}": {"min_n_cal_required": int(np.ceil(m / q - 1)),
                                       "min_total_guides_required": int(np.ceil(m / q - 1)) + m}
                       for m in (1, 3, 4, 10, 27)} for q in QS}

    out = {
        "A_grouping_defect": defect,
        "B_oracle_leakfree_vs_published": {"leak_free": m_clean, "published_leaky": m_leaky,
                                           "published_reported": {"roc_auc": 0.925,
                                                                  "average_precision": 0.281}},
        "C_burden_fidelity_per_real_guide": fid,
        "C_published_reported_burden_rho_over_80_pseudo_guides": 0.944,
        "D_cfbh_real_guide_level": cfbh,
        "E_structural_feasibility_bound": {
            "rule": ("conformal p-values are bounded below by 1/(n_cal+1); BH rejects the k-th "
                     "smallest only if p_(k) <= q*k/m, so a non-empty selection needs at least "
                     "k_min = ceil(m/(q*(n_cal+1))) candidates AT the minimal p-value"),
            "implies": "to certify a single candidate (k=1): n_cal >= m/q - 1",
            "table": feas,
            "published_setup": {"n_cal": 53, "n_test": 27, "min_attainable_p": round(1 / 54, 4),
                                "bh_threshold_smallest_p_q0.10": round(0.10 / 27, 4),
                                "feasible": bool(1 / 54 <= 0.10 / 27)},
        },
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "a1_regrouped.json").write_text(json.dumps(out, indent=2))
    print("\n[a1] -> audit/results/a1_regrouped.json")


if __name__ == "__main__":
    main()
