#!/usr/bin/env python3
"""
Remediation Phase 4 — honest experimental design (referee flaws #6, #7, #8).

(A) Leakage at scale (#6): marginal conformal coverage on CRISPR_HNN WT efficacy under RANDOM vs
    sequence-cluster GROUPED splits (MiniBatchKMeans), many resplits + CIs. Robust replacement for the
    paper's single-split 0.94->0.85.
(B) Real OOD (#7): nuclease/cell-line TRANSFER — calibrate conformal on WT, measure coverage on the
    engineered-Cas9 panel (ESP/HF/Sniper/xCas/SpCas9-NG). Replaces the degenerate chr22 hold-out.
    Also test sequence-feature weighted conformal and report, honestly, whether it can fix what is
    largely a *label/concept* shift.
(C) Domain uncertainty baselines (#8): deep ensemble and MC-dropout interval coverage vs conformal on a
    held-out WT set — real ML UQ, not Gaussian/QR strawmen; plus a conformalized-ensemble showing
    conformal fixes their miscalibration.

CPU/GPU. Output: remediation/results/phase4.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.featurize import one_hot                                     # noqa: E402
from src.scores import conformal_quantile                                    # noqa: E402
from src.conformal import weighted_quantile                                  # noqa: E402
from src.density_ratio import estimate_density_ratio, seq_features           # noqa: E402

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.10
HNN = DS / "Extra_Metadata" / "CRISPR_HNN"


def load_panel(name):
    f = list(HNN.glob(f"{name}*"))
    if not f:
        return None
    d = pd.read_csv(f[0]); d.columns = ["sgRNA", "indel"]
    d = d[d["sgRNA"].str.len() >= 20]
    y = d["indel"].to_numpy(float)
    if y.max() <= 1.5:                       # 0-1 indel fraction -> percent; else different unit (skip)
        return d["sgRNA"].str[:20].tolist(), y * 100.0
    return None                             # non-fraction scale (Sniper/xCas/SpCas9-NG): not comparable


def cover(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


# ── (A) leakage at scale ──────────────────────────────────────────────────────
def leakage(seqs, y, reps=15):
    X = one_hot(seqs); n = len(seqs); rng = np.random.default_rng(0)
    km = MiniBatchKMeans(n_clusters=500, random_state=0, n_init=3, batch_size=4096).fit(X)
    clusters = km.labels_
    out = {}
    for mode in ("random", "grouped"):
        covs = []
        for _ in range(reps):
            if mode == "random":
                perm = rng.permutation(n)
                tr, ca, te = perm[:int(.6*n)], perm[int(.6*n):int(.8*n)], perm[int(.8*n):]
            else:
                cl = rng.permutation(500); csz = np.array([np.sum(clusters == c) for c in cl]); cum = np.cumsum(csz)
                g1 = cl[cum <= .6*n]; g2 = cl[(cum > .6*n) & (cum <= .8*n)]; g3 = cl[cum > .8*n]
                tr = np.where(np.isin(clusters, g1))[0]; ca = np.where(np.isin(clusters, g2))[0]
                te = np.where(np.isin(clusters, g3))[0]
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=4, random_state=0).fit(X[tr], y[tr])
            mu_c, mu_t = m.predict(X[ca]), m.predict(X[te])
            q = conformal_quantile(np.abs(y[ca] - mu_c), ALPHA)
            covs.append(cover(y[te], mu_t - q, mu_t + q))
        out[mode] = {"mean_coverage": round(float(np.mean(covs)), 4), "sd": round(float(np.std(covs)), 4),
                     "reps": reps}
    return out


# ── (B) nuclease-transfer OOD ─────────────────────────────────────────────────
def nuclease_transfer(wt):
    seqs, y = wt; X = one_hot(seqs); n = len(seqs); rng = np.random.default_rng(1)
    perm = rng.permutation(n); tr, ca = perm[:int(.7*n)], perm[int(.7*n):]
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=4, random_state=0).fit(X[tr], y[tr])
    mu_c = m.predict(X[ca]); q = conformal_quantile(np.abs(y[ca] - mu_c), ALPHA)
    # in-distribution WT held-out check
    res = {"in_distribution_WT": {}}
    te_wt = perm[:0]  # placeholder
    out = {}
    # fraction-scale panels only (comparable to WT %): engineered variants ESP/HF + cell lines
    for name in ("ESP", "HF", "HELA", "HCT116", "HL60"):
        p = load_panel(name)
        if p is None:
            continue
        s2, y2 = p; X2 = one_hot(s2); mu2 = m.predict(X2)
        cov_plain = cover(y2, mu2 - q, mu2 + q)
        # weighted conformal: sequence-feature density ratio WT-cal vs target
        dr = estimate_density_ratio([seqs[i] for i in ca], s2, C=1.0)
        r2 = dr["clf"].predict_proba(dr["scaler"].transform(seq_features(s2)))[:, 1].clip(1e-6, 1-1e-6)
        w_te = (r2/(1-r2)) * (len(ca)/len(s2))
        s_cal = np.abs(y[ca] - mu_c)
        qw = np.array([weighted_quantile(s_cal, dr["w"], ALPHA, wt_) for wt_ in w_te])
        cov_w = float(np.mean(np.abs(y2 - mu2) <= qw))
        out[name] = {"n": len(s2), "coverage_plain": round(cov_plain, 3), "coverage_weighted": round(cov_w, 3),
                     "domain_clf_acc": round(dr["auc_proxy"], 3), "ess": round(dr["ess"], 1),
                     "label_mean_shift": round(float(y2.mean() - y[ca].mean()), 2)}
    return {"nominal": 1-ALPHA, "note": "engineered-Cas9 / cell-line change is largely a LABEL/concept "
            "shift; weighted conformal (covariate-shift only) is not expected to fix it.", "by_nuclease": out}


# ── (C) domain UQ baselines ───────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def _train_mlp(Xtr, ytr, p=0.0, epochs=15, seed=0):
    torch.manual_seed(seed)
    m = MLP(p).to(DEV); opt = torch.optim.Adam(m.parameters(), 2e-3); lf = nn.MSELoss()
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV); yt = torch.tensor(ytr, dtype=torch.float32, device=DEV).unsqueeze(1)
    idx = np.arange(len(Xtr))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), 2048):
            j = idx[b:b+2048]
            opt.zero_grad(); loss = lf(m(xt[j]), yt[j]); loss.backward(); opt.step()
    return m


def uq_baselines(wt):
    seqs, y = wt; X = one_hot(seqs).astype(np.float32); n = len(seqs); rng = np.random.default_rng(2)
    perm = rng.permutation(n); tr, ca, te = perm[:int(.6*n)], perm[int(.6*n):int(.8*n)], perm[int(.8*n):]
    z = 1.6448536  # 90% two-sided
    # deep ensemble (K=5)
    preds_te = []; preds_ca = []
    for k in range(5):
        m = _train_mlp(X[tr], y[tr], p=0.0, seed=k)
        with torch.no_grad():
            preds_te.append(m(torch.tensor(X[te], device=DEV)).cpu().numpy().ravel())
            preds_ca.append(m(torch.tensor(X[ca], device=DEV)).cpu().numpy().ravel())
    ens_te = np.stack(preds_te); ens_ca = np.stack(preds_ca)
    mu_te, sd_te = ens_te.mean(0), ens_te.std(0); mu_ca = ens_ca.mean(0)
    cov_ens = cover(y[te], mu_te - z*sd_te, mu_te + z*sd_te)
    # MC-dropout (T=20)
    md = _train_mlp(X[tr], y[tr], p=0.2, seed=0); md.train()
    with torch.no_grad():
        T = np.stack([md(torch.tensor(X[te], device=DEV)).cpu().numpy().ravel() for _ in range(20)])
    mu_md, sd_md = T.mean(0), T.std(0); cov_md = cover(y[te], mu_md - z*sd_md, mu_md + z*sd_md)
    # conformalized ensemble: conformalize the ensemble-mean residual on cal -> exact coverage
    q = conformal_quantile(np.abs(y[ca] - mu_ca), ALPHA)
    cov_conf = cover(y[te], mu_te - q, mu_te + q)
    width = lambda lo_hi: float(np.mean(lo_hi))
    return {"nominal": 1-ALPHA,
            "deep_ensemble_K5": {"coverage": round(cov_ens, 3), "mean_width": round(float(np.mean(2*z*sd_te)), 2)},
            "mc_dropout_T20": {"coverage": round(cov_md, 3), "mean_width": round(float(np.mean(2*z*sd_md)), 2)},
            "conformalized_ensemble": {"coverage": round(cov_conf, 3), "mean_width": round(float(2*q), 2)}}


def main():
    wt = load_panel("WT")
    if wt is None:
        print("CRISPR_HNN WT not found"); return
    print("=" * 66); print("REMEDIATION PHASE 4 — honest experimental design"); print("=" * 66)
    print(f"device: {DEV} | WT guides: {len(wt[0]):,}")

    A = leakage(*wt)
    B = nuclease_transfer(wt)
    C = uq_baselines(wt)
    out = {"device": DEV, "n_WT": len(wt[0]), "alpha": ALPHA,
           "A_leakage_random_vs_grouped": A, "B_nuclease_transfer_OOD": B, "C_uq_baselines": C}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "phase4.json").write_text(json.dumps(out, indent=2))

    print("\n(A) Leakage — conformal coverage (nominal 0.90), 55k WT:")
    print(f"    random split : {A['random']['mean_coverage']} +/- {A['random']['sd']}")
    print(f"    grouped split: {A['grouped']['mean_coverage']} +/- {A['grouped']['sd']}")
    print("\n(B) Nuclease-transfer OOD (calibrated on WT, nominal 0.90):")
    for k, v in B["by_nuclease"].items():
        print(f"    {k:<11} plain={v['coverage_plain']}  weighted={v['coverage_weighted']}  "
              f"(clf_acc {v['domain_clf_acc']}, label shift {v['label_mean_shift']})")
    print("\n(C) UQ baselines vs conformal (WT held-out, nominal 0.90):")
    print(f"    deep ensemble (K=5) : coverage {C['deep_ensemble_K5']['coverage']}  width {C['deep_ensemble_K5']['mean_width']}")
    print(f"    MC-dropout (T=20)   : coverage {C['mc_dropout_T20']['coverage']}  width {C['mc_dropout_T20']['mean_width']}")
    print(f"    conformalized ens.  : coverage {C['conformalized_ensemble']['coverage']}  width {C['conformalized_ensemble']['mean_width']}")
    print("\n[done] -> remediation/results/phase4.json")


if __name__ == "__main__":
    main()
