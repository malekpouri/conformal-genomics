#!/usr/bin/env python3
"""
Referee audit A2 — reimplement the deep-ensemble and MC-dropout baselines correctly (roadmap R4).

The defect: remediation/phase4_honest_design.py:146-158 trains ensemble members with MSELoss and
forms the interval as mu +- 1.645 * sd(member means).  That is the EPISTEMIC spread only, with no
aleatoric/observation-noise term, so the interval has mean width 0.95 on a 0-100 indel scale and
its coverage (0.024) is arithmetically forced.  The reference methods both include a predictive
noise term:
  * deep ensembles (Lakshminarayanan et al., NeurIPS 2017): each member has a Gaussian NLL head
    (mu_k, sigma_k^2); the ensemble is a uniform mixture, so
    mu* = mean(mu_k),  sigma*^2 = mean(sigma_k^2 + mu_k^2) - mu*^2.
  * MC-dropout (Gal & Ghahramani, ICML 2016): predictive variance = sample variance of the
    stochastic forward passes PLUS the observation-noise term.

This script reports, on the same split and the same network, all four variants:
  (1) the published epistemic-only ensemble            [reproduces the 0.024 artifact]
  (2) a correct NLL deep ensemble (mixture variance)
  (3) the published epistemic-only MC-dropout          [reproduces the 0.538 artifact]
  (4) a correct MC-dropout with observation noise
plus split-conformal on the ensemble mean, and conformalized intervals for the corrected baselines.

CPU-only.  Output: audit/results/a2_uq_baselines.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.featurize import one_hot                                     # noqa: E402
from src.scores import conformal_quantile                                    # noqa: E402

HNN = DS / "Extra_Metadata" / "CRISPR_HNN"
ALPHA = 0.10
Z = 1.6448536
DEV = "cpu"
K = 5
T_MC = 20
EPOCHS = 20


def load_wt():
    f = sorted(HNN.glob("WT*"))[0]
    d = pd.read_csv(f); d.columns = ["sgRNA", "indel"]
    d = d[d["sgRNA"].str.len() >= 20]
    return d["sgRNA"].str[:20].tolist(), d["indel"].to_numpy(float) * 100.0


class MLP(nn.Module):
    """Shared trunk; `heteroscedastic` adds a log-variance head (Gaussian NLL)."""

    def __init__(self, p=0.0, heteroscedastic=False):
        super().__init__()
        self.het = heteroscedastic
        self.trunk = nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Dropout(p),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p))
        self.mu = nn.Linear(64, 1)
        self.logvar = nn.Linear(64, 1) if heteroscedastic else None

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu(h)
        if not self.het:
            return mu, None
        # targets are standardised during training, so sigma ~ 1 is the right scale:
        # clamp log-variance to [-6, 4] => sigma in [0.05, 7.4] standardised units
        return mu, self.logvar(h).clamp(-6.0, 4.0)


def train(Xtr, ytr, p=0.0, het=False, seed=0, epochs=EPOCHS, warmup=5):
    """Train on STANDARDISED targets; NLL training is preceded by an MSE warm-up on the mean head.

    Training the Gaussian-NLL objective directly on 0-100 targets with a 2e-3 learning rate makes
    the variance head run away to its clamp (loss is minimised early by inflating sigma, after
    which the gradient on mu vanishes).  Standardising y and warming mu up with MSE is the standard
    remedy and is what makes the baseline a fair one.
    """
    torch.manual_seed(seed)
    ym, ys = float(np.mean(ytr)), float(np.std(ytr)) or 1.0
    m = MLP(p, het).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEV).unsqueeze(1)
    idx = np.arange(len(Xtr)); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(idx)
        use_nll = het and ep >= warmup
        for b in range(0, len(idx), 2048):
            j = idx[b:b + 2048]
            opt.zero_grad()
            mu, lv = m(xt[j])
            if use_nll:                               # Gaussian negative log-likelihood
                loss = (0.5 * lv + 0.5 * (yt[j] - mu) ** 2 / lv.exp()).mean()
            else:
                loss = ((yt[j] - mu) ** 2).mean()
            loss.backward(); opt.step()
    m._ym, m._ys = ym, ys                             # de-standardisation constants
    return m


def train_published(Xtr, ytr, p=0.0, seed=0, epochs=15):
    """Faithful reproduction of remediation/phase4_honest_design.py:_train_mlp.

    Kept verbatim (MSE loss, raw 0-100 targets, np.random.shuffle, 15 epochs) so that the published
    coverage number is reproduced under its own training recipe; otherwise a change in the interval
    formula and a change in training would be confounded.
    """
    torch.manual_seed(seed)
    m = MLP(p, False).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV).unsqueeze(1)
    idx = np.arange(len(Xtr))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), 2048):
            j = idx[b:b + 2048]
            opt.zero_grad()
            mu, _ = m(xt[j])
            ((yt[j] - mu) ** 2).mean().backward(); opt.step()
    m._ym, m._ys = 0.0, 1.0
    return m


def predict(m, X, train_mode=False, reps=1):
    """Returns (mu, var) in ORIGINAL label units."""
    m.train() if train_mode else m.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    mus, vars_ = [], []
    with torch.no_grad():
        for _ in range(reps):
            mu, lv = m(xt)
            mus.append(mu.cpu().numpy().ravel() * m._ys + m._ym)
            vars_.append((lv.exp().cpu().numpy().ravel() * m._ys ** 2) if lv is not None
                         else np.zeros(len(X), dtype=float))
    return np.stack(mus), np.stack(vars_)


def cover(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def report(y, mu, var, label, nominal_z=Z):
    sd = np.sqrt(np.maximum(var, 0.0))
    return {"method": label, "coverage": round(cover(y, mu - nominal_z * sd, mu + nominal_z * sd), 3),
            "mean_width": round(float(np.mean(2 * nominal_z * sd)), 2)}


def main():
    seqs, y = load_wt()
    X = one_hot(seqs).astype(np.float32)
    n = len(seqs)
    rng = np.random.default_rng(2)                       # same seed/split as phase4
    perm = rng.permutation(n)
    tr, ca, te = perm[:int(.6 * n)], perm[int(.6 * n):int(.8 * n)], perm[int(.8 * n):]
    print(f"[a2] WT n={n:,}  train={len(tr):,} cal={len(ca):,} test={len(te):,}  nominal={1-ALPHA}")

    res = {"dataset": {"source": "CRISPR_HNN WT indel (%)", "n": int(n),
                       "n_train": int(len(tr)), "n_cal": int(len(ca)), "n_test": int(len(te))},
           "nominal": 1 - ALPHA, "label_sd": round(float(y.std()), 2)}

    # ---------------- (1)+(2) deep ensembles: MSE members vs Gaussian-NLL members ----------------
    # faithful reproduction of the published arm, under its own training recipe
    np.random.seed(0)
    pub_te = []
    for k in range(K):
        m_pub = train_published(X[tr], y[tr], p=0.0, seed=k)
        pub_te.append(predict(m_pub, X[te])[0][0])
    pub_te = np.stack(pub_te)
    res["deep_ensemble_published_verbatim_recipe"] = report(
        y[te], pub_te.mean(0), pub_te.var(0),
        "phase4 recipe verbatim: MSE members on raw targets, mu +- z*sd(member means)")
    m_pub_md = train_published(X[tr], y[tr], p=0.2, seed=0)
    pub_md, _ = predict(m_pub_md, X[te], train_mode=True, reps=T_MC)
    res["mc_dropout_published_verbatim_recipe"] = report(
        y[te], pub_md.mean(0), pub_md.var(0),
        "phase4 recipe verbatim: MSE dropout net, mu +- z*sd(T passes)")
    print(f"     published recipe verbatim: ensemble {res['deep_ensemble_published_verbatim_recipe']['coverage']}"
          f" / dropout {res['mc_dropout_published_verbatim_recipe']['coverage']}")

    mse_mu_te, mse_mu_ca = [], []
    nll_mu_te, nll_var_te, nll_mu_ca, nll_var_ca = [], [], [], []
    for k in range(K):
        m_mse = train(X[tr], y[tr], p=0.0, het=False, seed=k)
        mu_te, _ = predict(m_mse, X[te]); mu_ca, _ = predict(m_mse, X[ca])
        mse_mu_te.append(mu_te[0]); mse_mu_ca.append(mu_ca[0])
        m_nll = train(X[tr], y[tr], p=0.0, het=True, seed=100 + k)
        mu_te, v_te = predict(m_nll, X[te]); mu_ca, v_ca = predict(m_nll, X[ca])
        nll_mu_te.append(mu_te[0]); nll_var_te.append(v_te[0])
        nll_mu_ca.append(mu_ca[0]); nll_var_ca.append(v_ca[0])
        print(f"     member {k+1}/{K} trained (MSE + NLL)")

    mse_mu_te = np.stack(mse_mu_te); mse_mu_ca = np.stack(mse_mu_ca)
    ens_mu = mse_mu_te.mean(0); ens_epistemic_var = mse_mu_te.var(0)
    res["deep_ensemble_published_epistemic_only"] = report(
        y[te], ens_mu, ens_epistemic_var, "mu +- z*sd(member means), MSE members (as published)")

    nll_mu_te = np.stack(nll_mu_te); nll_var_te = np.stack(nll_var_te)
    mix_mu = nll_mu_te.mean(0)
    mix_var = (nll_var_te + nll_mu_te ** 2).mean(0) - mix_mu ** 2      # Lakshminarayanan mixture
    res["deep_ensemble_corrected_gaussian_nll_mixture"] = report(
        y[te], mix_mu, mix_var, "Gaussian-NLL deep ensemble, mixture predictive variance")

    # ---------------- (3)+(4) MC-dropout: epistemic-only vs with observation noise ----------------
    m_md = train(X[tr], y[tr], p=0.2, het=False, seed=0)
    md_mu, _ = predict(m_md, X[te], train_mode=True, reps=T_MC)
    md_mu_ca, _ = predict(m_md, X[ca], train_mode=True, reps=T_MC)
    res["mc_dropout_published_epistemic_only"] = report(
        y[te], md_mu.mean(0), md_mu.var(0), "mu +- z*sd(T stochastic passes) (as published)")
    # observation noise tau^-1 estimated on the calibration split (Gal & Ghahramani eq. 8)
    tau_inv = float(np.mean((y[ca] - md_mu_ca.mean(0)) ** 2) - np.mean(md_mu_ca.var(0)))
    tau_inv = max(tau_inv, 1e-6)
    res["mc_dropout_corrected_with_observation_noise"] = report(
        y[te], md_mu.mean(0), md_mu.var(0) + tau_inv,
        "MC-dropout predictive variance incl. observation noise (tau^-1 from calibration split)")
    res["mc_dropout_corrected_with_observation_noise"]["tau_inverse"] = round(tau_inv, 2)
    # heteroscedastic dropout net trained by NLL (the fully correct version)
    m_md_nll = train(X[tr], y[tr], p=0.2, het=True, seed=0)
    hmu, hvar = predict(m_md_nll, X[te], train_mode=True, reps=T_MC)
    res["mc_dropout_corrected_nll_head"] = report(
        y[te], hmu.mean(0), hvar.mean(0) + hmu.var(0),
        "MC-dropout with Gaussian-NLL head: E[sigma^2] + Var[mu]")

    # ---------------- conformal, applied to the SAME predictors ----------------
    q_mse = conformal_quantile(np.abs(y[ca] - mse_mu_ca.mean(0)), ALPHA)
    res["conformalized_mse_ensemble_mean"] = {
        "method": "split conformal on the MSE ensemble mean",
        "coverage": round(cover(y[te], ens_mu - q_mse, ens_mu + q_mse), 3),
        "mean_width": round(float(2 * q_mse), 2)}
    nll_mu_ca = np.stack(nll_mu_ca); nll_var_ca = np.stack(nll_var_ca)
    mix_mu_ca = nll_mu_ca.mean(0)
    mix_var_ca = (nll_var_ca + nll_mu_ca ** 2).mean(0) - mix_mu_ca ** 2
    sd_ca = np.sqrt(np.maximum(mix_var_ca, 1e-9)); sd_te = np.sqrt(np.maximum(mix_var, 1e-9))
    q_std = conformal_quantile(np.abs(y[ca] - mix_mu_ca) / sd_ca, ALPHA)     # normalised score
    res["conformalized_nll_ensemble_normalised"] = {
        "method": "split conformal with a variance-normalised score on the NLL ensemble",
        "coverage": round(cover(y[te], mix_mu - q_std * sd_te, mix_mu + q_std * sd_te), 3),
        "mean_width": round(float(np.mean(2 * q_std * sd_te)), 2),
        "note": "adaptive width; conformal's remaining advantage is validity without a "
                "distributional assumption, not a coverage gap"}

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "a2_uq_baselines.json").write_text(json.dumps(res, indent=2))
    print("\n[a2] coverage at nominal 0.90 (label sd = %.1f):" % y.std())
    for k, v in res.items():
        if isinstance(v, dict) and "coverage" in v:
            print(f"     {v['coverage']:.3f}  width {v['mean_width']:8.2f}  {k}")
    print("\n[a2] -> audit/results/a2_uq_baselines.json")


if __name__ == "__main__":
    main()
