#!/usr/bin/env python3
"""
ConformalGen — Phase 2 (revision): comprehensive ablation + selective error control + rigor.

Produces results/json/phase2_comprehensive_ablation.json with:
  1. base predictor metrics (efficacy: RMSE/MAE/R2/Spearman; off-target: logRMSE/Spearman/AUC)
  2. interval baselines (uncalibrated QR, parametric Gaussian, standard split-CP) with bootstrap CIs
  3. score-family ablation {absolute, directional, CQR, joint inf-norm} x {split-CP, +Mondrian}
     (coverage + bootstrap 95% CI, width, yield) at alpha in {0.10, 0.05}
  4. weighted conformal under a *principled* sequence-feature density ratio (generative shift) +
     sensitivity over classifier regularization
  5. leakage analysis: standard guide-level split vs sequence-cluster grouped split
  6. conformal selection: finite-sample FAR control at q in {0.05, 0.10} (post-selection guarantee)

CPU-only, 0 GB VRAM.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.predictors import PropertyPredictors                         # noqa: E402
from src.models.featurize import featurize                                   # noqa: E402
from src.scores import conformal_quantile                                    # noqa: E402
from src.conformal import MondrianConformal, WeightedConformal               # noqa: E402
from src.stats_utils import bootstrap_ci, paired_delta_ci                    # noqa: E402
from src.baselines import uncalibrated_qr, gaussian_interval, standard_split_cp  # noqa: E402
from src.density_ratio import estimate_density_ratio, seq_features, sensitivity  # noqa: E402
from src.selection import ConformalSelector, bh_select                       # noqa: E402
from src.guided_generation import ReservoirGenerator                         # noqa: E402

ALPHAS = (0.10, 0.05)
TAU_EFF, TAU_OFF = 50.0, 12.0
QS = (0.05, 0.10)
SEED = 0


def _load(split, folder="splits"):
    d = pd.read_csv(PROJECT / "data" / folder / f"{split}.csv")
    return d["seq"].tolist(), d["y_eff"].to_numpy(float), d["off_id"].to_numpy(float)


def _cov_ci(mask, seed=SEED):
    ci = bootstrap_ci(np.asarray(mask, float), seed=seed)
    return {"coverage": round(ci["estimate"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)]}


# ── 1. predictor metrics ─────────────────────────────────────────────────────
def predictor_metrics(pp, te_seq, te_eff, te_off):
    mu = pp.eff.predict_mu(te_seq); z = np.log1p(te_off); zmu = pp.off.mu.predict(featurize(te_seq))
    rmse = float(np.sqrt(np.mean((te_eff - mu) ** 2))); mae = float(np.mean(np.abs(te_eff - mu)))
    r2 = float(1 - np.sum((te_eff - mu) ** 2) / np.sum((te_eff - te_eff.mean()) ** 2))
    sp_eff = float(spearmanr(mu, te_eff).statistic)
    log_rmse = float(np.sqrt(np.mean((z - zmu) ** 2))); count_rmse = float(np.sqrt(np.mean((te_off - np.expm1(zmu)) ** 2)))
    sp_off = float(spearmanr(zmu, z).statistic)
    y_bin = (te_off > TAU_OFF).astype(int)
    auc = float(roc_auc_score(y_bin, zmu)) if 0 < y_bin.sum() < len(y_bin) else float("nan")
    return {"efficacy": {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 3),
                         "Spearman": round(sp_eff, 3), "label_sd": round(float(te_eff.std(ddof=1)), 3)},
            "offtarget": {"logRMSE": round(log_rmse, 3), "countRMSE": round(count_rmse, 2),
                          "Spearman": round(sp_off, 3), f"AUC(off>{int(TAU_OFF)})": round(auc, 3)}}


# ── 2. interval baselines ────────────────────────────────────────────────────
def interval_baselines(pp, cal_seq, cal_eff, cal_off, te_seq, te_eff, te_off):
    out = {}
    z_cal, z_te = np.log1p(cal_off), np.log1p(te_off)
    mu_c, mu_t = pp.eff.predict_mu(cal_seq), pp.eff.predict_mu(te_seq)
    sig_t = pp.eff.predict_scale(te_seq)
    zmu_c, zmu_t = pp.off.mu.predict(featurize(cal_seq)), pp.off.mu.predict(featurize(te_seq))
    zsig_t = pp.off.predict_scale(te_seq)
    for a in ALPHAS:
        qlo_t, qhi_t = pp.eff.predict_quantiles(te_seq, a)
        zqlo_t = pp.off.q_lo[a].predict(featurize(te_seq)); zqhi_t = pp.off.q_hi[a].predict(featurize(te_seq))
        out[f"alpha={a}"] = {
            "efficacy": [uncalibrated_qr(te_eff, qlo_t, qhi_t),
                         gaussian_interval(te_eff, mu_t, sig_t, a),
                         standard_split_cp(cal_eff, mu_c, te_eff, mu_t, a)],
            "offtarget": [uncalibrated_qr(z_te, zqlo_t, zqhi_t, count_scale=True) | {"scale": "count"},
                          gaussian_interval(z_te, zmu_t, zsig_t, a, count_scale=True) | {"scale": "count"},
                          standard_split_cp(z_cal, zmu_c, z_te, zmu_t, a, count_scale=True) | {"scale": "count"}],
        }
    return out


# ── 3. score-family ablation (split-CP vs +Mondrian) ─────────────────────────
def _terciles(v):
    e = np.quantile(v, [1/3, 2/3]); return np.digitize(v, e)


def score_family_ablation(pp, cal_seq, cal_eff, cal_off, te_seq, te_eff, te_off):
    z_cal, z_te = np.log1p(cal_off), np.log1p(te_off)
    mu_c, mu_t = pp.eff.predict_mu(cal_seq), pp.eff.predict_mu(te_seq)
    zmu_c, zmu_t = pp.off.mu.predict(featurize(cal_seq)), pp.off.mu.predict(featurize(te_seq))
    sig_c, sig_t = pp.eff.predict_scale(cal_seq), pp.eff.predict_scale(te_seq)
    zsig_c, zsig_t = pp.off.predict_scale(cal_seq), pp.off.predict_scale(te_seq)
    out = {}
    for a in ALPHAS:
        qlo_c, qhi_c = pp.eff.predict_quantiles(cal_seq, a); qlo_t, qhi_t = pp.eff.predict_quantiles(te_seq, a)
        zql_c = pp.off.q_lo[a].predict(featurize(cal_seq)); zqh_c = pp.off.q_hi[a].predict(featurize(cal_seq))
        zql_t = pp.off.q_lo[a].predict(featurize(te_seq)); zqh_t = pp.off.q_hi[a].predict(featurize(te_seq))

        fam = {}
        # --- absolute (efficacy, two-sided) ---
        q = conformal_quantile(np.abs(cal_eff - mu_c), a)
        cov = (te_eff >= mu_t - q) & (te_eff <= mu_t + q)
        fam["absolute_eff"] = {"split_cp": _cov_ci(cov) | {"width": round(float(2 * q), 3)}}
        gm = MondrianConformal(a).calibrate(np.abs(cal_eff - mu_c), _terciles(mu_c))
        qm = gm.q_for(_terciles(mu_t)); covm = (te_eff >= mu_t - qm) & (te_eff <= mu_t + qm)
        fam["absolute_eff"]["mondrian"] = _cov_ci(covm) | {"width": round(float(2 * qm.mean()), 3)}

        # --- directional (efficacy lower) ---
        q = conformal_quantile(mu_c - cal_eff, a); cov = te_eff >= mu_t - q
        fam["directional_eff"] = {"split_cp": _cov_ci(cov) | {"slack": round(float(q), 3)}}

        # --- directional (off upper) + Mondrian on off strata ---
        q = conformal_quantile(z_cal - zmu_c, a); cov = z_te <= zmu_t + q
        fam["directional_off"] = {"split_cp": _cov_ci(cov) | {"meanU_count": round(float(np.mean(np.expm1(zmu_t + q))), 2)}}
        gm = MondrianConformal(a).calibrate(z_cal - zmu_c, _terciles(zmu_c))
        qm = gm.q_for(_terciles(zmu_t)); covm = z_te <= zmu_t + qm
        fam["directional_off"]["mondrian"] = _cov_ci(covm) | {"meanU_count": round(float(np.mean(np.expm1(zmu_t + qm))), 2)}

        # --- CQR (efficacy) ---
        q = conformal_quantile(np.maximum(qlo_c - cal_eff, cal_eff - qhi_c), a)
        cov = (te_eff >= qlo_t - q) & (te_eff <= qhi_t + q)
        fam["cqr_eff"] = {"split_cp": _cov_ci(cov) | {"width": round(float(np.mean((qhi_t + q) - (qlo_t - q))), 3)}}
        # --- CQR (off) ---
        q = conformal_quantile(np.maximum(zql_c - z_cal, z_cal - zqh_c), a)
        cov = (z_te >= zql_t - q) & (z_te <= zqh_t + q)
        fam["cqr_off"] = {"split_cp": _cov_ci(cov) | {"width_count": round(float(np.mean(np.expm1(zqh_t + q) - np.expm1(np.maximum(zql_t - q, 0)))), 2)}}

        # --- joint inf-norm (both objectives; Prop 3) + Mondrian on off strata ---
        s_mo = np.maximum((mu_c - cal_eff) / sig_c, (z_cal - zmu_c) / zsig_c)
        q = conformal_quantile(s_mo, a)
        L = mu_t - q * sig_t; U = np.expm1(zmu_t + q * zsig_t)
        cov = (te_eff >= L) & (te_off <= U)
        fam["joint_infnorm"] = {"split_cp": _cov_ci(cov) | {"eff_meanL": round(float(L.mean()), 2), "off_meanU_count": round(float(U.mean()), 2)}}
        gm = MondrianConformal(a).calibrate(s_mo, _terciles(zmu_c))
        qm = gm.q_for(_terciles(zmu_t))
        Lm = mu_t - qm * sig_t; Um = np.expm1(zmu_t + qm * zsig_t); covm = (te_eff >= Lm) & (te_off <= Um)
        fam["joint_infnorm"]["mondrian"] = _cov_ci(covm) | {"eff_meanL": round(float(Lm.mean()), 2), "off_meanU_count": round(float(Um.mean()), 2)}

        out[f"alpha={a}"] = fam
    return out


# ── 4. weighted conformal under principled (sequence-feature) generative shift ─
def weighted_conformal_shift(pp, cal_seq, cal_eff, cal_off):
    te_seq, te_eff, te_off = _load("test")
    aligned = ReservoirGenerator(te_seq, te_eff, te_off, tilt=1.2)
    sh_seq, sh_eff, sh_off = aligned.sample(2000, seed=42)              # generative covariate shift
    dr = estimate_density_ratio(cal_seq, sh_seq, C=1.0, seed=SEED)
    w_cal = dr["w"]
    # test-point weights via same classifier
    from src.density_ratio import _BASES  # noqa
    r_sh = dr["clf"].predict_proba(dr["scaler"].transform(seq_features(sh_seq)))[:, 1].clip(1e-6, 1 - 1e-6)
    w_sh = (r_sh / (1 - r_sh)) * (len(cal_seq) / len(sh_seq))

    zmu_c = pp.off.mu.predict(featurize(cal_seq)); z_cal = np.log1p(cal_off)
    zmu_sh = pp.off.mu.predict(featurize(sh_seq)); z_sh = np.log1p(sh_off)
    s_cal = z_cal - zmu_c                                              # directional off-upper score
    s_sh = z_sh - zmu_sh

    def _u_stats(q):                                                   # robust width report (q may be +inf)
        U = np.expm1(zmu_sh + q); fin = np.isfinite(U)
        return {"pct_finite_bound": round(float(fin.mean()), 3),
                "medianU_count_finite": round(float(np.median(U[fin])), 3) if fin.any() else None,
                "meanU_count_finite": round(float(U[fin].mean()), 3) if fin.any() else None}

    out = {}
    for a in ALPHAS:
        q_plain = conformal_quantile(s_cal, a)
        cov_plain = s_sh <= q_plain
        wc = WeightedConformal(a).calibrate(s_cal, w_cal)
        q_w = wc.q_for(w_sh); cov_w = s_sh <= q_w
        out[f"alpha={a}"] = {
            "plain": _cov_ci(cov_plain) | _u_stats(q_plain),
            "weighted": _cov_ci(cov_w) | _u_stats(q_w),
            "coverage_delta_ci": paired_delta_ci(cov_w.astype(float), cov_plain.astype(float)),
        }
    out["density_ratio"] = {"ess": round(dr["ess"], 1), "clf_acc": round(dr["auc_proxy"], 3),
                            "note": ("strong shift (clf acc ~0.93) => low ESS => some weighted bounds are "
                                     "unbounded (q=+inf); reported via pct_finite_bound. Matches R1 Thm-4 "
                                     "honesty clause: weighted CP degrades as the density ratio grows extreme."),
                            "sensitivity_over_C": sensitivity(cal_seq, sh_seq, seed=SEED)}
    return out


# ── 5. leakage: standard vs grouped split ────────────────────────────────────
def leakage_analysis():
    out = {}
    for folder, name in [("splits", "standard_guide_level"), ("splits_grouped", "sequence_cluster_grouped")]:
        tr = _load("train", folder); ca = _load("calibration", folder); te = _load("test", folder)
        pp = PropertyPredictors().fit(*tr, alphas=ALPHAS)
        cal_seq, cal_eff, cal_off = ca; te_seq, te_eff, te_off = te
        mu_c, mu_t = pp.eff.predict_mu(cal_seq), pp.eff.predict_mu(te_seq)
        zmu_c, zmu_t = pp.off.mu.predict(featurize(cal_seq)), pp.off.mu.predict(featurize(te_seq))
        z_cal, z_te = np.log1p(cal_off), np.log1p(te_off)
        res = {}
        for a in ALPHAS:
            q = conformal_quantile(mu_c - cal_eff, a); cov_e = te_eff >= mu_t - q
            q2 = conformal_quantile(z_cal - zmu_c, a); cov_o = z_te <= zmu_t + q2
            res[f"alpha={a}"] = {"directional_eff": _cov_ci(cov_e), "directional_off": _cov_ci(cov_o)}
        out[name] = res
    return out


# ── 6. cfBH conformal selection (rigorous; Jin & Candes 2023) ────────────────
# FAR control is a statement about E[FAR] under exchangeability; estimated by Monte-Carlo over resplits
# of the exchangeable D_cal u D_te pool (T_hat fixed on D_tr). We report the full 4-metric panel
# (FAR, Yield, Precision, Power/recall of true-OK) and a q-grid trade-off. A tilted pool is a
# NEGATIVE CONTROL: exchangeability is broken, so FAR control is not guaranteed there.
SEL_TAU_EFF, SEL_TAU_OFF, SEL_RESPLITS = 45.0, 20.0, 300
SEL_QGRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def _nanmean(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def conformal_selection_cfbh(tr, cal, te):
    sel = ConformalSelector(SEL_TAU_EFF, SEL_TAU_OFF).fit(*tr)
    cal_seq, cal_eff, cal_off = cal; te_seq, te_eff, te_off = te
    seq = list(cal_seq) + list(te_seq)
    eff = np.concatenate([cal_eff, te_eff]); off = np.concatenate([cal_off, te_off])
    n_pool, n_cal = len(seq), len(cal_seq)
    rng = np.random.default_rng(SEED)

    panels = {q: {"FAR": [], "yield": [], "precision": [], "power": []} for q in SEL_QGRID}
    for _ in range(SEL_RESPLITS):
        perm = rng.permutation(n_pool); ci, ti = perm[:n_cal], perm[n_cal:]
        sel.calibrate([seq[i] for i in ci], eff[ci], off[ci])
        tseq = [seq[i] for i in ti]; teff = eff[ti]; toff = off[ti]
        p = sel.pvalues(tseq); ok = (teff >= SEL_TAU_EFF) & (toff <= SEL_TAU_OFF); n_ok = int(ok.sum())
        for q in SEL_QGRID:
            m = bh_select(p, q); ns = int(m.sum())
            panels[q]["FAR"].append(float(np.mean(~ok[m])) if ns else 0.0)
            panels[q]["yield"].append(ns / len(tseq))
            panels[q]["precision"].append(float(np.mean(ok[m])) if ns else np.nan)
            panels[q]["power"].append(float(np.sum(ok & m) / n_ok) if n_ok else np.nan)

    q_grid = []
    for q in SEL_QGRID:
        pn = panels[q]; far_ci = bootstrap_ci(np.array(pn["FAR"]), seed=SEED)
        q_grid.append({"q": q, "mean_FAR": round(_nanmean(pn["FAR"]), 4),
                       "FAR_ci95": [round(far_ci["lo"], 4), round(far_ci["hi"], 4)],
                       "FAR_le_q": bool(_nanmean(pn["FAR"]) <= q + 1e-9),
                       "mean_yield": round(_nanmean(pn["yield"]), 4),
                       "mean_precision": round(_nanmean(pn["precision"]), 4),
                       "mean_power": round(_nanmean(pn["power"]), 4)})

    # negative control on a tilted (non-exchangeable) pool at q=0.10
    sel.calibrate(cal_seq, cal_eff, cal_off)
    sh_seq, sh_eff, sh_off = ReservoirGenerator(te_seq, te_eff, te_off, tilt=1.2).sample(2000, seed=42)
    neg = sel.evaluate(sh_seq, sh_eff, sh_off, 0.10)
    return {"thresholds": {"tau_eff": SEL_TAU_EFF, "tau_off": SEL_TAU_OFF},
            "n_resplits": SEL_RESPLITS, "q_grid": q_grid,
            "levels": {f"q={q}": next(r for r in q_grid if r["q"] == q) for q in QS},
            "shifted_negative_control_q0.10": {"empirical_FAR": round(neg["empirical_FAR"], 4),
                                               "yield": round(neg["yield"], 4),
                                               "note": "exchangeability broken -> FAR control not guaranteed"}}


def main():
    import joblib
    ck = PROJECT / "models" / "predictors.joblib"
    pp = joblib.load(ck) if ck.exists() else PropertyPredictors().fit(*_load("train"), alphas=ALPHAS)
    tr = _load("train"); cal = _load("calibration"); te = _load("test")

    results = {
        "config": {"tau_eff": TAU_EFF, "tau_off": TAU_OFF, "alphas": list(ALPHAS), "far_levels": list(QS),
                   "bootstrap_B": 2000, "vram_gb": 0.0},
        "predictor_metrics": predictor_metrics(pp, *te),
        "interval_baselines": interval_baselines(pp, *cal, *te),
        "score_family_ablation": score_family_ablation(pp, *cal, *te),
        "weighted_conformal_generative_shift": weighted_conformal_shift(pp, *cal),
        "leakage_standard_vs_grouped": leakage_analysis(),
        "conformal_selection_cfBH": conformal_selection_cfbh(tr, cal, te),
    }
    (PROJECT / "results" / "json").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results" / "json" / "phase2_comprehensive_ablation.json").write_text(json.dumps(results, indent=2))

    # ── console report ──
    pm = results["predictor_metrics"]
    print("PREDICTOR:", "eff", pm["efficacy"], "| off", pm["offtarget"])
    print("\nSCORE-FAMILY ABLATION (alpha=0.10):")
    fa = results["score_family_ablation"]["alpha=0.1"]
    for fam, v in fa.items():
        base = v["split_cp"]; mond = v.get("mondrian")
        line = f"  {fam:<16} split-CP cov={base['coverage']:.3f} {base['ci95']}"
        if mond: line += f" | +Mondrian cov={mond['coverage']:.3f} {mond['ci95']}"
        print(line)
    print("\nWEIGHTED CONFORMAL under generative shift:")
    for a in ALPHAS:
        w = results["weighted_conformal_generative_shift"][f"alpha={a}"]
        print(f"  alpha={a}: plain cov={w['plain']['coverage']:.3f} medU={w['plain']['medianU_count_finite']} -> "
              f"weighted cov={w['weighted']['coverage']:.3f} medU={w['weighted']['medianU_count_finite']} "
              f"(%finite {w['weighted']['pct_finite_bound']})")
    print("  density-ratio:", results["weighted_conformal_generative_shift"]["density_ratio"])
    print("\nLEAKAGE (standard vs grouped, alpha=0.10):")
    lk = results["leakage_standard_vs_grouped"]
    for name, r in lk.items():
        rr = r["alpha=0.1"]
        print(f"  {name:<28} dir_eff={rr['directional_eff']['coverage']:.3f} dir_off={rr['directional_off']['coverage']:.3f}")
    sel = results["conformal_selection_cfBH"]
    print(f"\ncfBH CONFORMAL SELECTION (@ tau_eff={sel['thresholds']['tau_eff']}, "
          f"tau_off={sel['thresholds']['tau_off']}; {sel['n_resplits']} resplits):")
    print(f"  {'q':>5}{'FAR':>9}{'FAR<=q':>8}{'yield':>8}{'precision':>11}{'power':>8}")
    for r in sel["q_grid"]:
        print(f"  {r['q']:>5}{r['mean_FAR']:>9.4f}{str(r['FAR_le_q']):>8}{r['mean_yield']:>8.3f}"
              f"{r['mean_precision']:>11.3f}{r['mean_power']:>8.3f}")
    ng = sel["shifted_negative_control_q0.10"]
    print(f"  negative control (tilted, non-exchangeable) q=0.10: FAR={ng['empirical_FAR']:.3f} "
          f"yield={ng['yield']:.3f}")
    print("\n[done] -> results/json/phase2_comprehensive_ablation.json")


if __name__ == "__main__":
    main()
