#!/usr/bin/env python3
"""
ConformalGen — Phase 2: fit base predictors, compute non-conformity scores, report metrics.

Fits (on D_tr only) the point predictors, CQR quantile heads, and conditional-scale models for both
objectives (efficacy, off-target); evaluates predictor quality on D_te (RMSE, pinball, quantile
crossing); computes all four RFC-001 non-conformity score families on D_cal and reports their
dispersion. CPU-only (0 GB VRAM), within the <= 2 GB budget.

Outputs: models/predictors.joblib , results/json/phase2_metrics.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.predictors import PropertyPredictors                 # noqa: E402
from src.scores import (score_abs, score_signed_lower, score_signed_upper,   # noqa: E402
                        score_cqr, score_mo, conformal_quantile, pinball_loss)

DATA = PROJECT / "data"
ALPHAS = (0.10, 0.05)


def _load(split):
    d = pd.read_csv(DATA / "splits" / f"{split}.csv")
    return d["seq"].tolist(), d["y_eff"].to_numpy(float), d["off_id"].to_numpy(float)


def _rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
def _mae(a, b):  return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))
def _disp(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 4), "sd": round(float(x.std(ddof=1)), 4),
            "min": round(float(x.min()), 4), "median": round(float(np.median(x)), 4),
            "max": round(float(x.max()), 4)}


def main():
    tr_seq, tr_eff, tr_off = _load("train")
    te_seq, te_eff, te_off = _load("test")
    cal_seq, cal_eff, cal_off = _load("calibration")

    pp = PropertyPredictors().fit(tr_seq, tr_eff, tr_off, alphas=ALPHAS)
    (PROJECT / "models").mkdir(exist_ok=True)
    joblib.dump(pp, PROJECT / "models" / "predictors.joblib")

    # ---- predictor quality on D_te -----------------------------------------
    mu_eff_te = pp.eff.predict_mu(te_seq)
    mu_off_te = pp.off.predict_mu(te_seq)                     # original (count) scale
    perf = {
        "efficacy_point": {"rmse": round(_rmse(te_eff, mu_eff_te), 3), "mae": round(_mae(te_eff, mu_eff_te), 3),
                           "label_sd": round(float(np.std(te_eff, ddof=1)), 3)},
        "offtarget_point": {"rmse_count": round(_rmse(te_off, mu_off_te), 3),
                            "mae_count": round(_mae(te_off, mu_off_te), 3),
                            "rmse_log1p": round(_rmse(np.log1p(te_off), np.log1p(np.maximum(mu_off_te, 0))), 3),
                            "label_sd_count": round(float(np.std(te_off, ddof=1)), 3)},
        "quantile_pinball": {}, "quantile_crossing_rate": {},
    }
    # quantile heads: pinball loss at each level + crossing rate (monotonicity)
    for a in ALPHAS:
        lo_e, hi_e = pp.eff.predict_quantiles(te_seq, a)
        lo_o, hi_o = pp.off.predict_quantiles(te_seq, a)     # original scale
        perf["quantile_pinball"][f"alpha={a}"] = {
            "eff_lo(tau=%.3f)" % (a/2): round(pinball_loss(te_eff, lo_e, a/2), 3),
            "eff_hi(tau=%.3f)" % (1-a/2): round(pinball_loss(te_eff, hi_e, 1-a/2), 3),
            "off_lo(tau=%.3f)" % (a/2): round(pinball_loss(te_off, lo_o, a/2), 3),
            "off_hi(tau=%.3f)" % (1-a/2): round(pinball_loss(te_off, hi_o, 1-a/2), 3)}
        perf["quantile_crossing_rate"][f"alpha={a}"] = {
            "eff": round(float(np.mean(lo_e > hi_e)), 4), "off": round(float(np.mean(lo_o > hi_o)), 4)}

    # ---- non-conformity score dispersion on D_cal --------------------------
    mu_eff_c = pp.eff.predict_mu(cal_seq)
    mu_off_c = pp.off.predict_mu(cal_seq)                     # count scale
    # for the off objective, scores (b)/(d) are computed on the log1p space the predictor is fit on
    zmu_off_c = pp.off.mu.predict(_feat(cal_seq))             # log1p-space mean prediction
    z_off_c = np.log1p(cal_off)
    sig_eff_c = pp.eff.predict_scale(cal_seq)
    sig_off_c = pp.off.predict_scale(cal_seq)                 # log1p-space scale

    scores = {
        "abs_eff": _disp(score_abs(cal_eff, mu_eff_c)),
        "abs_off_log1p": _disp(score_abs(z_off_c, zmu_off_c)),
        "signed_lower_eff": _disp(score_signed_lower(cal_eff, mu_eff_c)),
        "signed_upper_off_log1p": _disp(score_signed_upper(z_off_c, zmu_off_c)),
        "mo_joint": _disp(score_mo(cal_eff, mu_eff_c, sig_eff_c, z_off_c, zmu_off_c, sig_off_c)),
    }
    cqr = {}
    for a in ALPHAS:
        lo_e, hi_e = pp.eff.predict_quantiles(cal_seq, a)
        zlo_o = pp.off.q_lo[a].predict(_feat(cal_seq)); zhi_o = pp.off.q_hi[a].predict(_feat(cal_seq))
        cqr[f"alpha={a}"] = {"cqr_eff": _disp(score_cqr(cal_eff, lo_e, hi_e)),
                             "cqr_off_log1p": _disp(score_cqr(z_off_c, zlo_o, zhi_o))}
    scores["cqr"] = cqr

    # illustrative calibrated quantiles (what Phase 3 will consume)
    calibrated_q = {}
    for a in ALPHAS:
        calibrated_q[f"alpha={a}"] = {
            "q_signed_lower_eff": round(conformal_quantile(score_signed_lower(cal_eff, mu_eff_c), a), 4),
            "q_signed_upper_off_log1p": round(conformal_quantile(score_signed_upper(z_off_c, zmu_off_c), a), 4),
            "q_mo_joint": round(conformal_quantile(score_mo(cal_eff, mu_eff_c, sig_eff_c, z_off_c, zmu_off_c, sig_off_c), a), 4)}

    out = {"alphas": list(ALPHAS), "n_train": len(tr_seq), "n_cal": len(cal_seq), "n_test": len(te_seq),
           "vram_gb": 0.0, "backend": "scikit-learn HistGradientBoosting (CPU)",
           "predictor_performance": perf, "score_dispersion_on_calibration": scores,
           "illustrative_calibrated_quantiles": calibrated_q}
    (PROJECT / "results" / "json").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results" / "json" / "phase2_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"predictor_performance": perf,
                      "score_dispersion_on_calibration": scores,
                      "illustrative_calibrated_quantiles": calibrated_q}, indent=2))
    print("[done] models/predictors.joblib + results/json/phase2_metrics.json")


def _feat(seqs):
    from src.models.featurize import featurize
    return featurize(seqs)


if __name__ == "__main__":
    main()
