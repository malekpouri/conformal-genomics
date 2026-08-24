#!/usr/bin/env python3
"""
ConformalGen — Phase 3: real-data calibration & coverage check.

Fits predictors on D_tr (or loads the Phase-2 artifact), calibrates on D_cal, and evaluates
empirical marginal coverage + interval width on D_te for all four RFC-001 non-conformity scores,
at nominal alpha in {0.10, 0.05}. Because a single 200-guide test split is a noisy draw, we also
report a resplit-mean coverage: the predictor is held fixed (trained on D_tr) while the pooled
D_cal u D_te guides are repartitioned 500x into calibration/test — the valid conformal Monte-Carlo
that estimates E[coverage] (rigorous multi-resplit RQ1 is Phase 5).

CPU-only, 0 GB VRAM. Output: results/json/phase3_calibration.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.predictors import PropertyPredictors                     # noqa: E402
from src.models.featurize import featurize                               # noqa: E402
from src.scores import conformal_quantile                                # noqa: E402

DATA = PROJECT / "data"
ALPHAS = (0.10, 0.05)
N_RESPLIT = 500
SEED = 0


def _load(split):
    d = pd.read_csv(DATA / "splits" / f"{split}.csv")
    return d["seq"].tolist(), d["y_eff"].to_numpy(float), d["off_id"].to_numpy(float)


def _predictors():
    import joblib
    ck = PROJECT / "models" / "predictors.joblib"
    if ck.exists():
        return joblib.load(ck)
    seq, eff, off = _load("train")
    return PropertyPredictors().fit(seq, eff, off, alphas=ALPHAS)


def main():
    pp = _predictors()
    cal_seq, cal_eff, cal_off = _load("calibration")
    te_seq, te_eff, te_off = _load("test")

    # pool D_cal u D_te; precompute predictions/scores once (predictor fixed) -> cheap resplits
    seq = cal_seq + te_seq
    y_eff = np.concatenate([cal_eff, te_eff]); y_off = np.concatenate([cal_off, te_off])
    z_off = np.log1p(y_off)
    Xf = featurize(seq)
    mu_eff = pp.eff.predict_mu(seq)
    zmu_off = pp.off.mu.predict(Xf)                       # log1p-space mean
    sig_eff = pp.eff.predict_scale(seq)
    sig_off = pp.off.predict_scale(seq)                   # log1p-space scale
    n_pool = len(seq); n_cal = len(cal_seq)

    def score_arrays(a):
        qlo_e, qhi_e = pp.eff.predict_quantiles(seq, a)   # original scale (efficacy)
        zqlo_o = pp.off.q_lo[a].predict(Xf); zqhi_o = pp.off.q_hi[a].predict(Xf)  # log1p (off)
        return {
            "abs_eff": np.abs(y_eff - mu_eff),
            "abs_off": np.abs(z_off - zmu_off),
            "signed_lower_eff": mu_eff - y_eff,
            "signed_upper_off": z_off - zmu_off,
            "cqr_eff": np.maximum(qlo_e - y_eff, y_eff - qhi_e),
            "cqr_off": np.maximum(zqlo_o - z_off, z_off - zqhi_o),
            "mo_joint": np.maximum((mu_eff - y_eff) / sig_eff, (z_off - zmu_off) / sig_off),
        }, (qlo_e, qhi_e, zqlo_o, zqhi_o)

    rng = np.random.default_rng(SEED)
    families = ["abs_eff", "abs_off", "signed_lower_eff", "signed_upper_off", "cqr_eff", "cqr_off", "mo_joint"]
    results = {}
    for a in ALPHAS:
        s, (qlo_e, qhi_e, zqlo_o, zqhi_o) = score_arrays(a)
        nominal = 1 - a
        # fixed split: original cal (first n_cal) vs test (rest)
        cal_idx0 = np.arange(n_cal); te_idx0 = np.arange(n_cal, n_pool)
        fam_res = {}
        for f in families:
            q_fixed = conformal_quantile(s[f][cal_idx0], a)
            cov_fixed = float(np.mean(s[f][te_idx0] <= q_fixed))
            # resplit-mean coverage (predictor fixed; repartition pool)
            covs = []
            for _ in range(N_RESPLIT):
                perm = rng.permutation(n_pool); ci, ti = perm[:n_cal], perm[n_cal:]
                qq = conformal_quantile(s[f][ci], a)
                covs.append(np.mean(s[f][ti] <= qq))
            fam_res[f] = {"nominal": round(nominal, 4),
                          "coverage_fixed_split": round(cov_fixed, 4),
                          "coverage_resplit_mean": round(float(np.mean(covs)), 4),
                          "coverage_resplit_sd": round(float(np.std(covs)), 4),
                          "q_fixed": round(float(q_fixed), 4)}
        # interval widths on the fixed split (test rows)
        q = {f: conformal_quantile(s[f][cal_idx0], a) for f in families}
        te = te_idx0
        widths = {
            "abs_eff_width": round(float(2 * q["abs_eff"]), 3),
            "abs_off_width_count": round(float(np.mean(np.expm1(zmu_off[te] + q["abs_off"]) -
                                                       np.expm1(zmu_off[te] - q["abs_off"]))), 3),
            "signed_lower_eff_slack": round(float(q["signed_lower_eff"]), 3),
            "signed_lower_eff_mean_L": round(float(np.mean(mu_eff[te] - q["signed_lower_eff"])), 3),
            "signed_upper_off_mean_U_count": round(float(np.mean(np.expm1(zmu_off[te] + q["signed_upper_off"]))), 3),
            "cqr_eff_width": round(float(np.mean((qhi_e[te] + q["cqr_eff"]) - (qlo_e[te] - q["cqr_eff"]))), 3),
            "cqr_off_width_count": round(float(np.mean(np.expm1(zqhi_o[te] + q["cqr_off"]) -
                                                       np.expm1(np.maximum(zqlo_o[te] - q["cqr_off"], 0)))), 3),
            "mo_eff_mean_L": round(float(np.mean(mu_eff[te] - q["mo_joint"] * sig_eff[te])), 3),
            "mo_off_mean_U_count": round(float(np.mean(np.expm1(zmu_off[te] + q["mo_joint"] * sig_off[te]))), 3),
        }
        results[f"alpha={a}"] = {"coverage": fam_res, "widths_fixed_split": widths}

    out = {"alphas": list(ALPHAS), "n_cal": n_cal, "n_test": len(te_seq), "n_pool_resplit": n_pool,
           "n_resplits": N_RESPLIT, "vram_gb": 0.0,
           "note": ("coverage_resplit_mean holds the D_tr-trained predictor fixed and repartitions "
                    "the D_cal u D_te pool; it estimates E[coverage]. Off-target scores are on the "
                    "log1p scale (coverage is scale-invariant under the monotone transform)."),
           "results": results}
    (PROJECT / "results" / "json").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results" / "json" / "phase3_calibration.json").write_text(json.dumps(out, indent=2))

    # ---- console report ----
    print(f"{'family':<20}{'alpha':>6}{'nominal':>9}{'cov(fixed)':>12}{'cov(resplit)':>14}{'sd':>7}")
    for a in ALPHAS:
        for f in families:
            r = results[f"alpha={a}"]["coverage"][f]
            print(f"{f:<20}{a:>6}{r['nominal']:>9}{r['coverage_fixed_split']:>12}"
                  f"{r['coverage_resplit_mean']:>14}{r['coverage_resplit_sd']:>7}")
    print("\n[done] -> results/json/phase3_calibration.json")


if __name__ == "__main__":
    main()
