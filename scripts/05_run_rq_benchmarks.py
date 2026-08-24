#!/usr/bin/env python3
"""
ConformalGen — Phase 5: RQ benchmark suite (RFC-001 §4).

RQ1  Marginal coverage validity : >=500 MC resplits of D_cal/D_te at alpha in {0.05,0.10} across all
     four non-conformity scores (absolute / signed directional / CQR / joint inf-norm) per objective;
     report mean coverage, sd, and signed deviation from nominal.
RQ2  Efficiency & set width      : interval widths, hyper-rectangle areas and acceptance yields per
     score family; Mondrian stratification (by predicted off-target magnitude) to compress off-target
     width on the heavy tail.
RQ3  Conditional & OOD transfer  : chromosome-22 holdout —
       (a) plain split-conformal pooled across chromosomes (OOD coverage/width gap on sparse chr22),
       (b) Mondrian / group-conditional across in-distribution chromosomes (conditional validity),
       (c) weighted / covariate-shift conformal (kernel LR on chromosome length) to recover efficient
           valid coverage under the genomic shift.

CPU-only, 0 GB VRAM. Output: results/json/rq_benchmarks.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.models.predictors import PropertyPredictors                        # noqa: E402
from src.models.featurize import featurize                                  # noqa: E402
from src.scores import conformal_quantile                                   # noqa: E402
from src.conformal import MondrianConformal, weighted_quantile              # noqa: E402
from src.guided_generation import ConformalGuidedGenerator, ReservoirGenerator  # noqa: E402
from src.selection import ConformalSelector, target_transform, conformal_pvalues, bh_select  # noqa: E402

DATA = PROJECT / "data"
ALPHAS = (0.10, 0.05)
N_RESPLIT = 500
SEED = 0
HELDOUT_CHROM = "chr22"
DROP_CHROM = {"chrM", "chrY"}          # chrY dropped (near-degenerate); consistent with Phase-1 chrM drop

# GRCh38 chromosome lengths (bp) — the external genomic covariate for the RQ3 shift
CHROM_LEN = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259,
    "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422,
    "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}


def _load(split):
    d = pd.read_csv(DATA / "splits" / f"{split}.csv")
    return d["seq"].tolist(), d["y_eff"].to_numpy(float), d["off_id"].to_numpy(float)


def _predictors():
    import joblib
    ck = PROJECT / "models" / "predictors.joblib"
    if ck.exists():
        return joblib.load(ck)
    return PropertyPredictors().fit(*_load("train"), alphas=ALPHAS)


# ═════════════════════════════════════════════════════════════════════════════
def rq1(pp):
    """Marginal coverage validity across >=500 resplits, all scores, both alphas."""
    cal_seq, cal_eff, cal_off = _load("calibration")
    te_seq, te_eff, te_off = _load("test")
    seq = cal_seq + te_seq
    y_eff = np.concatenate([cal_eff, te_eff]); y_off = np.concatenate([cal_off, te_off])
    z_off = np.log1p(y_off)
    Xf = featurize(seq)
    mu_eff = pp.eff.predict_mu(seq); zmu_off = pp.off.mu.predict(Xf)
    sig_eff = pp.eff.predict_scale(seq); sig_off = pp.off.predict_scale(seq)
    n_pool, n_cal = len(seq), len(cal_seq)
    rng = np.random.default_rng(SEED)
    out = {}
    for a in ALPHAS:
        qlo_e, qhi_e = pp.eff.predict_quantiles(seq, a)
        zqlo_o = pp.off.q_lo[a].predict(Xf); zqhi_o = pp.off.q_hi[a].predict(Xf)
        S = {"abs_eff": np.abs(y_eff - mu_eff), "abs_off": np.abs(z_off - zmu_off),
             "signed_lower_eff": mu_eff - y_eff, "signed_upper_off": z_off - zmu_off,
             "cqr_eff": np.maximum(qlo_e - y_eff, y_eff - qhi_e),
             "cqr_off": np.maximum(zqlo_o - z_off, z_off - zqhi_o),
             "mo_joint": np.maximum((mu_eff - y_eff) / sig_eff, (z_off - zmu_off) / sig_off)}
        perms = [rng.permutation(n_pool) for _ in range(N_RESPLIT)]
        fam = {}
        for f, s in S.items():
            covs = []
            for perm in perms:
                ci, ti = perm[:n_cal], perm[n_cal:]
                q = conformal_quantile(s[ci], a)
                covs.append(np.mean(s[ti] <= q))
            covs = np.asarray(covs)
            fam[f] = {"nominal": round(1 - a, 4), "mean_coverage": round(float(covs.mean()), 4),
                      "sd": round(float(covs.std()), 4),
                      "signed_dev_from_nominal": round(float(covs.mean() - (1 - a)), 4)}
        out[f"alpha={a}"] = fam
    return out


# ═════════════════════════════════════════════════════════════════════════════
def rq2(pp):
    """Efficiency/width/area/yield per score family + Mondrian compression on off-target tail."""
    cal_seq, cal_eff, cal_off = _load("calibration")
    te_seq, te_eff, te_off = _load("test")
    z_cal_off, z_te_off = np.log1p(cal_off), np.log1p(te_off)
    Xc, Xt = featurize(cal_seq), featurize(te_seq)
    mu_eff_c, mu_eff_t = pp.eff.predict_mu(cal_seq), pp.eff.predict_mu(te_seq)
    zmu_off_c, zmu_off_t = pp.off.mu.predict(Xc), pp.off.mu.predict(Xt)
    sig_eff_c, sig_eff_t = pp.eff.predict_scale(cal_seq), pp.eff.predict_scale(te_seq)
    sig_off_c, sig_off_t = pp.off.predict_scale(cal_seq), pp.off.predict_scale(te_seq)

    out = {}
    for a in ALPHAS:
        qlo_e_c, qhi_e_c = pp.eff.predict_quantiles(cal_seq, a)
        qlo_e_t, qhi_e_t = pp.eff.predict_quantiles(te_seq, a)
        zqlo_c, zqhi_c = pp.off.q_lo[a].predict(Xc), pp.off.q_hi[a].predict(Xc)
        zqlo_t, zqhi_t = pp.off.q_lo[a].predict(Xt), pp.off.q_hi[a].predict(Xt)

        # per-family calibrated quantiles
        q_abs_e = conformal_quantile(np.abs(cal_eff - mu_eff_c), a)
        q_abs_o = conformal_quantile(np.abs(z_cal_off - zmu_off_c), a)
        q_dir_e = conformal_quantile(mu_eff_c - cal_eff, a)
        q_dir_o = conformal_quantile(z_cal_off - zmu_off_c, a)
        q_cqr_e = conformal_quantile(np.maximum(qlo_e_c - cal_eff, cal_eff - qhi_e_c), a)
        q_cqr_o = conformal_quantile(np.maximum(zqlo_c - z_cal_off, z_cal_off - zqhi_c), a)
        s_mo_c = np.maximum((mu_eff_c - cal_eff) / sig_eff_c, (z_cal_off - zmu_off_c) / sig_off_c)
        q_mo = conformal_quantile(s_mo_c, a)

        def off_width(zmu, ql, qh, q):   # count-scale two-sided width helper
            return np.mean(np.expm1(qh + q if qh is not None else zmu + q)
                           - np.expm1(np.maximum((ql - q) if ql is not None else (zmu - q), 0)))

        eff_abs_w = float(2 * q_abs_e)
        off_abs_w = float(np.mean(np.expm1(zmu_off_t + q_abs_o) - np.expm1(np.maximum(zmu_off_t - q_abs_o, 0))))
        eff_cqr_w = float(np.mean((qhi_e_t + q_cqr_e) - (qlo_e_t - q_cqr_e)))
        off_cqr_w = float(np.mean(np.expm1(zqhi_t + q_cqr_o) - np.expm1(np.maximum(zqlo_t - q_cqr_o, 0))))
        # directional (one-sided) "widths" = slack; joint uses per-point scaled slack
        eff_dir_w = float(q_dir_e)
        off_dir_w = float(np.mean(np.expm1(zmu_off_t + q_dir_o)))          # mean U_off (count)
        eff_mo_w = float(np.mean(q_mo * sig_eff_t))
        off_mo_w = float(np.mean(np.expm1(zmu_off_t + q_mo * sig_off_t)))

        fam = {
            "absolute":   {"eff_width": round(eff_abs_w, 3), "off_width_count": round(off_abs_w, 3),
                           "hyperrect_area": round(eff_abs_w * off_abs_w, 3)},
            "directional":{"eff_slack": round(eff_dir_w, 3), "off_meanU_count": round(off_dir_w, 3)},
            "cqr":        {"eff_width": round(eff_cqr_w, 3), "off_width_count": round(off_cqr_w, 3),
                           "hyperrect_area": round(eff_cqr_w * off_cqr_w, 3)},
            "joint_infnorm": {"eff_slack_mean": round(eff_mo_w, 3), "off_meanU_count": round(off_mo_w, 3)},
        }

        # ── Mondrian on the off-target heavy tail: stratify by predicted off magnitude (terciles) ──
        edges = np.quantile(zmu_off_c, [1/3, 2/3])
        grp = lambda z: np.digitize(z, edges)                              # 0/1/2 low/med/high
        s_off = z_cal_off - zmu_off_c                                       # signed upper (off) score
        mc = MondrianConformal(a).calibrate(s_off, grp(zmu_off_c))
        s_off_t = z_te_off - zmu_off_t
        q_pool = conformal_quantile(s_off, a)
        gt = grp(zmu_off_t)
        q_m = mc.q_for(gt)
        cov_pool = float(np.mean(s_off_t <= q_pool))
        cov_mond = float(np.mean(s_off_t <= q_m))
        w_pool = float(np.mean(np.expm1(zmu_off_t + q_pool)))
        w_mond = float(np.mean(np.expm1(zmu_off_t + q_m)))
        strata = {}
        for g in (0, 1, 2):
            m = gt == g
            if m.any():
                strata[["low", "med", "high"][g]] = {
                    "n": int(m.sum()), "cov_pooled": round(float(np.mean(s_off_t[m] <= q_pool)), 3),
                    "cov_mondrian": round(float(np.mean(s_off_t[m] <= q_m[m])), 3),
                    "meanU_pooled": round(float(np.mean(np.expm1(zmu_off_t[m] + q_pool))), 2),
                    "meanU_mondrian": round(float(np.mean(np.expm1(zmu_off_t[m] + q_m[m]))), 2)}
        fam["mondrian_offtarget"] = {"nominal": round(1 - a, 3),
                                     "pooled": {"coverage": round(cov_pool, 3), "meanU_count": round(w_pool, 2)},
                                     "mondrian": {"coverage": round(cov_mond, 3), "meanU_count": round(w_mond, 2)},
                                     "by_stratum": strata}
        out[f"alpha={a}"] = fam
    return out


# ═════════════════════════════════════════════════════════════════════════════
def _long_offtarget():
    d = pd.read_csv(DATA / "offtarget_by_chromosome.csv")
    d = d[~d["chrom"].isin(DROP_CHROM)].copy()
    d["len"] = d["chrom"].map(CHROM_LEN).astype(float)
    return d


def rq3():
    """Chromosome-22 OOD: plain vs Mondrian (in-dist) vs weighted covariate-shift conformal."""
    d = _long_offtarget()
    len_mean, len_std = d["len"].mean(), d["len"].std()
    d["len_z"] = (d["len"] - len_mean) / len_std

    def feats(df):
        return np.column_stack([featurize(df["seq"].tolist()), df["len_z"].to_numpy()])

    ind = d[d["chrom"] != HELDOUT_CHROM]                        # in-distribution chromosomes
    ood = d[d["chrom"] == HELDOUT_CHROM]                        # held-out chr22
    tr = ind[ind["split"] == "train"]; cal = ind[ind["split"] == "calibration"]
    ind_te = ind[ind["split"] == "test"]; ood_te = ood[ood["split"] == "test"]

    reg = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                        min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    reg.fit(feats(tr), np.log1p(tr["near_count"].to_numpy(float)))

    def zmu(df):
        return reg.predict(feats(df))

    s_cal = np.log1p(cal["near_count"].to_numpy(float)) - zmu(cal)         # signed-upper (log) score
    s_ind_te = np.log1p(ind_te["near_count"].to_numpy(float)) - zmu(ind_te)
    zmu_ood = zmu(ood_te)
    s_ood = np.log1p(ood_te["near_count"].to_numpy(float)) - zmu_ood

    out = {"heldout": HELDOUT_CHROM, "n_train_pairs": len(tr), "n_cal_pairs": len(cal),
           "n_ind_test_pairs": len(ind_te), "n_ood_test_pairs": len(ood_te)}
    for a in ALPHAS:
        nominal = 1 - a
        # (a) plain split-conformal pooled across in-dist chromosomes, applied to chr22
        q_plain = conformal_quantile(s_cal, a)
        cov_ood_plain = float(np.mean(s_ood <= q_plain))
        width_ood_plain = float(np.mean(np.expm1(zmu_ood + q_plain)))       # mean U_off (count) on chr22

        # (b) Mondrian across in-distribution chromosomes (conditional coverage on known groups)
        mc = MondrianConformal(a).calibrate(s_cal, cal["chrom"].to_numpy())
        q_ind = mc.q_for(ind_te["chrom"].to_numpy())
        cov_ind_mond = float(np.mean(s_ind_te <= q_ind))
        # per-chromosome conditional coverage spread, pooled vs Mondrian
        per_chrom = {}
        for c in sorted(ind_te["chrom"].unique()):
            m = ind_te["chrom"].to_numpy() == c
            per_chrom[c] = {"cov_pooled": round(float(np.mean(s_ind_te[m] <= q_plain)), 3),
                            "cov_mondrian": round(float(np.mean(s_ind_te[m] <= q_ind[m])), 3)}
        worst_pool = min(v["cov_pooled"] for v in per_chrom.values())
        worst_mond = min(v["cov_mondrian"] for v in per_chrom.values())

        # (c) weighted covariate-shift conformal on chr22: kernel LR weights on chromosome length
        h = 0.5 * cal["len_z"].std()
        len_z_cal = cal["len_z"].to_numpy(); len_z_ood = float(ood["len_z"].iloc[0])
        w = np.exp(-((len_z_cal - len_z_ood) ** 2) / (2 * h * h))
        qw = np.array([weighted_quantile(s_cal, w, a, test_weight=1.0) for _ in range(1)])[0]
        cov_ood_weighted = float(np.mean(s_ood <= qw))
        width_ood_weighted = float(np.mean(np.expm1(zmu_ood + qw)))

        out[f"alpha={a}"] = {
            "nominal": round(nominal, 3),
            "a_plain_on_chr22": {"coverage": round(cov_ood_plain, 3), "meanU_count": round(width_ood_plain, 3)},
            "b_mondrian_in_distribution": {"pooled_coverage": round(float(np.mean(s_ind_te <= q_plain)), 3),
                                           "mondrian_coverage": round(cov_ind_mond, 3),
                                           "worst_chrom_cov_pooled": round(worst_pool, 3),
                                           "worst_chrom_cov_mondrian": round(worst_mond, 3),
                                           "per_chromosome": per_chrom},
            "c_weighted_on_chr22": {"coverage": round(cov_ood_weighted, 3),
                                    "meanU_count": round(width_ood_weighted, 3),
                                    "bandwidth_len_z": round(float(h), 3)},
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
def rq_guided_pareto(pp):
    """Yield vs design-threshold Pareto trade-off (for Figure 5), reusing Phase-4 policy."""
    cal_seq, cal_eff, cal_off = _load("calibration")
    te_seq, te_eff, te_off = _load("test")
    gg = ConformalGuidedGenerator(pp).calibrate(cal_seq, cal_eff, cal_off, ALPHAS)
    aligned = ReservoirGenerator(te_seq, te_eff, te_off, tilt=1.2)
    s, ye, yo = aligned.sample(2000, seed=42)
    curves = {}
    for a in ALPHAS:
        rows = []
        for te_ in [40, 45, 50, 55, 60]:
            for to_ in [5, 10, 15, 20, 25]:
                for mode in ("point", "conformal_directional"):
                    r = gg.evaluate(s, ye, yo, te_, to_, a, mode)
                    rows.append({"tau_eff": te_, "tau_off": to_, "mode": mode, "yield": r["yield"],
                                 "design_precision": r["design_precision"],
                                 "bound_coverage": r["post_selection_bound_coverage"]})
        curves[f"alpha={a}"] = rows
    return curves


# ═════════════════════════════════════════════════════════════════════════════
# cfBH conformal selection (Jin & Candes 2023): real-oracle result + semisynthetic power-vs-fidelity.
# Design criterion OK = {y_eff >= SEL_TAU_EFF and y_off <= SEL_TAU_OFF}; controls FAR <= q.
SEL_TAU_EFF, SEL_TAU_OFF = 45.0, 20.0
SEL_MC = 200


def _panel(p, ok, q):
    m = bh_select(p, q); ns = int(m.sum()); nok = int(ok.sum())
    return (float(np.mean(~ok[m])) if ns else 0.0, ns / len(p),
            float(np.mean(ok[m])) if ns else np.nan,
            float(np.sum(ok & m) / nok) if nok else np.nan)


def _nanmean(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return round(float(a.mean()), 4) if a.size else None


def rq_conformal_selection():
    """cfBH FAR control: honest real-oracle panel + semisynthetic power vs oracle fidelity rho."""
    tr = _load("train"); cal = _load("calibration"); te = _load("test")
    sel = ConformalSelector(SEL_TAU_EFF, SEL_TAU_OFF).fit(*tr)
    s_eff, s_off = sel.s_eff, sel.s_off
    # pooled exchangeable set with fixed T (from true labels)
    seq = list(cal[0]) + list(te[0]); eff = np.concatenate([cal[1], te[1]]); off = np.concatenate([cal[2], te[2]])
    T = target_transform(eff, off, SEL_TAU_EFF, SEL_TAU_OFF, s_eff, s_off)
    ok_all = (eff >= SEL_TAU_EFF) & (off <= SEL_TAU_OFF)
    N, n_cal = len(seq), len(cal[0]); rng = np.random.default_rng(SEED)
    Tz = (T - T.mean()) / (T.std() + 1e-12); sdT = T.std()

    # real predictor effective fidelity
    That_te_real = sel._That(te[0]); T_te = target_transform(te[1], te[2], SEL_TAU_EFF, SEL_TAU_OFF, s_eff, s_off)
    rho_real = float(np.corrcoef(That_te_real, T_te)[0, 1])

    # real-oracle cfBH, MC over exchangeable resplits (4-metric panel). The full semisynthetic power/
    # n_cal/m/perfect-oracle sensitivity lives in scripts/09 -> phase4_selection_sensitivity.json (fig7).
    real = {}
    for q in (0.05, 0.10):
        F, Y, P, W, NS = [], [], [], [], []
        for _ in range(SEL_MC):
            perm = rng.permutation(N); ci, ti = perm[:n_cal], perm[n_cal:]
            sel.calibrate([seq[i] for i in ci], eff[ci], off[ci])
            p = sel.pvalues([seq[i] for i in ti])
            f, y, pr, w = _panel(p, ok_all[ti], q); F.append(f); Y.append(y); P.append(pr); W.append(w)
            NS.append(int((bh_select(p, q)).sum()))
        real[f"q={q}"] = {"mean_FAR": _nanmean(F), "FAR_le_q": bool(np.mean(F) <= q + 1e-9),
                          "mean_yield": _nanmean(Y), "mean_precision": _nanmean(P),
                          "mean_power": _nanmean(W), "mean_n_selected": round(float(np.mean(NS)), 2)}
    return {"thresholds": {"tau_eff": SEL_TAU_EFF, "tau_off": SEL_TAU_OFF}, "n_mc": SEL_MC,
            "real_oracle": real, "rho_real_effective": round(rho_real, 3),
            "note": ("Rigorous cfBH (full-calibration p-values) controls FAR but is underpowered on the "
                     "weak surrogate (see scripts/09 sensitivity: power vs fidelity/n_cal/m + perfect "
                     "oracle).")}


def main():
    pp = _predictors()
    results = {"config": {"n_resplit": N_RESPLIT, "alphas": list(ALPHAS), "seed": SEED,
                          "heldout_chrom": HELDOUT_CHROM, "vram_gb": 0.0},
               "RQ1_marginal_coverage": rq1(pp),
               "RQ2_efficiency_width": rq2(pp),
               "RQ3_ood_transfer": rq3(),
               "guided_pareto": rq_guided_pareto(pp),
               "conformal_selection_cfBH": rq_conformal_selection()}
    (PROJECT / "results" / "json").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results" / "json" / "rq_benchmarks.json").write_text(json.dumps(results, indent=2))

    # ── console report ──
    print("═══ RQ1  Marginal coverage validity (500 resplits) ═══")
    for a in ALPHAS:
        print(f" alpha={a} (nominal {1-a}):")
        for f, r in results["RQ1_marginal_coverage"][f"alpha={a}"].items():
            print(f"   {f:<18} mean={r['mean_coverage']:.4f}  sd={r['sd']:.4f}  dev={r['signed_dev_from_nominal']:+.4f}")
    print("\n═══ RQ2  Efficiency & width + Mondrian off-target compression ═══")
    for a in ALPHAS:
        m = results["RQ2_efficiency_width"][f"alpha={a}"]["mondrian_offtarget"]
        print(f" alpha={a}: off-target mean U(count) pooled={m['pooled']['meanU_count']} "
              f"(cov {m['pooled']['coverage']}) -> Mondrian={m['mondrian']['meanU_count']} "
              f"(cov {m['mondrian']['coverage']})")
        for name, st in m["by_stratum"].items():
            print(f"     {name:<4} U: {st['meanU_pooled']:>7} -> {st['meanU_mondrian']:>7}  "
                  f"cov {st['cov_pooled']} -> {st['cov_mondrian']}")
    print("\n═══ RQ3  chr22 OOD transfer ═══")
    for a in ALPHAS:
        r = results["RQ3_ood_transfer"][f"alpha={a}"]
        print(f" alpha={a} (nominal {r['nominal']}):")
        print(f"   (a) plain on chr22   : cov={r['a_plain_on_chr22']['coverage']}  "
              f"meanU={r['a_plain_on_chr22']['meanU_count']}")
        b = r["b_mondrian_in_distribution"]
        print(f"   (b) Mondrian in-dist : worst-chrom cov {b['worst_chrom_cov_pooled']} -> {b['worst_chrom_cov_mondrian']}")
        print(f"   (c) weighted on chr22: cov={r['c_weighted_on_chr22']['coverage']}  "
              f"meanU={r['c_weighted_on_chr22']['meanU_count']}")
    cs = results["conformal_selection_cfBH"]
    print(f"\n═══ cfBH conformal selection (FAR control; rho_real={cs['rho_real_effective']}) ═══")
    for q in ("q=0.05", "q=0.1"):
        r = cs["real_oracle"][q]
        print(f"  real-oracle {q}: FAR={r['mean_FAR']} (<=q {r['FAR_le_q']}) power={r['mean_power']} "
              f"n_sel={r['mean_n_selected']}  (fidelity/n_cal/m sensitivity -> scripts/09)")
    print("\n[done] -> results/json/rq_benchmarks.json")


if __name__ == "__main__":
    main()
