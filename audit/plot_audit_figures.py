#!/usr/bin/env python3
"""
Referee audit figures (roadmap M5).  Reads only audit/results/*.json and writes 300-DPI panels to
figures/.  Every panel states n, the nominal level where one applies, and an explicit note when a
quantity is undefined (e.g. precision over an empty selection) instead of dropping it silently.

  figA1  leakage effect: the same learner and features, wrong vs right grouping
  figA2  off-target baselines: pair-level discrimination and per-guide burden fidelity
  figA3  the feasibility bound and the cohort size a certified selection needs
  figA4  uncertainty baselines, correctly implemented
  figA5  cross-context transfer and small-sample recalibration

Output: figures/figA1_leakage.png ... figA5_transfer_recalibration.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE.parent / "figures"
DPI = 300

LEAK = "#c0392b"
CLEAN = "#1f6f8b"
FREE = "#2e7d32"
GREY = "#7f8c8d"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "figure.autolayout": True})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_a1():
    j = load("a1_regrouped.json")
    if not j:
        return
    lf = j["B_oracle_leakfree_vs_published"]["leak_free"]
    lk = j["B_oracle_leakfree_vs_published"]["published_leaky"]
    fid = j["C_burden_fidelity_per_real_guide"]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))

    # (a) pair-level AUC / AP
    labels = ["ROC-AUC", "Average\nprecision"]
    x = np.arange(2); w = 0.36
    ax[0].bar(x - w/2, [lk["pooled_roc_auc"], lk["pooled_average_precision"]], w,
              color=LEAK, label="published grouping\n(80 pseudo-guides)")
    ax[0].bar(x + w/2, [lf["pooled_roc_auc"], lf["pooled_average_precision"]], w,
              color=CLEAN, label="leak-free\n(10 real guides)")
    ax[0].axhline(0.696, color=GREY, ls=":", lw=1)
    ax[0].text(-0.45, 0.655, "naive mismatch count (AUC 0.696)", color=GREY, fontsize=6.5,
               ha="left", va="top")
    for xi, v in zip(x - w/2, [lk["pooled_roc_auc"], lk["pooled_average_precision"]]):
        ax[0].text(xi, v + .02, f"{v:.3f}", ha="center", fontsize=7.5, color=LEAK)
    for xi, v in zip(x + w/2, [lf["pooled_roc_auc"], lf["pooled_average_precision"]]):
        ax[0].text(xi, v + .02, f"{v:.3f}", ha="center", fontsize=7.5, color=CLEAN)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(a) Pair-level, same learner\n(prevalence 1.26%)")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.62, 1.0))

    # (b) per-guide burden fidelity with guide-bootstrap CIs
    keys = ["ml_oracle_published_leaky", "ml_oracle_leakfree",
            "baseline_candidate_site_count", "baseline_mm3_site_count"]
    names = ["ML oracle\nleaky", "ML oracle\nleak-free", "site count\n(free)",
             "mm$\\leq$3 count\n(free)"]
    cols = [LEAK, CLEAN, FREE, FREE]
    for i, (k, c) in enumerate(zip(keys, cols)):
        r = fid[k]["rho_spearman"]; lo, hi = fid[k]["rho_ci95_guide_bootstrap"]
        ax[1].errorbar(i, r, yerr=[[r - lo], [hi - r]], fmt="o", color=c, capsize=3, ms=5)
        ax[1].text(i + .12, r, f"{r:.3f}", fontsize=7.5, color=c, va="center")
    ax[1].axhline(0, color=GREY, lw=.8)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names, fontsize=6.8)
    ax[1].set_xlim(-0.6, 3.8)
    ax[1].set_ylabel("Spearman $\\rho$ vs validated burden")
    ax[1].set_ylim(-1.05, 1.15)
    ax[1].set_title("(b) Per-guide burden fidelity\n(n = 10 guides, 95% guide bootstrap)")

    # (c) what the grouping key actually is
    d = j["A_grouping_defect"]
    ax[2].axis("off")
    ax[2].set_title("(c) The grouping key")
    txt = (f"rows: 584,949\n\n"
           f"`sgRNA_seq` minus '_':  {d['n_pipeline_groups']} groups\n"
           f"`sgRNA_type` (canonical):  {d['n_real_guides']} guides\n\n"
           f"= 10 protospacers x 8 representations\n"
           f"   (4 PAM variants x 2 alignment\n"
           f"    hypotheses: mismatch-only / bulge)\n\n"
           f"features use the first 20 nt only, so\n"
           f"guide-side features are identical\n"
           f"across a 'guide-disjoint' split")
    ax[2].text(0.0, 0.92, txt, va="top", fontsize=7.6, family="monospace")
    fig.savefig(FIG / "figA1_leakage.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("figA1_leakage.png")


def fig_a2():
    j = load("a5_domain_baselines.json")
    if not j:
        return
    order = ["mismatch_count", "mit_hsu_2013", "position_substitution_logreg",
             "mismatch_pattern_oracle_21feat", "manuscript_oracle_181feat"]
    pretty = {"mismatch_count": "mismatch count", "mit_hsu_2013": "MIT / Hsu-Zhang",
              "position_substitution_logreg": "pos.-subst. logistic",
              "mismatch_pattern_oracle_21feat": "21-feat mismatch",
              "manuscript_oracle_181feat": "181-feat oracle"}
    fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.2))
    x = np.arange(len(order))
    auc = [j["baselines"][k]["pooled_roc_auc"] for k in order]
    ap = [j["baselines"][k]["pooled_average_precision"] for k in order]
    wg = [j["baselines"][k]["within_guide_roc_auc_mean"] for k in order]
    wgap = [j["baselines"][k]["within_guide_average_precision_mean"] for k in order]
    cols = [GREY, FREE, GREY, FREE, CLEAN]
    ax[0].bar(x - .2, auc, .38, color=cols, label="pooled")
    ax[0].bar(x + .2, wg, .38, color=cols, alpha=.45, label="within-guide mean")
    ax[0].set_xticks(x); ax[0].set_xticklabels([pretty[k] for k in order], fontsize=6.4, rotation=32, ha="right")
    ax[0].set_ylim(0.5, 0.95); ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title("(a) Discrimination, leak-free")
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[1].bar(x - .2, ap, .38, color=cols)
    ax[1].bar(x + .2, wgap, .38, color=cols, alpha=.45)
    ax[1].axhline(j["cohort"]["prevalence"], color=LEAK, ls=":", lw=1)
    ax[1].text(len(order) - .5, j["cohort"]["prevalence"] + .0035, "prevalence 1.26%",
               fontsize=6.5, color=LEAK, ha="right", va="bottom")
    ax[1].set_xticks(x); ax[1].set_xticklabels([pretty[k] for k in order], fontsize=6.4, rotation=32, ha="right")
    ax[1].set_ylabel("Average precision")
    ax[1].set_title("(b) Average precision\n(the metric that matters at 1.26%)")
    bf = j["burden_fidelity"]
    bk = [k for k in ["baseline_candidate_site_count", "baseline_mm3_site_count", "mit_hsu_2013",
                      "mismatch_pattern_oracle_21feat", "position_substitution_logreg",
                      "manuscript_oracle_181feat"] if k in bf]
    names = {"baseline_candidate_site_count": "candidate site count (free)",
             "baseline_mm3_site_count": "mm$\\leq$3 site count (free)",
             "mit_hsu_2013": "MIT / Hsu-Zhang",
             "mismatch_pattern_oracle_21feat": "21-feature mismatch pattern",
             "position_substitution_logreg": "position-substitution logistic",
             "manuscript_oracle_181feat": "181-feature oracle"}
    for i, k in enumerate(bk):
        r = bf[k]["rho_spearman"]; lo, hi = bf[k]["rho_ci95_guide_bootstrap"]
        c = FREE if k.startswith("baseline") else (CLEAN if "oracle" in k else GREY)
        ax[2].errorbar(r, i, xerr=[[r - lo], [hi - r]], fmt="o", color=c, capsize=3, ms=5)
    ax[2].axvline(0, color=GREY, lw=.8)
    ax[2].set_yticks(range(len(bk))); ax[2].set_yticklabels([names[k] for k in bk], fontsize=6.8)
    ax[2].set_xlabel("Spearman $\\rho$ vs validated burden")
    ax[2].set_xlim(-1.05, 1.1)
    ax[2].set_title("(c) Per-guide burden fidelity\n(n = 10, 95% guide bootstrap)")
    fig.savefig(FIG / "figA2_baselines.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("figA2_baselines.png")


def fig_a3():
    j6 = load("a6_matched_yield_cohort.json")
    if not j6:
        return
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.9))

    # (a) the feasibility bound: n_cal needed to certify one candidate out of m
    ms = np.array([1, 3, 10, 30, 100, 300, 1000, 2000])
    for q, c in [(0.10, CLEAN), (0.20, FREE)]:
        ax[0].plot(ms, ms / q - 1, "o-", color=c, ms=4, label=f"q = {q:.2f}")
    ax[0].axhline(10, color=LEAK, ls="--", lw=1)
    ax[0].text(1.2, 12, "validated cohort available (10 guides)", color=LEAK, fontsize=6.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("candidate pool size $m$")
    ax[0].set_ylabel("calibration guides required")
    ax[0].set_title("(a) $n_{cal} \\geq m/q - 1$\nto certify one candidate")
    ax[0].legend(fontsize=7, frameon=False)

    # (b) cfBH power vs cohort size, by oracle fidelity (design calculation)
    ns = [10, 20, 40, 60, 80, 120, 200, 400]
    for rho, c in [(0.90, GREY), (0.95, CLEAN), (0.99, FREE)]:
        p = [j6["sweep"][f"q=0.1|rho={rho}|n={n}"]["cfbh"]["mean_power"] for n in ns]
        ax[1].plot(ns, p, "o-", ms=4, color=c, label=f"$\\rho$ = {rho}")
    meas = j6["measured"].get("leakfree_oracle_rho")
    if meas is not None and f"q=0.1|rho={meas}|n=10" in j6["sweep"]:
        p = [j6["sweep"][f"q=0.1|rho={meas}|n={n}"]["cfbh"]["mean_power"] for n in ns]
        ax[1].plot(ns, p, "s--", ms=4, color=LEAK,
                   label=f"measured oracle\n($\\rho$ = {meas})")
    ax[1].axvspan(0, 10, color=LEAK, alpha=.08)
    ax[1].set_xlabel("validated guides in cohort"); ax[1].set_ylabel("cfBH power (q = 0.10)")
    ax[1].set_ylim(-0.03, 1.0)
    ax[1].set_title("(b) Power needs fidelity AND cohort\n(design calculation; shaded = data on hand)")
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper left")

    # (c) matched-yield comparison against the tuned threshold
    rho = 0.99
    own = [j6["sweep"][f"q=0.1|rho={rho}|n={n}"]["tuned_threshold_own_operating_point"]["P_FAR_gt_q"]
           for n in ns]
    mat = [j6["sweep"][f"q=0.1|rho={rho}|n={n}"]["tuned_threshold_at_cfbh_yield"]["P_FAR_gt_q"]
           for n in ns]
    cf = [j6["sweep"][f"q=0.1|rho={rho}|n={n}"]["cfbh"]["P_FAR_gt_q"] for n in ns]
    ax[2].plot(ns, own, "o-", color=LEAK, ms=4, label="tuned threshold,\nits own yield")
    ax[2].plot(ns, mat, "s-", color=FREE, ms=4, label="tuned threshold,\nat cfBH's yield")
    ax[2].plot(ns, cf, "^-", color=CLEAN, ms=4, label="cfBH")
    ax[2].set_xlabel("validated guides in cohort")
    ax[2].set_ylabel("$\\Pr(\\mathrm{FAR} > q)$")
    ax[2].set_ylim(-0.03, 0.6)
    ax[2].set_title(f"(c) The violation gap is a yield gap\n($\\rho$ = {rho}, q = 0.10)")
    ax[2].legend(fontsize=6.5, frameon=False)
    fig.savefig(FIG / "figA3_feasibility.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("figA3_feasibility.png")


def fig_a4():
    j = load("a2_uq_baselines.json")
    if not j:
        return
    arms = [("deep_ensemble_published_verbatim_recipe", "deep ensemble\n(published recipe)", LEAK),
            ("mc_dropout_published_verbatim_recipe", "MC-dropout\n(published recipe)", LEAK),
            ("deep_ensemble_corrected_gaussian_nll_mixture", "deep ensemble\n(NLL mixture)", FREE),
            ("mc_dropout_corrected_with_observation_noise", "MC-dropout\n(+ obs. noise)", FREE),
            ("conformalized_mse_ensemble_mean", "split conformal\n(ensemble mean)", CLEAN),
            ("conformalized_nll_ensemble_normalised", "split conformal\n(normalised score)", CLEAN)]
    arms = [a for a in arms if a[0] in j]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    y = np.arange(len(arms))
    cov = [j[k]["coverage"] for k, _, _ in arms]
    wid = [j[k]["mean_width"] for k, _, _ in arms]
    cols = [c for _, _, c in arms]
    ax[0].barh(y, cov, color=cols)
    ax[0].axvline(j["nominal"], color="k", ls="--", lw=1)
    ax[0].text(j["nominal"] + .01, len(arms) - .4, f"nominal {j['nominal']:.2f}", fontsize=7)
    for yi, v in zip(y, cov):
        ax[0].text(v + .01, yi, f"{v:.3f}", va="center", fontsize=7.5)
    ax[0].set_yticks(y); ax[0].set_yticklabels([n for _, n, _ in arms], fontsize=7)
    ax[0].set_xlim(0, 1.12); ax[0].set_xlabel("empirical coverage")
    ax[0].set_title(f"(a) Coverage, n = {j['dataset']['n_test']:,} held-out guides")
    ax[1].barh(y, wid, color=cols)
    for yi, v in zip(y, wid):
        ax[1].text(v + .6, yi, f"{v:.1f}", va="center", fontsize=7.5)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("mean interval width (indel %)")
    ax[1].set_title(f"(b) Width  (label sd = {j['label_sd']})")
    fig.savefig(FIG / "figA4_uq_baselines.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("figA4_uq_baselines.png")


def fig_a5():
    j = load("a3_transfer_shift.json")
    if not j:
        return
    panel = j["panel"]
    names = list(panel)
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    x = np.arange(len(names))
    before = [panel[n]["coverage_WT_calibrated"] for n in names]
    after = [panel[n]["recalibrated_on_target_labels"].get("n_target_labels=50", {})
             .get("mean_coverage", np.nan) for n in names]
    ax[0].bar(x - .2, before, .38, color=LEAK, label="WT-calibrated")
    ax[0].bar(x + .2, after, .38, color=FREE, label="recalibrated on 50\ntarget-domain labels")
    ax[0].axhline(j["nominal"], color="k", ls="--", lw=1)
    ax[0].text(len(names) - .6, j["nominal"] + .02, f"nominal {j['nominal']:.2f}", fontsize=7,
               ha="right")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax[0].set_ylabel("empirical coverage"); ax[0].set_ylim(0, 1.08)
    ax[0].set_title("(a) Transfer failure and its remedy")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")

    ov = [panel[n]["overlap_with_WT_sgRNAs"] for n in names]
    da = [panel[n]["domain_classifier_auc_sequence_disjoint"] for n in names]
    ax[1].scatter(ov, da, c=[LEAK if o > .5 else FREE for o in ov], s=36)
    for n, o, a in zip(names, ov, da):
        ax[1].annotate(n, (o, a), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax[1].axhline(0.5, color=GREY, ls=":", lw=1)
    ax[1].text(.5, .52, "chance -- no covariate shift to reweight", fontsize=6.5, color=GREY)
    ax[1].set_xlabel("sgRNA-set overlap with WT calibration data")
    ax[1].set_ylabel("domain-classifier AUC\n(sequence-disjoint)")
    ax[1].set_ylim(0.4, 1.02)
    ax[1].set_title("(b) ESP/HF carry no covariate shift")
    fig.savefig(FIG / "figA5_transfer_recalibration.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("figA5_transfer_recalibration.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_a1(); fig_a2(); fig_a3(); fig_a4(); fig_a5()
    print("[plots] -> figures/figA*.png")
