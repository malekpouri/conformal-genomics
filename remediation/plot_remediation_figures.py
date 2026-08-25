#!/usr/bin/env python3
"""Render the remediation evidence figures (300 DPI) into figures/ for the refactored manuscript."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIG = PROJECT / "figures"; FIG.mkdir(exist_ok=True)
R = HERE / "results"
DPI = 300
C = {"a": "#4c72b0", "b": "#55a868", "c": "#c44e52", "n": "#333333", "g": "#bdbdbd"}


def L(name):
    return json.loads((R / name).read_text())


# ── figR1: validated off-target oracle + real-data cfBH ──
def figR1():
    p2 = L("phase2.json")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    aucs = [0.696, p2["richer_oracle"]["phase1_mmonly_roc_auc"], p2["richer_oracle"]["roc_auc"]]
    labs = ["naive\nMM count", "mismatch\npattern", "rich\n(+one-hot)"]
    bars = axL.bar(labs, aucs, color=[C["g"], C["a"], C["b"]])
    for b, v in zip(bars, aucs):
        axL.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    axL.axhline(0.5, color=C["n"], ls=":", lw=1, label="chance")
    axL.set_ylim(0.5, 1.0); axL.set_ylabel("ROC-AUC (guide-disjoint CV)")
    axL.set_title("Validated off-target oracle vs CIRCLE-seq cleavage")
    axL.legend(fontsize=8); axL.grid(axis="y", alpha=0.25)

    lv = p2["real_data_cfbh_offtarget_only"]["levels"]
    qs = ["q=0.1", "q=0.2"]; x = np.arange(2); w = 0.26
    power = [lv[q]["mean_power"] for q in qs]; prec = [lv[q]["mean_precision"] for q in qs]
    far = [lv[q]["mean_FAR"] for q in qs]; qv = [0.10, 0.20]
    axR.bar(x - w, power, w, label="power (recall of safe)", color=C["a"])
    axR.bar(x, prec, w, label="precision", color=C["b"])
    axR.bar(x + w, far, w, label="FAR", color=C["c"])
    axR.plot(x + w, qv, "k_", ms=18, mew=2, label="q (FAR target)")
    for i in range(2):
        axR.text(x[i]-w, power[i]+0.02, f"{power[i]:.2f}", ha="center", fontsize=9)
        axR.text(x[i]+w, far[i]+0.02, f"{far[i]:.3f}", ha="center", fontsize=8, color=C["c"])
    axR.set_xticks(x); axR.set_xticklabels(["q=0.10", "q=0.20"]); axR.set_ylim(0, 1.08)
    axR.set_title("Real-data cfBH on VALIDATED CIRCLE-seq truth")
    axR.legend(fontsize=8, loc="center right"); axR.grid(axis="y", alpha=0.25)
    fig.suptitle("Validated off-target safety selection (Destination A)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figR1_validated_offtarget_cfbh.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── figR2: UQ baselines vs conformal ──
def figR2():
    c = L("phase4.json")["C_uq_baselines"]
    methods = ["Deep ensemble\n(K=5)", "MC-dropout\n(T=20)", "Conformalized\nensemble"]
    cov = [c["deep_ensemble_K5"]["coverage"], c["mc_dropout_T20"]["coverage"], c["conformalized_ensemble"]["coverage"]]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(methods, cov, color=[C["c"], C["c"], C["b"]])
    for b, v in zip(bars, cov):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=11)
    ax.axhline(0.90, color=C["n"], ls="--", lw=1.2, label="nominal 0.90")
    ax.set_ylim(0, 1.0); ax.set_ylabel("empirical coverage (WT held-out)")
    ax.set_title("Real ML uncertainty baselines miscalibrate;\nconformalizing the same ensemble is exact")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "figR2_uq_baselines.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── figR3: cross-nuclease/cell-line transfer failure ──
def figR3():
    b = L("phase4.json")["B_nuclease_transfer_OOD"]["by_nuclease"]
    names = list(b.keys()); x = np.arange(len(names)); w = 0.38
    plain = [b[k]["coverage_plain"] for k in names]; wtd = [b[k]["coverage_weighted"] for k in names]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w/2, plain, w, label="plain conformal", color=C["g"])
    ax.bar(x + w/2, wtd, w, label="weighted conformal", color=C["a"])
    ax.axhline(0.90, color=C["n"], ls="--", lw=1.2, label="nominal 0.90")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_ylabel("coverage (WT-calibrated)")
    ax.set_title("Coverage does NOT transfer across nuclease / cell-line\n(concept shift; weighted conformal cannot fix it)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "figR3_transfer_shift.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── figR4: deployment regime — cfBH vs naive; enumerated vs sequence-only ──
def figR4():
    d = json.loads((PROJECT / "results" / "json" / "deployment_benchmarks.json").read_text())
    sw = d["stringency_sweep"]
    strat = ["p10_strict", "p25", "p50_median"]; x = np.arange(len(strat)); w = 0.35
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    powE = [sw["assay_enumerated"][s]["q=0.1"]["cfbh_power"] for s in strat]
    powS = [sw["sequence_only"][s]["q=0.1"]["cfbh_power"] for s in strat]
    axL.bar(x - w/2, powE, w, label="alignment-enumerated oracle (ρ=0.94)", color=C["a"])
    axL.bar(x + w/2, powS, w, label="sequence-only oracle (ρ≈0)", color=C["c"])
    axL.set_xticks(x); axL.set_xticklabels(["p10\n(strict)", "p25", "p50\n(median)"])
    axL.set_ylabel("cfBH power (recall of safe)"); axL.set_ylim(0, 1)
    axL.set_title("cfBH power collapses without alignment (q=0.10)")
    axL.legend(fontsize=8); axL.grid(axis="y", alpha=0.25)
    # right: FAR — cfBH (certified) vs naive fixed-cutoff (uncontrolled)
    regimes = [("assay_enumerated", "p25", "enum p25"), ("assay_enumerated", "p50_median", "enum p50"),
               ("sequence_only", "p50_median", "seq-only p50")]
    cf = [sw[o][s]["q=0.1"]["cfbh_FAR"] for o, s, _ in regimes]
    nv = [sw[o][s]["q=0.1"]["naive_fixedcut_FAR"] for o, s, _ in regimes]
    xr = np.arange(len(regimes))
    axR.bar(xr - w/2, cf, w, label="cfBH (certified)", color=C["b"])
    axR.bar(xr + w/2, nv, w, label="naive fixed-cutoff", color=C["g"])
    axR.axhline(0.10, color=C["n"], ls="--", lw=1.2, label="q = 0.10")
    for i, v in enumerate(nv):
        axR.text(xr[i]+w/2, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
    axR.set_xticks(xr); axR.set_xticklabels([r[2] for r in regimes]); axR.set_ylim(0, 0.6)
    axR.set_ylabel("empirical FAR"); axR.set_title("cfBH certifies FAR≤q; the heuristic is uncontrolled")
    axR.legend(fontsize=8); axR.grid(axis="y", alpha=0.25)
    fig.suptitle("Deployment regime: cfBH fails safe; off-target burden needs alignment", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figR4_deployment.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── figR5: referee closure — calibrated-threshold FAR violations (#4) + aligned regime (#7) ──
def figR5():
    d = json.loads((PROJECT / "results" / "json" / "final_referee_closure.json").read_text())
    bl = d["calibrated_threshold_vs_cfbh"]; ar = d["aligned_regime_hash7"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    qs = ["q=0.1", "q=0.2"]; x = np.arange(2); w = 0.35
    cf = [bl[q]["cfbh"]["P_FAR_gt_q"] for q in qs]; ct = [bl[q]["calibrated_threshold"]["P_FAR_gt_q"] for q in qs]
    axL.bar(x - w/2, cf, w, label="cfBH (certified)", color=C["b"])
    axL.bar(x + w/2, ct, w, label="calibrated threshold", color=C["c"])
    for i in range(2):
        axL.text(x[i]+w/2, ct[i]+0.01, f"{ct[i]:.0%}", ha="center", fontsize=10)
        axL.text(x[i]-w/2, cf[i]+0.01, f"{cf[i]:.0%}", ha="center", fontsize=10, color=C["b"])
    axL.set_xticks(x); axL.set_xticklabels(["q=0.10", "q=0.20"]); axL.set_ylim(0, 0.55)
    axL.set_ylabel("P(test FAR > q)  — safety-target violation rate")
    axL.set_title("#4  Calibrated tuning violates FAR; cfBH certifies it")
    axL.legend(fontsize=9); axL.grid(axis="y", alpha=0.25)
    r = ar["burden_fidelity_rho_vs_validated"]
    labs = ["all\ncandidates", "mm≤3\n(aligned)", "naive mm≤3\ncount"]
    vals = [r["all_candidates"], r["mm<=3_aligned"], r["naive_mm<=3_count"]]
    bars = axR.bar(labs, vals, color=[C["a"], C["b"], C["g"]])
    for b, v in zip(bars, vals):
        axR.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
    axR.axhline(0.85, color=C["n"], ls="--", lw=1.2, label="ρ = 0.85 threshold")
    axR.set_ylim(0.5, 1.0); axR.set_ylabel("burden fidelity ρ vs validated truth")
    axR.set_title("#7  Oracle survives the genome-wide-align (mm≤3) regime")
    axR.legend(fontsize=9); axR.grid(axis="y", alpha=0.25)
    fig.suptitle("Referee closure: distribution-free certificate (#4) and aligned-regime robustness (#7)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figR5_referee_closure.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── figR6: external cross-assay replication — the oracle does not transfer ──
def figR6():
    d = json.loads((PROJECT / "results" / "json" / "external_replication.json").read_text())
    rows = [("CIRCLE-seq\n(in-assay)", 0.925, 0.944, C["b"]),
            ("SITE-Seq\n(cross-assay)", d["cross_assay_oracle_SITEseq"]["pair_roc_auc"],
             d["cross_assay_oracle_SITEseq"]["per_guide_burden_rho"], C["c"]),
            ("GUIDE-seq\n(cross-assay)", d["cross_assay_oracle_GUIDEseq"]["pair_roc_auc"],
             d["cross_assay_oracle_GUIDEseq"]["per_guide_burden_rho"], C["c"])]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    labs = [r[0] for r in rows]; cols = [r[3] for r in rows]
    axL.bar(labs, [r[1] for r in rows], color=cols)
    for i, r in enumerate(rows):
        axL.text(i, r[1] + 0.01, f"{r[1]:.3f}", ha="center", fontsize=10)
    axL.axhline(0.5, color=C["n"], ls=":", lw=1, label="chance"); axL.set_ylim(0.4, 1.0)
    axL.set_ylabel("pair ROC-AUC"); axL.set_title("Off-target oracle: strong in-assay, weak cross-assay")
    axL.legend(fontsize=8); axL.grid(axis="y", alpha=0.25)
    axR.bar(labs, [r[2] for r in rows], color=cols)
    for i, r in enumerate(rows):
        axR.text(i, r[2] + 0.01, f"{r[2]:.3f}", ha="center", fontsize=10)
    axR.axhline(0.0, color=C["n"], lw=0.8); axR.set_ylim(-0.05, 1.0)
    axR.set_ylabel("per-guide burden fidelity ρ"); axR.set_title("Fidelity collapses across assays")
    axR.grid(axis="y", alpha=0.25)
    fig.suptitle("External replication: the oracle is assay-specific (cfBH still fails safe, FAR ≤ q)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figR6_external_replication.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figR1(); figR2(); figR3(); figR4(); figR5(); figR6()
    for f in ("figR1_validated_offtarget_cfbh", "figR2_uq_baselines", "figR3_transfer_shift",
              "figR4_deployment", "figR5_referee_closure", "figR6_external_replication"):
        p = FIG / f"{f}.png"
        print(f"wrote {p.relative_to(PROJECT)} ({p.stat().st_size//1024} KB)")
