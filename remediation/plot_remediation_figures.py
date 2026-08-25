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


if __name__ == "__main__":
    figR1(); figR2(); figR3()
    for f in ("figR1_validated_offtarget_cfbh", "figR2_uq_baselines", "figR3_transfer_shift"):
        p = FIG / f"{f}.png"
        print(f"wrote {p.relative_to(PROJECT)} ({p.stat().st_size//1024} KB)")
