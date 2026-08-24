#!/usr/bin/env python3
"""
ConformalGen — Phase 5 publication figures (300 DPI).

Reads results/json/rq_benchmarks.json (+ phase4_guided_generation.json) and writes:
  figures/fig1_framework_schematic.png   — Predictor -> Calibration -> Bounds -> Acceptance
  figures/fig2_rq1_coverage_validity.png — coverage vs theoretical band across scores/resplits
  figures/fig3_rq2_efficiency_width.png  — per-family widths/area + Mondrian off-target strata
  figures/fig4_rq3_ood_transfer.png      — chr22 OOD coverage/width gap + Mondrian/weighted recovery
  figures/fig5_guided_pareto.png         — conformal-guided yield vs design-threshold trade-off

CPU-only, headless (Agg backend), 0 GB VRAM.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
FIG = PROJECT / "figures"; FIG.mkdir(exist_ok=True)
RJSON = PROJECT / "results" / "json"
DPI = 300
C = {"point": "#c44e52", "conformal": "#4c72b0", "pooled": "#bdbdbd", "mondrian": "#4c72b0",
     "weighted": "#55a868", "nominal": "#333333"}


def _load(name):
    return json.loads((RJSON / name).read_text())


# ── Figure 1 : framework schematic ──
def fig1():
    fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.2)
    boxes = [(0.3, "Sequence\nGenerator\n(SFT / LLM /\ndiffusion)", "#eaeaf2"),
             (2.5, "Property\nPredictors\n(efficacy,\noff-target)", "#dbe7f3"),
             (4.7, "Split-Conformal\nCalibration\n(D_cal, alpha)", "#d5e8d4"),
             (6.9, "Directional &\nJoint inf-norm\nBounds\nL_eff, U_off", "#fff2cc"),
             (9.1, "Guarantee-Aware\nAcceptance\nL_eff>=tau_eff\nU_off<=tau_off", "#f8c9c9")]
    w, h, y = 1.7, 1.9, 0.7
    centers = []
    for x, label, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=col, ec="#555555", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)
        centers.append(x + w)
    for i in range(len(boxes) - 1):
        ax.add_patch(FancyArrowPatch((centers[i] + 0.02, y + h / 2), (boxes[i + 1][0] - 0.02, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#555555"))
    ax.text(5.55, 2.95, "ConformalGen: a universal, model-agnostic guarantee-aware wrapper",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.55, 0.35, "finite-sample, distribution-free (1-alpha) coverage over d=2 functional properties",
            ha="center", fontsize=9, style="italic", color="#555555")
    fig.tight_layout(); fig.savefig(FIG / "fig1_framework_schematic.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2 : RQ1 coverage validity ──
def fig2(rq):
    d = rq["RQ1_marginal_coverage"]; alphas = [0.10, 0.05]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, a in zip(axes, alphas):
        fam = d[f"alpha={a}"]; names = list(fam.keys())
        means = [fam[f]["mean_coverage"] for f in names]; sds = [fam[f]["sd"] for f in names]
        x = np.arange(len(names))
        lo = 1 - a; hi = 1 - a + 1 / (200 + 1)
        ax.axhspan(lo, hi, color="#4c72b0", alpha=0.18, label=f"theoretical band [{lo:.3f}, {hi:.3f}]")
        ax.axhline(lo, color=C["nominal"], ls="--", lw=1, label=f"nominal {lo:.2f}")
        ax.errorbar(x, means, yerr=sds, fmt="o", color=C["conformal"], capsize=3, ms=6)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("empirical coverage"); ax.set_title(f"RQ1  alpha={a}  (500 resplits)")
        ax.set_ylim(lo - 0.06, lo + 0.09); ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("RQ1 — Marginal coverage validity across non-conformity scores", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig2_rq1_coverage_validity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3 : RQ2 efficiency & width ──
def fig3(rq):
    d = rq["RQ2_efficiency_width"]["alpha=0.1"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # left: per-family widths (efficacy width, off-target width) — absolute vs cqr
    fams = ["absolute", "cqr"]
    eff_w = [d[f]["eff_width"] for f in fams]; off_w = [d[f]["off_width_count"] for f in fams]
    x = np.arange(len(fams)); bw = 0.35
    axL.bar(x - bw/2, eff_w, bw, label="efficacy width", color="#4c72b0")
    axL.bar(x + bw/2, off_w, bw, label="off-target width (count)", color="#dd8452")
    for i, f in enumerate(fams):
        axL.text(i, max(eff_w[i], off_w[i]) + 1, f"area={d[f]['hyperrect_area']:.0f}", ha="center", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels(fams); axL.set_ylabel("interval width")
    axL.set_title("RQ2  two-sided width & hyper-rectangle area (alpha=0.10)")
    axL.legend(fontsize=8); axL.grid(axis="y", alpha=0.25)
    # right: Mondrian off-target strata — pooled vs Mondrian width, coverage annotated
    m = rq["RQ2_efficiency_width"]["alpha=0.1"]["mondrian_offtarget"]["by_stratum"]
    strata = list(m.keys())
    up = [m[s]["meanU_pooled"] for s in strata]; um = [m[s]["meanU_mondrian"] for s in strata]
    x = np.arange(len(strata))
    axR.bar(x - bw/2, up, bw, label="pooled", color=C["pooled"])
    axR.bar(x + bw/2, um, bw, label="Mondrian", color=C["mondrian"])
    for i, s in enumerate(strata):
        axR.text(i - bw/2, up[i] + 1, f"{m[s]['cov_pooled']:.2f}", ha="center", fontsize=7.5, color="#a00")
        axR.text(i + bw/2, um[i] + 1, f"{m[s]['cov_mondrian']:.2f}", ha="center", fontsize=7.5, color="#060")
    axR.set_xticks(x); axR.set_xticklabels([f"{s}\noff-target" for s in strata])
    axR.set_ylabel("mean off-target upper bound U (count)")
    axR.set_title("RQ2  Mondrian strata: width + coverage (labels)")
    axR.legend(fontsize=8); axR.grid(axis="y", alpha=0.25)
    fig.suptitle("RQ2 — Efficiency, set width & Mondrian conditional validity", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig3_rq2_efficiency_width.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 4 : RQ3 OOD transfer ──
def fig4(rq):
    d = rq["RQ3_ood_transfer"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # left: chr22 coverage + width, plain vs weighted, both alphas
    alphas = [0.10, 0.05]; x = np.arange(len(alphas)); bw = 0.35
    plain = [d[f"alpha={a}"]["a_plain_on_chr22"]["coverage"] for a in alphas]
    wtd = [d[f"alpha={a}"]["c_weighted_on_chr22"]["coverage"] for a in alphas]
    plainU = [d[f"alpha={a}"]["a_plain_on_chr22"]["meanU_count"] for a in alphas]
    wtdU = [d[f"alpha={a}"]["c_weighted_on_chr22"]["meanU_count"] for a in alphas]
    axL.bar(x - bw/2, plain, bw, label="plain split-conformal", color=C["pooled"])
    axL.bar(x + bw/2, wtd, bw, label="weighted (shift)", color=C["weighted"])
    for a_i, a in enumerate(alphas):
        axL.axhline(1 - a, color=C["nominal"], ls="--", lw=1)
        axL.text(a_i - bw/2, plain[a_i] + 0.005, f"U={plainU[a_i]:.2f}", ha="center", fontsize=8)
        axL.text(a_i + bw/2, wtd[a_i] + 0.005, f"U={wtdU[a_i]:.2f}", ha="center", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels([f"alpha={a}\n(nom {1-a:.2f})" for a in alphas])
    axL.set_ylabel("coverage on chr22"); axL.set_ylim(0.85, 1.0)
    axL.set_title("RQ3  chr22 OOD: plain over-covers & inflates width;\nweighted recovers efficient coverage")
    axL.legend(fontsize=8, loc="lower right"); axL.grid(axis="y", alpha=0.25)
    # right: per-chromosome in-distribution coverage, pooled vs Mondrian (alpha=0.10)
    pc = d["alpha=0.1"]["b_mondrian_in_distribution"]["per_chromosome"]
    chroms = list(pc.keys())
    cp = [pc[c]["cov_pooled"] for c in chroms]; cm = [pc[c]["cov_mondrian"] for c in chroms]
    xc = np.arange(len(chroms))
    axR.plot(xc, cp, "o-", ms=3, color=C["pooled"], label="pooled")
    axR.plot(xc, cm, "s-", ms=3, color=C["mondrian"], label="Mondrian")
    axR.axhline(0.9, color=C["nominal"], ls="--", lw=1, label="nominal 0.90")
    axR.set_xticks(xc); axR.set_xticklabels([c.replace("chr", "") for c in chroms], fontsize=7)
    axR.set_xlabel("in-distribution chromosome"); axR.set_ylabel("conditional coverage")
    axR.set_title("RQ3  Group-conditional coverage across chromosomes (alpha=0.10)")
    axR.legend(fontsize=8); axR.grid(axis="y", alpha=0.25)
    fig.suptitle("RQ3 — Conditional & OOD transfer under genomic shift", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig4_rq3_ood_transfer.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 5 : guided yield vs threshold Pareto ──
def fig5(rq):
    rows = rq["guided_pareto"]["alpha=0.1"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # left: yield vs bound-coverage/precision, point vs conformal
    for mode, key, col in [("point", "design_precision", C["point"]),
                           ("conformal_directional", "bound_coverage", C["conformal"])]:
        pts = [(r["yield"], r[key]) for r in rows if r["mode"] == mode and r[key] is not None]
        if pts:
            ys, cs = zip(*pts)
            axL.scatter(ys, cs, s=28, color=col, alpha=0.75,
                        label=("point (precision)" if mode == "point" else "conformal (bound-cov)"))
    axL.axhline(0.9, color=C["nominal"], ls="--", lw=1, label="nominal 0.90")
    axL.set_xlabel("acceptance yield"); axL.set_ylabel("quality (precision / bound-coverage)")
    axL.set_title("Guided acceptance: yield vs guaranteed quality (alpha=0.10)")
    axL.legend(fontsize=8); axL.grid(alpha=0.25)
    # right: yield surface vs tau_off for a few tau_eff (conformal)
    for te_ in [40, 50, 60]:
        pts = sorted([(r["tau_off"], r["yield"]) for r in rows
                      if r["mode"] == "conformal_directional" and r["tau_eff"] == te_])
        to, yl = zip(*pts)
        axR.plot(to, yl, "o-", label=f"tau_eff={te_}")
    axR.set_xlabel("tau_off (max off-target)"); axR.set_ylabel("conformal acceptance yield")
    axR.set_title("Yield vs design thresholds (conformal_directional)")
    axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle("Figure 5 — Conformal-guided yield / quality trade-off", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig5_guided_pareto.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    rq = _load("rq_benchmarks.json")
    fig1(); fig2(rq); fig3(rq); fig4(rq); fig5(rq)
    for p in sorted(FIG.glob("fig*.png")):
        print(f"  wrote {p.relative_to(PROJECT)}  ({p.stat().st_size // 1024} KB)")
    print(f"[done] {len(list(FIG.glob('fig*.png')))} figures @ 300 DPI -> figures/")


if __name__ == "__main__":
    main()
