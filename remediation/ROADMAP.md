# ConformalGen — Remediation Roadmap (referee-driven)

**Owner:** Lead Researcher / Autonomous AI Engineer.
**Goal:** resolve the actionable flaws from the harsh review by rebuilding the empirical core, not by
editing prose. The statistical layer (conformal + cfBH + sensitivity) is already correct and stays; the
work is in the *oracle*, the *generator*, and the *experimental design* underneath it.

**Decision the whole project hinges on (set in Phase 1):**
> Destination **A** — a working applied-methods paper (real oracle strong enough that cfBH has real-data
> power), vs Destination **B** — an honest, rigorous *negative/benchmark* result (UQ is oracle-limited).
> Phase 1 is the go/no-go diagnostic.

---

## Phase 1 — Diagnostic: CIRCLE-seq retargeting + baseline power assessment  ← EXECUTE NOW
Invert the off-target target from the hand-built MM≤3 *count* to **experimentally validated CIRCLE-seq
cleavage**. Train a proper (guide-disjoint) off-target classifier on validated labels, measure its
fidelity, and read the implied cfBH power off the existing sensitivity curve.
- **Deliverables:** pair-level ROC-AUC / average-precision vs the naive MM≤3 score (paper's 0.696);
  per-guide *validated burden* fidelity ρ = Spearman(predicted, true), compared to the old surrogate;
  expected cfBH power at that ρ.
- **Go/no-go:** does ρ rise far enough (curve: ρ=0.9→power 0.11, 0.95→0.33, 0.99→0.71) to move real-data
  power off zero? → chooses A or B.
- *Addresses flaws #2 (vacuous guarantee), and diagnoses #1 (inert cfBH) and #9 (wide intervals).*

## Phase 2 — Build the validated-grounded off-target oracle at scale
If go: train the production off-target oracle on CIRCLE-seq (+ the II_* validated benchmark datasets),
indel/bulge-aware where data permit; expose it as the ConformalGen off-target oracle. Rebuild the guide
pool so each guide carries **matched efficacy + validated off-target burden**.
- *Addresses #2, #9; feeds #1.*

## Phase 3 — Real sequence generator
Train/wire a real autoregressive (or diffusion) sgRNA generator behind the existing
`GenericSequenceGenerator` interface; produce genuine candidate pools (replacing resampling emulation).
- *Addresses #4 (no real generator).*

## Phase 4 — Honest experimental design
Gene/locus-**grouped** splits as the *primary* evaluation (not a footnote); domain baselines
(DeepHF / DeepCRISPR, deep-ensemble & MC-dropout UQ); a *real* distribution shift via cell-line /
nuclease transfer using the CRISPR_HNN panel (WT/ESP/HF/Sniper/xCas/…), replacing the degenerate chr22.
- *Addresses #6 (leakage), #7 (degenerate OOD), #8 (strawman baselines).*

## Phase 5 — Re-run the guarantee stack on the real pipeline
Re-run marginal coverage, Mondrian, weighted conformal, and cfBH on the real oracle + real generator +
grouped splits; report real-data FAR / power / precision / coverage / interval widths with CIs. Render
the honest A-vs-B verdict from evidence.
- *Resolves or definitively characterizes #1, #5, #9.*

## Phase 6 — Reframe & finalize
Rewrite claims to match Phase-5 evidence (destination A or B); re-establish tex/md/json/figure
consistency; clean-environment reproduction.
- *Addresses #3 (position novelty honestly) and closes the loop.*

---

**Execution protocol:** one phase at a time; at each phase end, stop and report (1) what was
implemented, (2) empirical metrics (ρ, AUC, power, FAR, …), (3) blockers/observations, before
proceeding.
