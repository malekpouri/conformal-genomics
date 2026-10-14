# Correction notice (2026-10-14)

An internal audit of this repository found a **benchmark-grouping defect** that invalidates
several of the headline results previously stated in `README.md`, in the dashboard at
`docs/index.html`, and in `remediation/results/`. This notice records what was wrong, what the
corrected numbers are, and what still stands. The audit scripts that produce every number below
are in [`audit/`](audit/) and run on CPU.

## What was wrong

The CIRCLE-seq whole-dataset file (`I_1_CIRCLE_seq_10gRNA_wholeDataset.csv`) contains
**ten guides**, but lists them under **800 distinct on-target strings** — 80 after removing the
bulge marker. The 80 are 10 protospacers × 8 representations: four PAM variants
(NAGG/NCGG/NGGG/NTGG) crossed with two alignment hypotheses (a mismatch-only alignment and a
bulge alignment; the leading `-` is an alignment pad at index 0, not a bulge).

Our grouped cross-validation keyed on the sequence column (`sgRNA_seq`), not on the canonical
on-target (`sgRNA_type`). Because feature extraction truncates the guide to its first 20 nt, the
guide-side features of all eight representations are **bit-identical**. A split that was reported
as guide-disjoint therefore trained and tested on the same guide.

## Corrected numbers

Same learner, same features, same hyperparameters; only the grouping key differs.

| quantity | as published | corrected (grouped on `sgRNA_type`) |
|---|---|---|
| off-target oracle pair ROC-AUC | 0.925 | **0.762** |
| average precision (positive rate 1.26%) | 0.281 | **0.034** |
| per-guide burden fidelity ρ | 0.944 – 1.000 | **0.018** [−0.80, 0.66] |
| cfBH power at q = 0.10 / 0.20 | 0.584 / 0.796 | **0.000 / 0.000** |

A per-guide correlation of exactly 1.000 under the published grouping is itself the diagnostic:
the model was recalling guides it had already seen.

Three further claims do not survive:

- **Uncertainty repair.** The deep-ensemble and MC-dropout arms formed intervals from the spread
  of member means only, omitting the predictive-noise term their source papers specify. The
  reported coverage of 0.024 at nominal 0.90 is reproduced exactly under that recipe, but with
  the mixture predictive variance of Lakshminarayanan et al. (2017) the same ensemble covers at
  **0.911**, and MC-dropout with the observation-noise term of Gal & Ghahramani (2016) covers at
  **0.919**. Split conformal covers at 0.903. Conformal is a competitive alternative, not a repair.
- **Certificate versus tuned threshold.** The reported violation gap is a *yield* gap. Constrained
  to accept exactly what cfBH accepted, the tuned threshold's FAR is 0.001 — identical to cfBH's —
  and its violation rate falls from 0.37–0.47 to 0.01.
- **"~80 validated guides" protocol.** This counted representations, not guides. The replacement
  is a feasibility bound (below).
- **Domain-classifier accuracy 0.778** on nuclease transfer was duplicate memorisation; with a
  sequence-disjoint split it is 0.506, i.e. chance. eSpCas9 and SpCas9-HF1 share 93.7% and 95.4%
  of their guide sets with the wild-type calibration data, so there is no covariate shift to
  reweight on those panels.

## What stands, and what is new

- In-distribution split-conformal coverage is nominal (0.899–0.900 at 1−α = 0.90, n = 55,603).
- FAR control held in every configuration examined; cfBH fails safe by selecting nothing.
- **New — a feasibility bound.** Conformal p-values are at least 1/(n_cal+1), so BH at level q
  over m candidates rejects the k-th smallest only if p₍ₖ₎ ≤ qk/m. A non-empty accepted set needs
  at least ⌈m/(q(n_cal+1))⌉ candidates at the minimal p-value, and certifying one candidate needs
  **n_cal ≥ m/q − 1**. Certifying one guide from a pool of 2,000 at q = 0.10 needs 19,999
  validated calibration guides; the public cohort supplies ten. This explains the empty accepted
  sets without any appeal to oracle quality.
- **New — leak-free baselines.** Per-guide off-target burden is predicted almost perfectly by the
  free count of enumerated near-match sites (ρ = 0.964 [0.79, 1.00]); at the pair level the
  training-free MIT/Hsu–Zhang score (AUC 0.783) is not beaten by the 181-feature oracle
  (AUC 0.762). Removing the oracle's one-hot guide-identity channels *improves* every metric
  (AUC 0.790, AP 0.045, ρ 0.867).
- **New — a transfer remedy.** Recalibrating on 50 target-domain labels restores nominal coverage
  (0.899–0.906) in every nuclease and cell-line panel.

## Scope

This notice covers the claims published in this repository and its GitHub Pages dashboard. No
version of this work was posted to arXiv or bioRxiv, and none was published in a journal, so
there is no external record to correct. A new manuscript built on the corrected analysis is in
preparation.

Reproduce: `python audit/a1_regrouped_oracle_cfbh.py` through `a6_...py`, then
`python audit/plot_audit_figures.py`. Raw inputs are the public CIRCLE-seq, SITE-Seq, GUIDE-seq
and CRISPR_HNN files; they are not redistributed here.
