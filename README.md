<h1 align="center">ConformalGen</h1>
<p align="center">
  <b>Distribution-free uncertainty quantification for generative genomic design</b><br/>
  <sub>A model-agnostic conformal-prediction wrapper with finite-sample coverage guarantees</sub>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  <img alt="VRAM" src="https://img.shields.io/badge/VRAM-0GB%20(CPU)-76b900.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-38%20passing-brightgreen.svg">
</p>

> ⚠️ **Correction (2026-10-14).** Earlier versions of this README reported an off-target oracle
> ROC-AUC of 0.925 and cfBH power of 0.58–0.80. Those figures came from cross-validation grouped on
> a column that encodes each of the benchmark's ten guides eight times; corrected, they are 0.762
> and 0.000. Read **[CORRECTION.md](CORRECTION.md)** before using any number from this repository.

> **What this is.** ConformalGen is an open-source statistical library for **generative biological
> sequence design**, evaluated against **experimentally measured off-target cleavage**. It wraps any
> predictor in **split-conformal prediction** for finite-sample, distribution-free (1−α) coverage, and
> adds a **conformalized-Benjamini–Hochberg (cfBH)** step that controls the False Acceptance Rate
> (FAR ≤ q). The guarantees hold; what the audit in [`audit/`](audit/) shows is that the *data* needed
> to make them bite — validated per-guide off-target cleavage — are roughly two orders of magnitude
> short of the requirement **n_cal ≥ m/q − 1**, so on public cohorts cfBH correctly certifies nothing.

📊 **Live dashboard:** [malekpouri.github.io/conformal-genomics](https://malekpouri.github.io/conformal-genomics)

ConformalGen is **model-agnostic** *by formulation*. The sequence generator and the property oracles
are treated as opaque, pluggable components: any object that emits candidate sequences — an SFT
language model, an autoregressive sampler, a diffusion model, or a preference-aligned policy — is a
drop-in. ConformalGen never modifies or retrains them; it converts their outputs into checkable
guarantees.

## Why conformal?

A point predictor $\hat\mu(x)$ gives a number, not a guarantee. Filtering "accept if
$\hat\mu_\text{eff}\ge\tau$" controls nothing: you cannot say how often an accepted guide is actually
below threshold, and generative sampling shifts the candidate distribution away from where your error
bars were calibrated. Split-conformal prediction fixes this with a finite-sample guarantee
under exchangeability:

$$1-\alpha \;\le\; \Pr\big(y \in C(x)\big) \;\le\; 1-\alpha + \tfrac{1}{n+1}.$$

## Features

- **Four non-conformity scores** — absolute residual, signed directional (one-sided efficacy lower /
  off-target upper bounds), conformalized quantile regression (CQR), and a multi-objective
  standardized $\infty$-norm score for **single-quantile joint** coverage over several properties at
  once (no union bound).
- **Selective acceptance with FAR control** — a Benjamini–Hochberg **conformal-selection (cfBH)** step
  (Jin & Candès 2023) that provably controls the False Acceptance Rate ($\mathrm{FAR}\le q$).
- **Conditional & shift-robust calibration** — Mondrian (group-conditional) and
  covariate-shift-weighted conformal (82-feature sequence-composition density ratio) for heavy-tailed
  strata and out-of-distribution transfer.
- **Honest by design** — coverage is w.r.t. the *oracle labels*; a CIRCLE-seq/Tsai audit quantifies
  surrogate fidelity, and a sensitivity study shows selection power tracks oracle quality while FAR
  control holds unconditionally.
- **Commodity hardware** — the full reference pipeline runs on **CPU (0 GB VRAM)**.
- **Reproducible** — one script regenerates every metric and figure deterministically.

## Installation

```bash
git clone https://github.com/malekpouri/conformal-genomics.git
cd conformal-genomics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Core dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

## Quickstart

```python
import joblib
from src.guided_generation import ConformalGuidedGenerator, ReservoirGenerator

# 1) load (or fit) property predictors and calibrate on a held-out set
pp = joblib.load("models/predictors.joblib")
gg = ConformalGuidedGenerator(pp).calibrate(cal_seq, cal_eff, cal_off, alphas=(0.10, 0.05))

# 2) wrap ANY generator and filter with guaranteed bounds
gen = ReservoirGenerator(candidate_seqs, y_eff, y_off, tilt=1.2)   # or your own GenericSequenceGenerator
result = gg.run(gen, n_samples=2000, tau_eff=50.0, tau_off=12.0,
                alpha=0.10, mode="conformal_directional")
print(result["yield"], result["design_precision"], result["post_selection_bound_coverage"])
```

For finite-sample FAR control on the accept/reject decision:

```python
from src.selection import ConformalSelector
sel = ConformalSelector(tau_eff=45.0, tau_off=20.0).fit(train_seq, train_eff, train_off)
sel.calibrate(cal_seq, cal_eff, cal_off)
print(sel.evaluate(test_seq, test_eff, test_off, q=0.10))   # FAR <= q, + yield/precision/power
```

Plug in your own generator by subclassing `GenericSequenceGenerator` and implementing
`.sample(n_samples, seed) -> (seqs, y_eff, y_off)`.

## Reproduce everything

```bash
./reproduce.sh
```

Runs Stages 01–09 (data → predictors → calibration → guided generation → RQ benchmarks → figures →
ablation/FAR → biological audit → selection sensitivity), the **validated-oracle remediation studies**
(`remediation/` — CIRCLE-seq off-target oracle, real generator, honest leakage/OOD/baseline analyses),
and the full 38-test suite. The prepared `data/` and committed metrics/figures ship with the repo, so a
fresh clone reproduces from Stage 02 onward; stages needing the large, non-redistributed raw sources
(genome hits, CIRCLE-seq, CRISPR_HNN) auto-skip when absent. Outputs: `results/json/`,
`remediation/results/`, and 300-DPI figures in `figures/`.

## Results at a glance

> Several rows of this table previously reported inflated figures caused by a benchmark-grouping
> defect. See **[CORRECTION.md](CORRECTION.md)** for the full account; the numbers below are the
> corrected ones, and [`audit/`](audit/) reproduces every one of them.

| Result | Finding |
|---|---|
| **Benchmark leakage (new)** | the CIRCLE-seq file lists **10 guides under 80 on-target strings** (4 PAM variants × 2 alignment hypotheses). Grouping on `sgRNA_seq` instead of `sgRNA_type` trains and tests on the same guide: **ROC-AUC 0.925 → 0.762**, **AP 0.281 → 0.034**, **burden ρ 1.000 → 0.018** |
| **Validated off-target oracle** | leak-free, the 181-feature oracle reaches **ROC-AUC 0.762 / AP 0.034** at a 1.26% positive rate, and carries **no** per-guide burden signal (ρ 0.018 [−0.80, 0.66]) |
| **Free baselines (new)** | per-guide burden is predicted by the free **count of enumerated near-match sites (ρ 0.964)**; at pair level the training-free **MIT/Hsu–Zhang score (AUC 0.783)** beats the trained oracle. Dropping the oracle's guide-identity channels *improves* every metric (AUC 0.790, AP 0.045, ρ 0.867) |
| **Feasibility bound (new)** | conformal p ≥ 1/(n_cal+1) ⇒ certifying one candidate from a pool of *m* at level *q* needs **n_cal ≥ m/q − 1**. Ten validated guides cannot certify anything from a pool of thousands; the accepted set is empty in **≥96.7%** of splits and cfBH power is **0.000** |
| **Real-data cfBH** | FAR ≤ q holds, and cfBH **fails safe** (selects nothing) — but at this cohort size that is all it does |
| **Real generator** | char-level autoregressive GRU on 55,603 sequences → 100% novel/unique candidates; the full generate → chr22 PAM scan → score → certify loop runs end to end and accepts nothing |
| **Coverage** | split-conformal marginal coverage is **nominal** at 55k scale (0.899–0.900 at 1−α = 0.90) |
| **UQ baselines** | the published 0.024 / 0.545 figures omit the predictive-noise term. Implemented as their papers specify, **deep ensembles cover 0.911** and **MC-dropout 0.919**; conformal covers 0.903. Conformal is assumption-free, not a repair |
| **Certificate vs tuning** | the tuned threshold's violation gap is a **yield gap**: at matched yield its FAR is **0.001**, identical to cfBH's, and its violation rate falls from 0.37–0.47 to 0.01 |
| **Transfer, and its remedy** | coverage collapses across nuclease/cell-line (0.01–0.51); **50 target-domain labels restore it to 0.899–0.906 in every panel**. eSpCas9/SpCas9-HF1 share 93.7%/95.4% of their guides with calibration, so there is no covariate shift to reweight there (domain AUC 0.506) |

## Repository layout

```
conformal-genomics/
├── src/
│   ├── models/           # 81-feature featurization + property predictors (point + CQR heads)
│   ├── scores.py         # four non-conformity scores + conformal quantile
│   ├── conformal.py      # SplitConformal / MondrianConformal / WeightedConformal
│   ├── selection.py      # cfBH conformal p-values + BH -> FAR control
│   ├── density_ratio.py  # 82-feature domain classifier -> w(x), ESS, sensitivity
│   ├── baselines.py      # uncalibrated QR, parametric Gaussian, standard split-CP
│   ├── stats_utils.py    # bootstrap + paired-delta 95% CIs
│   └── guided_generation.py  # pluggable generator interface + acceptance policy
├── scripts/              # 01_prepare_data … 09_selection_sensitivity
├── remediation/          # validated-oracle rebuild: CIRCLE-seq oracle, real generator, honest design
├── tests/                # 38 tests
├── data/                 # pool, splits/, splits_grouped/, per-chromosome table, data_card.md
├── results/json/         # pipeline metrics   ·   remediation/results/ # validated-oracle metrics
├── figures/              # publication figures (300 DPI), incl. figR1–figR3
├── docs/                 # GitHub Pages dashboard
└── reproduce.sh          # one-line end-to-end driver (pipeline + remediation + tests)
```

## Citation & research line

ConformalGen is the certified **selection** layer that wraps a multi-objective CRISPR generator such as
**CRISPGen** (generation) → **ConformalGen** (certified safety selection):

- Malekpouri M, Lotfi S. *CRISPGen: A Deep Generative Framework for Multi-Objective CRISPR/Cas9 Guide
  RNA Design via Conditional Latent Diffusion and Dual-Critic Reinforcement Learning.* Computational
  Biology and Chemistry, 2026. [doi:10.1016/j.compbiolchem.2026.109328](https://doi.org/10.1016/j.compbiolchem.2026.109328)
  · [github.com/malekpouri/CRISPGen](https://github.com/malekpouri/CRISPGen)

```bibtex
@software{conformalgen,
  author  = {Malekpouri, Mohammad},
  title   = {ConformalGen: Distribution-Free Uncertainty Quantification for Generative Genomic Design},
  year    = {2026},
  url     = {https://github.com/malekpouri/conformal-genomics},
  license = {MIT}
}

@article{malekpouri2026crispgen,
  author  = {Malekpouri, Mohammad and Lotfi, Somayeh},
  title   = {{CRISPGen}: A Deep Generative Framework for Multi-Objective {CRISPR/Cas9} Guide {RNA} Design
             via Conditional Latent Diffusion and Dual-Critic Reinforcement Learning},
  journal = {Computational Biology and Chemistry},
  year    = {2026}, pages = {109328},
  doi     = {10.1016/j.compbiolchem.2026.109328}
}
```

## License

[MIT](LICENSE) © Mohammad Malekpouri.
