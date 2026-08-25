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

> **What this is.** ConformalGen is an open-source statistical library for **generative biological
> sequence design**, validated against **experimentally measured off-target cleavage**. Its working core
> is **off-target safety selection**: an off-target oracle grounded in **CIRCLE-seq** (ROC-AUC 0.925)
> drives a **conformalized-Benjamini–Hochberg (cfBH)** step that provably controls the False Acceptance
> Rate (FAR ≤ q) while recovering **58–80% of truly-safe guides** on real validated data. It also wraps
> any predictor in **split-conformal prediction** for finite-sample, distribution-free (1−α) coverage.

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
bars were calibrated. Split-conformal prediction fixes this with an **exact finite-sample guarantee**
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

| Result | Finding |
|---|---|
| **Validated off-target oracle** | trained on CIRCLE-seq, guide-disjoint **ROC-AUC 0.925** (burden fidelity ρ 0.944) — vs 0.696 for a naive mismatch count |
| **Real-data cfBH (primary)** | on **validated** CIRCLE-seq truth: **FAR ≤ q** with power **0.58** (q=0.10) / **0.80** (q=0.20) at precision ≥ 0.99 |
| **Real generator** | char-level autoregressive GRU on 55,603 sequences → 100% novel/unique candidates |
| **Coverage & leakage** | marginal coverage exact at 55k scale; **unchanged under grouped splits** (no leakage inflation) |
| **Baselines** | deep ensemble (0.02) and MC-dropout (0.54) miscalibrate at nominal 0.90; **conformalizing the same ensemble → 0.91 (exact)** |
| **Deployment & fail-safe** | off-target burden needs **alignment-enumerated** sites (ρ 0.94; **ρ 0.879 in the Cas-OFFinder mm≤3 regime**); from sequence alone ρ ≈ 0 → cfBH **fails safe** (selects nothing) rather than falsely accepting |
| **Certificate vs tuning** | a calibration-**tuned** threshold violates FAR ≤ q on **36–42% of deployments** (p95 FAR 0.27–0.39); cfBH's distribution-free certificate never does |
| **Honest limits** | indel-efficacy oracle caps at Spearman ≤ 0.78 → efficacy/joint cfBH power ≈ 0; coverage does **not** transfer across nuclease/cell-line |

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

## Citation

```bibtex
@software{conformalgen,
  author  = {Malekpouri, Mohammad},
  title   = {ConformalGen: Distribution-Free Uncertainty Quantification for Generative Genomic Design},
  year    = {2026},
  url     = {https://github.com/malekpouri/conformal-genomics},
  license = {MIT}
}
```

## License

[MIT](LICENSE) © Mohammad Malekpouri.
