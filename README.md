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
> sequence design**. It wraps the property predictors you already have (e.g. on-target efficacy,
> off-target safety) in **split-conformal prediction**, turning point estimates into **finite-sample,
> distribution-free** prediction intervals with a user-chosen coverage level $1-\alpha$ — and a
> **bound-based acceptance policy** with an optional **conformal-selection (cfBH)** step that controls
> the False Acceptance Rate (FAR ≤ q) among accepted candidates in finite samples.

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
ablation/FAR → biological audit → selection sensitivity) and the full 38-test suite. The prepared
`data/` is shipped with the repo, so a fresh clone reproduces from Stage 02 onward; Stages 01 and 08
(rebuild from raw genomic / CIRCLE-seq sources) run only when those large, non-redistributed inputs are
present. Outputs: machine-readable metrics in `results/json/`, 300-DPI figures in `figures/`.

## Results at a glance

| Research question | Finding |
|---|---|
| **RQ1** marginal coverage | empirical coverage matches nominal to within **0.002** for all 4 scores (500 resplits); only conformal reaches nominal (parametric Gaussian covers 0.67/0.61) |
| **RQ2** efficiency & Mondrian | pooled calibration under-covers the heavy off-target tail (0.83 < 0.90); **Mondrian restores it** and tightens the low stratum ~11% |
| **RQ3** covariate shift | plain conformal over-covers a shifted pool; **weighted conformal** (82-feature density ratio) recovers coverage, reported with ESS and finite-bound proportion |
| **Selection (cfBH)** | FAR ≤ q unconditionally; power tracks oracle quality (0.98 at a perfect oracle; 0 on the real weak surrogate — an honest, oracle-limited result) |
| **Oracle fidelity** | surrogate off-target audited vs CIRCLE-seq: ROC-AUC 0.70, MM≤3 sensitivity 0.17 (specific but insensitive) |

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
├── tests/                # 38 tests
├── data/                 # pool, splits/, splits_grouped/, per-chromosome table, data_card.md
├── results/json/         # all metrics
├── figures/              # publication figures (300 DPI)
├── docs/                 # GitHub Pages dashboard
└── reproduce.sh          # one-line 9-stage end-to-end driver
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
