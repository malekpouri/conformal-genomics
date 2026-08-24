"""Conformal-guided generation & acceptance policy (RFC-001 §3.3, Algorithm 2).

A UNIVERSAL, model-agnostic statistical wrapper: it turns any biological-sequence generator into a
guarantee-aware design pipeline. The generator is treated as an abstract, pluggable interface
(`GenericSequenceGenerator`) — the wrapper never inspects generator internals, so it applies equally
to standard SFT priors, autoregressive LLMs, diffusion samplers, or pre-aligned policies.

Given a fitted PropertyPredictors and a calibration set, each candidate guide x is mapped to
calibrated directional bounds L_eff(x), U_off(x) (and joint inf-norm bounds), then ACCEPTED iff

        L_eff(x) >= tau_eff   AND   U_off(x) <= tau_off .

Because acceptance uses *guaranteed* bounds rather than point predictions, an accepted guide whose
bounds cover its oracle labels is provably design-satisfying:
        (L_eff >= tau_eff and y_eff >= L_eff)  =>  y_eff >= tau_eff, and symmetrically for off-target.
Hence post-selection bound-coverage lower-bounds design precision (§3.3).

Off-target is modelled on the log1p scale (predictor space); bounds map back to counts via expm1.
No GPU, no torch: pure sklearn predictors + NumPy.  See also src/conformal.py, src/scores.py.
"""
from __future__ import annotations

import numpy as np

from .models.featurize import featurize
from .scores import conformal_quantile

MODES = ("point", "conformal_directional", "conformal_joint")


# ═════════════════════════════════════════════════════════════════════════════
# Pluggable generator interface (model-agnostic)
# ═════════════════════════════════════════════════════════════════════════════
class GenericSequenceGenerator:
    """Abstract, pluggable sequence generator.

    Any external generator — SFT prior, autoregressive LLM, diffusion sampler, aligned policy — can
    implement `.sample(n_samples, seed) -> (seqs, y_eff, y_off)` and be filtered by the conformal
    wrapper unchanged. `y_eff`/`y_off` are oracle labels used only for post-hoc evaluation of coverage;
    a live generator that emits unlabeled candidates simply returns NaN label arrays (the wrapper still
    produces calibrated bounds and accept/reject decisions from sequences alone).
    """
    name = "generic"

    def sample(self, n_samples, seed=0):
        raise NotImplementedError


class ReservoirGenerator(GenericSequenceGenerator):
    """Reference generator backed by a labelled reservoir, with an optional preference `tilt`.

    tilt == 0  -> uniform draw: a standard *unaligned* generative prior (SFT-style baseline).
    tilt  > 0  -> draw weighted by softmax(tilt * [z(y_eff) - z(log1p(off))]): a generic *aligned*
                  prior favouring high-efficacy / low-off-target candidates.

    This is a transparent, dependency-free stand-in for a live generator that carries REAL oracle
    labels, so post-selection coverage is genuine. Swap in any `GenericSequenceGenerator` subclass
    (e.g. one wrapping a trained sampler) without touching the conformal wrapper.
    """

    def __init__(self, seqs, y_eff, y_off, tilt=0.0, name=None):
        self.seqs = list(seqs)
        self.y_eff = np.asarray(y_eff, float)
        self.y_off = np.asarray(y_off, float)
        self.tilt = float(tilt)
        self.name = name or (f"aligned_prior(tilt={tilt})" if tilt > 0 else "sft_prior")

    def weights(self):
        n = len(self.seqs)
        if self.tilt <= 0:
            return np.full(n, 1.0 / n)
        z_eff = (self.y_eff - self.y_eff.mean()) / (self.y_eff.std() + 1e-9)
        zl = np.log1p(self.y_off); zl = (zl - zl.mean()) / (zl.std() + 1e-9)
        logits = self.tilt * (z_eff - zl)
        w = np.exp(logits - logits.max())
        return w / w.sum()

    def sample(self, n_samples, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.seqs), size=n_samples, replace=True, p=self.weights())
        return [self.seqs[i] for i in idx], self.y_eff[idx], self.y_off[idx]

    def full(self):
        """Return the raw reservoir as-is (every candidate, no resampling) — for raw-pool evaluation."""
        return list(self.seqs), self.y_eff.copy(), self.y_off.copy()


# ═════════════════════════════════════════════════════════════════════════════
# Conformal guided-generation wrapper
# ═════════════════════════════════════════════════════════════════════════════
class ConformalGuidedGenerator:
    """Wraps predictors + a calibration set to filter candidate guides with coverage guarantees.

    Generator-agnostic: `evaluate`/`accept` operate on sequences (+ optional oracle labels), and
    `run` accepts any `GenericSequenceGenerator`."""

    def __init__(self, predictors):
        self.pp = predictors
        self.q = {}   # q[alpha] = {"low":..., "up":..., "mo":...}

    # ── calibration: directional & joint conformal quantiles on D_cal ──
    def calibrate(self, cal_seq, cal_eff, cal_off, alphas=(0.10, 0.05)):
        cal_eff = np.asarray(cal_eff, float)
        z_off = np.log1p(np.asarray(cal_off, float))
        mu_eff = self.pp.eff.predict_mu(cal_seq)
        zmu_off = self.pp.off.mu.predict(featurize(cal_seq))
        sig_eff = self.pp.eff.predict_scale(cal_seq)
        sig_off = self.pp.off.predict_scale(cal_seq)
        for a in alphas:
            q_low = conformal_quantile(mu_eff - cal_eff, a)          # signed lower (efficacy)
            q_up = conformal_quantile(z_off - zmu_off, a)            # signed upper (off, log space)
            s_mo = np.maximum((mu_eff - cal_eff) / sig_eff, (z_off - zmu_off) / sig_off)
            q_mo = conformal_quantile(s_mo, a)
            self.q[float(a)] = {"low": float(q_low), "up": float(q_up), "mo": float(q_mo)}
        return self

    # ── per-candidate predictions & calibrated bounds ──
    def bounds(self, seqs, alpha):
        q = self.q[float(alpha)]
        Xf = featurize(seqs)
        mu_eff = self.pp.eff.predict_mu(seqs)
        zmu_off = self.pp.off.mu.predict(Xf)
        sig_eff = self.pp.eff.predict_scale(seqs)
        sig_off = self.pp.off.predict_scale(seqs)
        return {
            "mu_eff": mu_eff,
            "mu_off": np.expm1(zmu_off),
            "L_dir": mu_eff - q["low"],                              # directional lower bound, efficacy
            "U_dir": np.expm1(zmu_off + q["up"]),                    # directional upper bound, off (counts)
            "L_mo": mu_eff - q["mo"] * sig_eff,                      # joint inf-norm lower bound, efficacy
            "U_mo": np.expm1(zmu_off + q["mo"] * sig_off),          # joint inf-norm upper bound, off
        }

    def _accept_bounds(self, b, mode):
        if mode == "conformal_directional":
            return b["L_dir"], b["U_dir"]
        if mode == "conformal_joint":
            return b["L_mo"], b["U_mo"]
        return None, None

    def accept(self, seqs, tau_eff, tau_off, alpha, mode="conformal_directional"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        b = self.bounds(seqs, alpha)
        if mode == "point":
            return (b["mu_eff"] >= tau_eff) & (b["mu_off"] <= tau_off)
        L, U = self._accept_bounds(b, mode)
        return (L >= tau_eff) & (U <= tau_off)

    # ── full evaluation on a labelled candidate pool ──
    def evaluate(self, seqs, y_eff, y_off, tau_eff, tau_off, alpha, mode="conformal_directional"):
        y_eff = np.asarray(y_eff, float); y_off = np.asarray(y_off, float)
        b = self.bounds(seqs, alpha)
        mask = self.accept(seqs, tau_eff, tau_off, alpha, mode)
        n, n_acc = len(seqs), int(mask.sum())
        out = {"mode": mode, "tau_eff": float(tau_eff), "tau_off": float(tau_off),
               "alpha": float(alpha), "n_candidates": n, "n_accepted": n_acc,
               "yield": float(n_acc / n) if n else 0.0,
               "design_precision": None, "post_selection_bound_coverage": None}
        labelled = np.all(np.isfinite(y_eff)) and np.all(np.isfinite(y_off))
        if n_acc and labelled:
            ya, yo = y_eff[mask], y_off[mask]
            out["design_precision"] = float(np.mean((ya >= tau_eff) & (yo <= tau_off)))
            L, U = self._accept_bounds(b, mode)
            if L is not None:
                out["post_selection_bound_coverage"] = float(np.mean((ya >= L[mask]) & (yo <= U[mask])))
        return out

    def run(self, generator, n_samples, tau_eff, tau_off, alpha, mode="conformal_directional", seed=0):
        """Sample from any GenericSequenceGenerator, then evaluate the acceptance policy."""
        seqs, ye, yo = generator.sample(n_samples, seed=seed)
        return self.evaluate(seqs, ye, yo, tau_eff, tau_off, alpha, mode)
