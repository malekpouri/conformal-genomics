"""Conformal selection with finite-sample FAR control (cfBH; Jin & Candes 2023).

Rigorous construction (no pre-filtering of calibration nulls). We reduce the multi-objective design
criterion to a single scalar target whose sign encodes design-satisfaction:

    T(y) = min{ (y_eff - tau_eff)/s_eff ,  (log(1+tau_off) - log(1+y_off))/s_off } ,
    OK(y)  <=>  T(y) >= 0 ,

for fixed positive scales s_eff, s_off (a min >= 0 iff both terms >= 0, regardless of the scales, so
the equivalence to OK is exact). A predictor T_hat(x) is trained on D_tr, and the monotone score is

    V(x, t) = t - T_hat(x).

For each test candidate j we test H_j: T_j < 0 (guide j is NOT design-satisfying) by comparing the
boundary score V(X_j, 0) = -T_hat(X_j) against the calibration scores V_i = V(X_i, T_i) over the
ENTIRE calibration set (Jin & Candes 2023, Eq. for cfBH):

    p_j = ( 1 + sum_{i=1}^n 1{ V(X_i, T_i) <= V(X_j, 0) } ) / (n + 1).

Under exchangeability these p_j are (super-)uniform for null candidates (monotonicity of V in t makes
the boundary substitution conservative), so Benjamini-Hochberg on {p_j} controls the False Acceptance
Rate FAR = E[ #{accepted, not OK} / max(#accepted,1) ] <= q in finite samples. Pure NumPy + sklearn;
CPU, 0 GB VRAM.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .models.featurize import featurize

_HGB = dict(max_iter=300, max_depth=3, learning_rate=0.05, min_samples_leaf=20,
            l2_regularization=1.0, random_state=42)


def target_transform(y_eff, y_off, tau_eff, tau_off, s_eff, s_off):
    """Scalar design target T(y); OK(y) <=> T(y) >= 0."""
    y_eff = np.asarray(y_eff, float); y_off = np.asarray(y_off, float)
    t_eff = (y_eff - tau_eff) / s_eff
    t_off = (np.log1p(tau_off) - np.log1p(y_off)) / s_off
    return np.minimum(t_eff, t_off)


def conformal_pvalues(V_cal, V_test_boundary, randomize=False, rng=None):
    """Marginal conformal p-values for cfBH (Jin & Candes 2023, Eq. for the calibration-conditional
    p-value). With boundary score b_j = V(X_j, 0):

      randomized (exactly uniform under H, no-ties not required):
          p_j = ( #{i: V_i < b_j} + U_j*(1 + #{i: V_i = b_j}) ) / (n+1),  U_j ~ Unif(0,1)
      deterministic (default; CONSERVATIVE, i.e. super-uniform -> valid, exact when scores are
      a.s. distinct so there are no ties):
          p_j = ( 1 + #{i: V_i <= b_j} ) / (n+1)                         [ = randomized with U_j := 1 ]

    Both are valid for BH; the deterministic form is used by default (reproducible) and the manuscript
    reports it, stating the no-strict-ties assumption under which it is exact."""
    Vc = np.sort(np.asarray(V_cal, float)); n = Vc.size
    b = np.asarray(V_test_boundary, float)
    lt = np.searchsorted(Vc, b, side="left")     # #{V_cal <  b}
    le = np.searchsorted(Vc, b, side="right")    # #{V_cal <= b}
    if randomize:
        U = (rng or np.random.default_rng()).random(b.shape)
        return (lt + U * (1 + (le - lt))) / (n + 1.0)
    return (1.0 + le) / (n + 1.0)                # conservative (U:=1); ties handled via `<=`


def bh_select(pvals, q):
    """Benjamini-Hochberg at level q; returns a boolean acceptance mask."""
    p = np.asarray(pvals, float); m = p.size
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, m + 1)) / m
    mask = np.zeros(m, bool)
    if passed.any():
        cutoff = p[order][np.max(np.nonzero(passed)[0])]
        mask = p <= cutoff
    return mask


class ConformalSelector:
    """cfBH conformal selection for the design criterion OK(y) = {y_eff>=tau_eff and y_off<=tau_off}."""

    def __init__(self, tau_eff, tau_off):
        self.tau_eff = float(tau_eff); self.tau_off = float(tau_off)
        self.reg = None; self.s_eff = None; self.s_off = None; self.V_cal = None

    def fit(self, train_seqs, train_eff, train_off):
        """Fit T_hat on D_tr; fix scales from training-label dispersion."""
        train_eff = np.asarray(train_eff, float); train_off = np.asarray(train_off, float)
        self.s_eff = float(np.std(train_eff)) or 1.0
        self.s_off = float(np.std(np.log1p(train_off))) or 1.0
        T = target_transform(train_eff, train_off, self.tau_eff, self.tau_off, self.s_eff, self.s_off)
        self.reg = HistGradientBoostingRegressor(**_HGB).fit(featurize(train_seqs), T)
        return self

    def _That(self, seqs):
        return self.reg.predict(featurize(seqs))

    def calibrate(self, cal_seqs, cal_eff, cal_off):
        """Calibration scores V_i = T_i - T_hat(X_i) over the ENTIRE calibration set (no filtering)."""
        T = target_transform(cal_eff, cal_off, self.tau_eff, self.tau_off, self.s_eff, self.s_off)
        self.V_cal = T - self._That(cal_seqs)
        return self

    def pvalues(self, test_seqs, randomize=False, rng=None):
        # boundary V(X_j, 0) = 0 - T_hat(X_j); V is monotone increasing in t so t=0 >= true (null) t
        return conformal_pvalues(self.V_cal, -self._That(test_seqs), randomize=randomize, rng=rng)

    def select(self, test_seqs, q):
        return bh_select(self.pvalues(test_seqs), q)

    def evaluate(self, test_seqs, test_eff, test_off, q):
        """4-metric panel: FAR, yield, precision, and selection power (recall of true OK)."""
        te = np.asarray(test_eff, float); to = np.asarray(test_off, float)
        ok = (te >= self.tau_eff) & (to <= self.tau_off)
        sel = self.select(test_seqs, q)
        n_sel = int(sel.sum()); n_ok = int(ok.sum())
        far = float(np.mean(~ok[sel])) if n_sel else 0.0
        prec = float(np.mean(ok[sel])) if n_sel else float("nan")
        power = float(np.sum(ok & sel) / n_ok) if n_ok else float("nan")
        return {"q": float(q), "n_candidates": len(test_seqs), "n_selected": n_sel, "n_true_ok": n_ok,
                "yield": float(n_sel / len(test_seqs)), "empirical_FAR": far,
                "precision": prec, "power_recall": power}
