"""Sequence featurization for ConformalGen predictors (CPU, dependency-light).

Guides are fixed-length 20-nt protospacers. We use a position-wise one-hot encoding
(4 x 20 = 80 features); optionally augmented with GC content. This keeps the base
predictors lightweight and CPU-only (0 GB VRAM), well within the compute budget.
"""
from __future__ import annotations
import numpy as np

NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
SEQ_LEN = 20


def one_hot(seqs) -> np.ndarray:
    """Position-wise one-hot: (N, 4*L) float32, row-major over (position, base)."""
    seqs = list(seqs)
    X = np.zeros((len(seqs), 4 * SEQ_LEN), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            X[i, j * 4 + NUC[c]] = 1.0
    return X


def gc_content(seqs) -> np.ndarray:
    return np.array([[(s.count("G") + s.count("C")) / len(s)] for s in seqs], dtype=np.float32)


def featurize(seqs, with_gc: bool = True) -> np.ndarray:
    """Default feature matrix: one-hot (+ GC). (N, 80) or (N, 81)."""
    oh = one_hot(seqs)
    return np.concatenate([oh, gc_content(seqs)], axis=1) if with_gc else oh
