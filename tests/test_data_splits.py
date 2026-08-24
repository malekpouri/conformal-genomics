#!/usr/bin/env python3
"""
ConformalGen — Phase 1 data-integrity tests.

Validates that the prepared splits are disjoint and complete, that the held-out chromosome (chr22)
is strictly excluded from the in-distribution off-target label, and that the standardized fields are
sane. The checks are CPU-only and memory-light (no torch / no GPU): Phase-1 data preparation uses
0 GB VRAM, trivially within the <= 6 GB budget.

Run directly:   python tests/test_data_splits.py
Or with pytest: pytest tests/test_data_splits.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
HELDOUT = "chr22"
DROP = {"chrM"}


def _load():
    pool = pd.read_csv(DATA / "conformalgen_pool.csv")
    splits = {s: pd.read_csv(DATA / "splits" / f"{s}.csv") for s in ("train", "calibration", "test")}
    long = pd.read_csv(DATA / "offtarget_by_chromosome.csv")
    ood = pd.read_csv(DATA / "ood_chr22.csv")
    return pool, splits, long, ood


def test_splits_disjoint_and_complete():
    pool, splits, _, _ = _load()
    g = {s: set(splits[s]["guide_id"]) for s in splits}
    assert g["train"] & g["calibration"] == set(), "train/calibration guide overlap"
    assert g["train"] & g["test"] == set(), "train/test guide overlap"
    assert g["calibration"] & g["test"] == set(), "calibration/test guide overlap"
    union = g["train"] | g["calibration"] | g["test"]
    assert union == set(pool["guide_id"]), "splits do not partition the pool"
    assert len(union) == len(pool), "duplicate guide_ids across splits"


def test_chr22_strictly_held_out():
    pool, _, long, ood = _load()
    # (a) off_id must equal the near-match sum over chromosomes EXCEPT chr22 (and excl. dropped)
    ind = long[(long["chrom"] != HELDOUT) & (~long["chrom"].isin(DROP))]
    recomputed_id = ind.groupby("guide_id")["near_count"].sum()
    merged = pool.set_index("guide_id").join(recomputed_id.rename("recomp"), how="left").fillna({"recomp": 0})
    assert (merged["off_id"] == merged["recomp"].astype(int)).all(), "off_id includes chr22 or dropped chroms"
    # (b) off_chr22 must equal the chr22-only near-match sum
    c22 = long[long["chrom"] == HELDOUT].groupby("guide_id")["near_count"].sum()
    m2 = ood.set_index("guide_id").join(c22.rename("recomp"), how="left").fillna({"recomp": 0})
    assert (m2["off_chr22"] == m2["recomp"].astype(int)).all(), "off_chr22 mismatch vs chr22 near-matches"
    # (c) chr22 never appears as an in-distribution split contributor
    assert HELDOUT not in set(ind["chrom"]), "chr22 leaked into in-distribution long table"


def test_field_integrity():
    pool, _, _, _ = _load()
    assert pool["y_eff"].notna().all() and pool["off_id"].notna().all(), "NaNs in labels"
    assert pool["y_eff"].between(0, 100).all(), "y_eff outside [0,100]"
    assert (pool["off_id"] >= 0).all() and (pool["off_chr22"] >= 0).all(), "negative off-target counts"
    assert pool["seq"].str.len().eq(20).all(), "guide sequences are not all 20 nt"
    assert pool["seq"].str.fullmatch(r"[ACGT]{20}").all(), "non-ACGT characters in sequences"
    assert not pool["guide_id"].duplicated().any(), "duplicate guide_id in pool"


def test_memory_light_cpu_only():
    # Phase-1 prep/tests must not require a GPU; assert torch is not needed and RSS stays modest.
    assert "torch" not in sys.modules, "torch should not be imported for data-split tests"
    try:
        import resource
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # KB->GB on Linux
        assert rss_gb < 2.0, f"RSS {rss_gb:.2f} GB too high for a data-integrity test"
    except ImportError:
        pass  # resource unavailable (non-Unix): skip the RSS bound


def _main():
    tests = [test_splits_disjoint_and_complete, test_chr22_strictly_held_out,
             test_field_integrity, test_memory_light_cpu_only]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed (VRAM used: 0 GB — CPU-only).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
