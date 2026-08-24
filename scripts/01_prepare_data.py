#!/usr/bin/env python3
"""
ConformalGen — Phase 1: data preparation & splits.

Standardises the oracle-labelled 1,000-guide pilot pool from existing project assets into a clean
per-guide conformal dataset with two functional properties (d=2):
  * y_eff  : on-target efficacy   = CRISPRon score (0-100), aggregated per guide
  * y_off  : off-target risk      = whole-genome near-match count (mismatches 0-3), split into
             off_id (in-distribution: all chromosomes EXCEPT the held-out chr22) and
             off_chr22 (OOD, held out for RQ3). Off-target *suppression* = low risk.

Splits are made at the GUIDE level (disjoint guides -> clean exchangeability): proper-train (D_tr),
calibration (D_cal), and in-distribution test (D_te). Chromosome 22 is strictly held out of the
in-distribution off-target label and reserved for the RQ3 out-of-distribution stress test.

CPU/pandas only (no GPU): trivially within the <=6 GB VRAM budget.

Outputs (data/):
  conformalgen_pool.csv          per-guide: guide_id, seq, y_eff, off_id, off_chr22, split
  offtarget_by_chromosome.csv    long: guide_id, seq, chrom, near_count (for Mondrian/RQ3)
  splits/{train,calibration,test}.csv
  ood_chr22.csv                  per-guide chr22 labels (RQ3)
  split_summary.json             machine-readable summary statistics
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
HELDOUT_CHROM = "chr22"
DROP_CHROM = {"chrM"}                 # mitochondrial: not a nuclear off-target concern
MAX_MM = 3
SEED = 42
FRACS = {"train": 0.60, "calibration": 0.20, "test": 0.20}


def load_efficacy() -> pd.Series:
    path = glob.glob(str(NOTEBOOK / "output_result_final*crispron.csv"))
    assert path, "CRISPRon efficacy file not found next to the notebook"
    cris = pd.read_csv(path[0])
    cris["guide_id"] = cris["ID"].str.rsplit("_p_", n=1).str[0]
    return cris.groupby("guide_id")["CRISPRon"].mean().rename("y_eff")


def load_offtarget() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(NOTEBOOK / "report" / "whole_genome_hits_raw.csv")
    mm_cols = [f"mm{m}" for m in range(MAX_MM + 1)]
    raw["near_count"] = raw[mm_cols].sum(axis=1)
    raw = raw[~raw["chrom"].isin(DROP_CHROM)].copy()
    seq = raw.groupby("guide_id")["seq"].first()
    off_id = (raw.loc[raw["chrom"] != HELDOUT_CHROM]
                 .groupby("guide_id")["near_count"].sum().rename("off_id"))
    off_ood = (raw.loc[raw["chrom"] == HELDOUT_CHROM]
                  .groupby("guide_id")["near_count"].sum().rename("off_chr22"))
    per_guide = pd.concat([seq, off_id, off_ood], axis=1)
    per_guide["off_id"] = per_guide["off_id"].fillna(0).astype(int)
    per_guide["off_chr22"] = per_guide["off_chr22"].fillna(0).astype(int)
    long = raw[["guide_id", "seq", "chrom", "near_count"]].reset_index(drop=True)
    return per_guide, long


def assign_splits(guide_ids: list[str]) -> dict[str, str]:
    rng = np.random.default_rng(SEED)
    ids = np.array(sorted(guide_ids))
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_tr = int(round(FRACS["train"] * n))
    n_cal = int(round(FRACS["calibration"] * n))
    split = {}
    for k, idx in enumerate(perm):
        gid = ids[idx]
        split[gid] = "train" if k < n_tr else ("calibration" if k < n_tr + n_cal else "test")
    return split


HAMMING_THRESH = 4                    # near-duplicate cutoff for sequence-cluster grouping


def sequence_clusters(seqs, thresh=HAMMING_THRESH):
    """Single-linkage clusters over the Hamming graph (edges when Hamming distance <= thresh).

    Groups near-duplicate / near-identical spacers so a grouped split can place a whole cluster on one
    side, testing for near-duplicate ('locus'-style) leakage that guide-level splitting can miss."""
    codes = np.array([[ord(c) for c in s] for s in seqs])
    n = len(seqs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i in range(n):
        d = (codes[i + 1:] != codes[i]).sum(axis=1)       # Hamming distance to all j>i
        for off in np.nonzero(d <= thresh)[0]:
            j = i + 1 + int(off)
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def assign_grouped_splits(guide_ids, seqs) -> tuple[dict[str, str], dict]:
    """Cluster-disjoint split: whole sequence-clusters are assigned to train/cal/test (~60/20/20)."""
    rng = np.random.default_rng(SEED)
    clusters = sequence_clusters(seqs)
    uniq = np.array(sorted(set(clusters.tolist())))
    perm = rng.permutation(len(uniq))
    # greedy fill by cumulative guide count to hit the target fractions
    sizes = {c: int((clusters == c).sum()) for c in uniq}
    n = len(guide_ids); n_tr = int(round(FRACS["train"] * n)); n_cal = int(round(FRACS["calibration"] * n))
    cum = 0; cl_split = {}
    for k in perm:
        c = uniq[k]
        cl_split[c] = "train" if cum < n_tr else ("calibration" if cum < n_tr + n_cal else "test")
        cum += sizes[c]
    split = {gid: cl_split[clusters[i]] for i, gid in enumerate(guide_ids)}
    non_singleton = int(sum(1 for c in uniq if sizes[c] > 1))
    info = {"hamming_threshold": HAMMING_THRESH, "n_clusters": int(len(uniq)),
            "n_non_singleton_clusters": non_singleton,
            "largest_cluster": int(max(sizes.values())),
            "n_guides_in_non_singleton_clusters": int(sum(v for v in sizes.values() if v > 1))}
    return split, info


def main():
    (DATA / "splits").mkdir(parents=True, exist_ok=True)
    y_eff = load_efficacy()
    per_guide, long = load_offtarget()

    df = per_guide.join(y_eff, how="inner").reset_index().rename(columns={"index": "guide_id"})
    df = df.dropna(subset=["y_eff", "seq"]).reset_index(drop=True)
    # guide-level split
    split = assign_splits(df["guide_id"].tolist())
    df["split"] = df["guide_id"].map(split)
    long = long[long["guide_id"].isin(df["guide_id"])].copy()
    long["split"] = long["guide_id"].map(split)

    # write standardized tables
    df.to_csv(DATA / "conformalgen_pool.csv", index=False)
    long.to_csv(DATA / "offtarget_by_chromosome.csv", index=False)
    for s in ("train", "calibration", "test"):
        df[df["split"] == s].to_csv(DATA / "splits" / f"{s}.csv", index=False)
    df[["guide_id", "seq", "y_eff", "off_chr22"]].to_csv(DATA / "ood_chr22.csv", index=False)

    # ---- grouped (sequence-cluster) split for leakage analysis --------------
    (DATA / "splits_grouped").mkdir(parents=True, exist_ok=True)
    gsplit, ginfo = assign_grouped_splits(df["guide_id"].tolist(), df["seq"].tolist())
    dfg = df.copy(); dfg["split"] = dfg["guide_id"].map(gsplit)
    for s in ("train", "calibration", "test"):
        dfg[dfg["split"] == s].to_csv(DATA / "splits_grouped" / f"{s}.csv", index=False)
    ginfo["split_sizes"] = {s: int((dfg["split"] == s).sum()) for s in ("train", "calibration", "test")}
    (DATA / "splits_grouped" / "grouped_split_info.json").write_text(json.dumps(ginfo, indent=2))

    # ---- summary statistics -------------------------------------------------
    def desc(x):
        x = np.asarray(x, float)
        return {"n": int(len(x)), "mean": round(float(x.mean()), 3),
                "sd": round(float(x.std(ddof=1)), 3) if len(x) > 1 else 0.0,
                "min": round(float(x.min()), 3), "median": round(float(np.median(x)), 3),
                "max": round(float(x.max()), 3)}

    chrom_counts = (long[long["chrom"] != HELDOUT_CHROM]
                    .groupby("chrom")["guide_id"].count().sort_values(ascending=False))
    # exchangeability check: KS test of y_eff and off_id between each split pair (random split -> expect NS)
    def ks(a, b, col):
        s, p = stats.ks_2samp(df.loc[df.split == a, col], df.loc[df.split == b, col])
        return {"ks_stat": round(float(s), 4), "p_value": round(float(p), 4)}
    exch = {f"{col}:{a}-vs-{b}": ks(a, b, col)
            for col in ("y_eff", "off_id") for a, b in [("train", "calibration"), ("train", "test"), ("calibration", "test")]}

    off_id_pct = {f"p{q}": int(np.percentile(df["off_id"], q)) for q in (25, 50, 75, 90, 95)}
    tau_off_candidate = int(np.percentile(df["off_id"], 50))   # baseline safety floor (median in-dist risk)

    summary = {
        "n_guides_total": int(len(df)),
        "held_out_chromosome": HELDOUT_CHROM,
        "dropped_chromosomes": sorted(DROP_CHROM),
        "max_mismatch": MAX_MM,
        "seed": SEED,
        "split_sizes": {s: int((df["split"] == s).sum()) for s in ("train", "calibration", "test")},
        "distinct_guides_per_split": {s: int(df.loc[df.split == s, "guide_id"].nunique())
                                      for s in ("train", "calibration", "test")},
        "guide_overlap_across_splits": int(  # must be 0
            len(set(df[df.split == "train"].guide_id) & set(df[df.split == "calibration"].guide_id))
            + len(set(df[df.split == "train"].guide_id) & set(df[df.split == "test"].guide_id))
            + len(set(df[df.split == "calibration"].guide_id) & set(df[df.split == "test"].guide_id))),
        "y_eff": desc(df["y_eff"]),
        "off_id_in_distribution": desc(df["off_id"]),
        "off_chr22_OOD": desc(df["off_chr22"]),
        "off_id_percentiles": off_id_pct,
        "tau_eff": 50.0,
        "tau_off_candidate_baseline_floor": tau_off_candidate,
        "n_in_distribution_chromosomes": int(long.loc[long.chrom != HELDOUT_CHROM, "chrom"].nunique()),
        "records_per_chromosome_in_distribution": {c: int(v) for c, v in chrom_counts.items()},
        "long_table_rows": int(len(long)),
        "calibration_guarantee_slack_1_over_n_plus_1": round(1.0 / (int((df.split == "calibration").sum()) + 1), 5),
        "exchangeability_ks_tests": exch,
        "guide_id_efficacy_calibration_note": (
            "conformal unit is the GUIDE (one record per guide); off_id aggregates genome-minus-chr22 "
            "near-matches, off_chr22 is the held-out OOD label. The long per-(guide,chromosome) table is "
            "provided for Mondrian/per-chromosome RQ3 analysis (within-guide records are dependent; "
            "use guide-level or Mondrian calibration there)."),
    }
    (DATA / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: summary[k] for k in (
        "n_guides_total", "split_sizes", "guide_overlap_across_splits", "y_eff",
        "off_id_in_distribution", "off_chr22_OOD", "off_id_percentiles",
        "tau_off_candidate_baseline_floor", "n_in_distribution_chromosomes",
        "calibration_guarantee_slack_1_over_n_plus_1")}, indent=2))
    print("[done] wrote data/{conformalgen_pool,offtarget_by_chromosome,ood_chr22,split_summary}.csv/json + data/splits/")


if __name__ == "__main__":
    main()
