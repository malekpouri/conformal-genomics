#!/usr/bin/env python3
"""
ConformalGen — Phase 3: biological definitions & post-hoc CIRCLE-seq audit.

Quantifies the fidelity of the *surrogate* computational off-target oracle (genome-wide Hamming<=3
near-match count on GRCh38, SpCas9 NGG PAM, 20-nt protospacer, no indels) against experimentally
validated off-target cleavage, WITHOUT using CIRCLE-seq in training or calibration. Two independent
audits:
  (1) CIRCLE-seq whole dataset (labelled 0/1): ROC-AUC of a mismatch-based score for discriminating
      validated vs non-validated candidate sites, and sensitivity of the MM<=3 rule.
  (2) Tsai et al. curated validated off-targets: fraction of true off-targets captured by MM<=3
      (recall) and the incidence of bulges the mismatch-only proxy cannot represent.

Outputs results/json/phase3_biological_audit.json. CPU-only, streaming/vectorized, 0 GB VRAM.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT.parents[1]            # .../Human genomes Dataset/notebook
MM_MAX = 3
PROTO_LEN = 20


def _find(name):
    for cand in (NOTEBOOK / name, NOTEBOOK.parent / name, PROJECT / "data" / "raw" / name):
        if cand.exists():
            return cand
    hits = list(NOTEBOOK.parent.glob(name)) + list(NOTEBOOK.glob(name))
    return hits[0] if hits else None


def _proto_matrix(series):
    """First PROTO_LEN nt (5' protospacer) of each cleaned sequence -> (N, L) uint8 char matrix."""
    s = series.str.replace("_", "", regex=False).str.upper().str[:PROTO_LEN]
    arr = np.frombuffer("".join(s.tolist()).encode(), dtype=np.uint8).reshape(len(s), PROTO_LEN)
    return arr


def audit_circle_seq():
    path = _find("I_1_CIRCLE_seq_10gRNA_wholeDataset.csv")
    if path is None:
        return {"status": "SKIPPED", "reason": "CIRCLE-seq dataset not found"}
    d = pd.read_csv(path, usecols=["sgRNA_seq", "off_seq", "label"])
    d = d[(d["sgRNA_seq"].str.replace("_", "", regex=False).str.len() >= PROTO_LEN) &
          (d["off_seq"].str.len() >= PROTO_LEN)].reset_index(drop=True)
    G = _proto_matrix(d["sgRNA_seq"]); O = _proto_matrix(d["off_seq"])
    mm = (G != O).sum(axis=1).astype(int)                       # protospacer mismatch count
    label = (d["label"].to_numpy() > 0).astype(int)
    score = -mm.astype(float)                                    # fewer mismatches => more likely real

    auc = float(roc_auc_score(label, score))
    within = mm <= MM_MAX
    pos, neg = label == 1, label == 0
    sens = float(np.mean(within[pos]))                           # recall of MM<=3 among validated
    spec = float(np.mean(~within[neg]))
    prec = float(np.mean(label[within])) if within.any() else float("nan")
    # per-guide averaged AUC (robust to guide imbalance)
    guide = d["sgRNA_seq"].str.replace("_", "", regex=False).to_numpy()
    aucs = []
    for gid in np.unique(guide):
        m = guide == gid
        if 0 < label[m].sum() < m.sum():
            aucs.append(roc_auc_score(label[m], score[m]))
    dist = {f"mm={k}": int(np.sum((mm == k) & pos)) for k in range(0, 7)}
    dist["mm>=7|validated"] = int(np.sum((mm >= 7) & pos))
    return {
        "status": "OK", "source": path.name, "n_pairs": int(len(d)),
        "n_validated": int(pos.sum()), "n_nonvalidated": int(neg.sum()),
        "n_guides": int(len(np.unique(guide))),
        "roc_auc_mismatch_score": round(auc, 3),
        "per_guide_mean_auc": round(float(np.mean(aucs)), 3) if aucs else None,
        "per_guide_auc_n": len(aucs),
        "mean_mm_validated": round(float(mm[pos].mean()), 3),
        "mean_mm_nonvalidated": round(float(mm[neg].mean()), 3),
        f"sensitivity_at_MM<={MM_MAX}": round(sens, 3),
        f"specificity_at_MM<={MM_MAX}": round(spec, 3),
        f"precision_at_MM<={MM_MAX}": round(prec, 4),
        "validated_mm_distribution": dist,
    }


def audit_tsai():
    path = _find("tsai_validated_offtargets.tsv")
    if path is None:
        return {"status": "SKIPPED", "reason": "Tsai TSV not found"}
    d = pd.read_csv(path, sep="\t")
    mm = pd.to_numeric(d["mismatches"], errors="coerce").dropna().astype(int)
    n = int(len(mm))
    within = float(np.mean(mm <= MM_MAX))
    has_bulge = None
    if {"bulgeRnaMmCount", "bulgeDnaMmCount"}.issubset(d.columns):
        b = (pd.to_numeric(d["bulgeRnaMmCount"], errors="coerce").fillna(-1) > 0) | \
            (pd.to_numeric(d["bulgeDnaMmCount"], errors="coerce").fillna(-1) > 0)
        has_bulge = round(float(b.mean()), 4)
    return {
        "status": "OK", "source": path.name, "n_validated_offtargets": n,
        "mean_mismatches": round(float(mm.mean()), 3),
        "median_mismatches": int(mm.median()),
        f"fraction_captured_at_MM<={MM_MAX}": round(within, 3),
        "mismatch_distribution": {f"mm={k}": int((mm == k).sum()) for k in range(0, 8)},
        "fraction_with_bulge": has_bulge,
    }


def main():
    circle = audit_circle_seq()
    tsai = audit_tsai()
    out = {
        "off_target_definition": {
            "genome_build": "GRCh38 (GCA_000001405.15, no-alt analysis set)",
            "nuclease": "SpCas9", "PAM": "NGG (3' of protospacer)",
            "protospacer_length_nt": PROTO_LEN, "max_mismatch_hamming": MM_MAX,
            "indels": "none (substitutions only; bulges not modelled)",
            "aggregation": "genome-wide near-match COUNT over chromosomes != chr22, excl. chrM/chrY",
            "surrogate_note": ("this MM<=3 count is a SURROGATE oracle; ConformalGen coverage is w.r.t. "
                               "this oracle label, not experimentally validated cleavage. This audit "
                               "quantifies surrogate fidelity; CIRCLE-seq is NOT used in train/calibration."),
        },
        "circle_seq_audit": circle,
        "tsai_validated_audit": tsai,
    }
    (PROJECT / "results" / "json").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results" / "json" / "phase3_biological_audit.json").write_text(json.dumps(out, indent=2))

    print("OFF-TARGET DEFINITION: GRCh38 | SpCas9 NGG | 20-nt protospacer | Hamming<=3 | no indels")
    if circle["status"] == "OK":
        print(f"\nCIRCLE-seq audit ({circle['n_pairs']:,} pairs, {circle['n_validated']:,} validated, "
              f"{circle['n_guides']} guides):")
        print(f"  ROC-AUC (mismatch score)       = {circle['roc_auc_mismatch_score']}")
        print(f"  per-guide mean AUC             = {circle['per_guide_mean_auc']} (n={circle['per_guide_auc_n']})")
        print(f"  mean mismatches  val/nonval    = {circle['mean_mm_validated']} / {circle['mean_mm_nonvalidated']}")
        print(f"  sensitivity @ MM<=3            = {circle['sensitivity_at_MM<=3']}")
        print(f"  specificity @ MM<=3            = {circle['specificity_at_MM<=3']}")
        print(f"  precision   @ MM<=3            = {circle['precision_at_MM<=3']}")
    if tsai["status"] == "OK":
        print(f"\nTsai validated off-targets ({tsai['n_validated_offtargets']}):")
        print(f"  mean/median mismatches         = {tsai['mean_mismatches']} / {tsai['median_mismatches']}")
        print(f"  fraction captured @ MM<=3      = {tsai['fraction_captured_at_MM<=3']}")
        print(f"  fraction with bulge (unmodelled)= {tsai['fraction_with_bulge']}")
    print("\n[done] -> results/json/phase3_biological_audit.json")


if __name__ == "__main__":
    main()
