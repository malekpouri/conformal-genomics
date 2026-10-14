#!/usr/bin/env python3
"""
Referee audit A4 — close the generate -> align -> score -> certify loop, and explain analytically
why the published loop accepted nothing (roadmap R7, M3, M2).

The defect: deployment_benchmarks.json reports cfbh_accept_rate_q0.10 = 0.0 over 2,000 generated
candidates, with the honest note that "Exhaustive Cas-OFFinder on hg38 for novel guides is not run
(no binary)".  The paper reads this as the oracle being uninformative and cfBH failing safe.  That
explanation is incomplete: with a small calibration cohort the selection is EMPTY BY CONSTRUCTION,
whatever the oracle does.

  conformal p-values are bounded below by 1/(n_cal+1);
  BH at level q over m candidates rejects the k-th smallest only if p_(k) <= q*k/m;
  so a non-empty selection requires  1/(n_cal+1) <= q/m * k  for some k <= m, i.e. at least
      k_min = ceil( m / (q * (n_cal + 1)) )
  candidates must attain the minimal p-value -- and if k_min > m the accepted set is necessarily
  empty no matter how good the predictor is.  For the published setup (n_cal = 80 pseudo-guides,
  m = 2000, q = 0.10) k_min = 248; for the honest cohort (n_cal = 10 real guides) k_min = 1819.

This script (1) trains the same GRU generator, (2) runs a REAL off-target enumeration for the
generated guides against GRCh38 chr22 (no external binary: PAM scan + bit-parallel mismatch count),
(3) scores every enumerated site with a CIRCLE-trained per-site classifier and sums to an aligned
burden, (4) runs the certification step, and (5) reports the feasibility bound against the measured
accept rate.  The alignment parameters are stated explicitly (roadmap M3).

CPU-only.  Output: audit/results/a4_generator_loop.json
"""
from __future__ import annotations
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DS = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from src.selection import conformal_pvalues, bh_select                       # noqa: E402
sys.path.insert(0, str(PROJECT / "remediation"))

PROTO = 20
CHROM = "chr22"
MAX_MM = 3                      # Cas-OFFinder-equivalent mismatch budget, no bulges (see M2)
N_GEN = 2000
QS = (0.10, 0.20)
SEED = 0
NUC = "ACGT"
_CODE = np.full(256, 255, np.uint8)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _i
    _CODE[ord(_c.lower())] = _i

ALIGNMENT_SPEC = {
    "genome_build": "GRCh38/hg38 (hg38.fa.gz, UCSC)",
    "chromosome_scanned": CHROM,
    "pam": "NGG, both strands (forward: 5'-protospacer(20)+NGG-3'; reverse: revcomp of CCN+20)",
    "mismatch_budget": f"<= {MAX_MM} over the 20-nt protospacer",
    "bulges": "none -- DNA/RNA bulges are not enumerated (mismatch-only regime)",
    "orientation": "both strands scanned independently",
    "ambiguous_bases": "windows containing any non-ACGT base are discarded",
    "on_target_self_hit": "the 0-mismatch self hit is retained (generated guides are novel, so it "
                          "is not guaranteed to exist)",
    "method": "exhaustive PAM scan + 2-bit packing + bit-parallel Hamming distance "
              "(np.bitwise_count); equivalent to a Cas-OFFinder mm<=3 search restricted to "
              f"{CHROM}",
}


# ------------------------------------------------------------------ genome
def read_chrom(fa_gz: Path, name: str) -> np.ndarray:
    """Stream a gzipped FASTA and return one chromosome as a uint8 code array (255 = ambiguous)."""
    chunks, inside = [], False
    with gzip.open(fa_gz, "rt") as f:
        for line in f:
            if line[0] == ">":
                if inside:
                    break
                inside = line[1:].split()[0] == name
                continue
            if inside:
                chunks.append(line.strip())
    seq = "".join(chunks)
    return _CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]


def windows(codes: np.ndarray) -> np.ndarray:
    """All 20-mers as a (n, 20) view via stride tricks."""
    return np.lib.stride_tricks.sliding_window_view(codes, PROTO)


def pack(mat: np.ndarray) -> np.ndarray:
    """Pack (n, 20) 2-bit codes into uint64."""
    out = np.zeros(mat.shape[0], np.uint64)
    for i in range(PROTO):
        out |= (mat[:, i].astype(np.uint64) & np.uint64(3)) << np.uint64(2 * i)
    return out


_M55 = np.uint64(0x5555555555555555)


def mismatches(sites: np.ndarray, guide: np.uint64) -> np.ndarray:
    x = sites ^ guide
    return np.bitwise_count((x | (x >> np.uint64(1))) & _M55).astype(np.uint8)


def enumerate_sites(codes: np.ndarray):
    """Return (packed_sites, site_code_matrix) for every NGG-adjacent 20-mer on both strands."""
    n = len(codes)
    # forward: protospacer [i, i+20), PAM = codes[i+21], codes[i+22] == G,G  (N at i+20)
    g = codes == 2                                                   # 'G'
    idx = np.nonzero(g[21:n - 1] & g[22:n])[0]                       # start of protospacer
    W = windows(codes)
    fwd = W[idx]
    # reverse: PAM CCN at [j, j+3) -> protospacer = revcomp(codes[j+3 : j+23])
    c = codes == 1                                                   # 'C'
    jdx = np.nonzero(c[:n - 23] & c[1:n - 22])[0]
    rev = W[jdx + 3][:, ::-1]
    rev = np.where(rev < 4, 3 - rev, rev)                            # complement (A<->T, C<->G)
    sites = np.concatenate([fwd, rev], 0)
    ok = (sites < 4).all(1)
    sites = sites[ok]
    return pack(sites), sites


# ------------------------------------------------------------------ generator (same as phase3)
def build_generator(spacers):
    from phase3_generator_multiobjective import GRUGen, encode, train_generator, generate  # noqa
    model = train_generator(spacers, epochs=25)
    return generate(model, N_GEN)


# ------------------------------------------------------------------ per-site oracle
def onehot_codes(mat):
    N = mat.shape[0]
    oh = np.zeros((N, PROTO, 4), np.float32)
    r, c = np.nonzero(mat < 4)
    oh[r, c, mat[r, c]] = 1.0
    return oh.reshape(N, PROTO * 4)


def train_site_oracle():
    """CIRCLE-trained per-site cleavage classifier on the MISMATCH-ONLY rows.

    Encoding of this benchmark file (established by referee audit M2):
      * a leading "-" is an alignment PAD, not a bulge (it is always at index 0);
      * "_" is the bulge marker, and it appears mid-sequence;
      * the file therefore mixes two alignment hypotheses for every guide -- 78,450 rows with an
        RNA bulge in the guide and 506,499 rows without, of which 174,429 carry a DNA bulge in the
        off-target.  Together with the 4 PAM variants these are the 8 "pseudo-guide" representations
        behind the R1 grouping defect.
    The deployable regime here is a mismatch-only genome scan (Cas-OFFinder mm<=3, no bulges), so
    the oracle is trained on rows where NEITHER sequence carries a "_".
    """
    path = sorted(DS.glob("I_1_CIRCLE_seq*csv"))[0]
    df = pd.read_csv(path, usecols=["sgRNA_seq", "off_seq", "label", "sgRNA_type"])
    g_raw = df["sgRNA_seq"].str.lstrip("-")
    o_raw = df["off_seq"].str.lstrip("-")
    keep = (~g_raw.str.contains("_")) & (~o_raw.str.contains("_")) & \
           (g_raw.str.len() >= PROTO) & (o_raw.str.len() >= PROTO)
    df = df[keep].assign(sgRNA_seq=g_raw[keep], off_seq=o_raw[keep]).reset_index(drop=True)
    G = _CODE[np.frombuffer("".join(df["sgRNA_seq"].str.replace("_", "", regex=False)
                                    .str.upper().str[:PROTO].tolist()).encode(), np.uint8)
              ].reshape(len(df), PROTO)
    O = _CODE[np.frombuffer("".join(df["off_seq"].str.upper().str[:PROTO].tolist()).encode(),
                            np.uint8)].reshape(len(df), PROTO)
    mism = (G != O).astype(np.float32); mm = mism.sum(1, keepdims=True)
    X = np.concatenate([mism, onehot_codes(G), onehot_codes(O), mm], 1).astype(np.float32)
    y = (df["label"].to_numpy() > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                         min_samples_leaf=50, random_state=42)
    clf.fit(X, y)
    return clf, int(len(df)), int(y.sum()), df


def aligned_burden(clf, guides_codes, packed_sites, site_codes, tag=""):
    """For each guide: enumerate mm<=MAX_MM sites on the scanned chromosome and sum P(cleavage)."""
    burdens, n_hits = [], []
    gp = pack(guides_codes)
    for i in range(len(gp)):
        mm = mismatches(packed_sites, gp[i])
        hit = np.nonzero(mm <= MAX_MM)[0]
        n_hits.append(len(hit))
        if len(hit) == 0:
            burdens.append(0.0); continue
        S = site_codes[hit]
        Gm = np.repeat(guides_codes[i][None, :], len(hit), 0)
        mism = (Gm != S).astype(np.float32)
        X = np.concatenate([mism, onehot_codes(Gm), onehot_codes(S),
                            mism.sum(1, keepdims=True)], 1).astype(np.float32)
        burdens.append(float(clf.predict_proba(X)[:, 1].sum()))
        if tag and (i + 1) % 250 == 0:
            print(f"     {tag}: {i+1}/{len(gp)} guides aligned")
    return np.array(burdens), np.array(n_hits)


def feasibility(n_cal, m, q):
    k_min = int(np.ceil(m / (q * (n_cal + 1))))
    return {"n_cal": int(n_cal), "m_candidates": int(m), "q": q,
            "min_attainable_pvalue": round(1.0 / (n_cal + 1), 6),
            "k_min_candidates_at_minimal_p_for_any_selection": k_min,
            "selection_possible_in_principle": bool(k_min <= m),
            "n_cal_required_to_select_one": int(np.ceil(m / q - 1))}


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED); np.random.seed(SEED)

    # --- generator (same corpus and architecture as remediation phase 3) ---
    hnn = sorted((DS / "Extra_Metadata" / "CRISPR_HNN").glob("WT*"))[0]
    wt = pd.read_csv(hnn); wt.columns = ["sgRNA", "indel"]
    wt = wt[wt["sgRNA"].str.len() >= PROTO]
    spacers = wt["sgRNA"].str[:PROTO].tolist()
    print(f"[a4] training GRU generator on {len(spacers):,} WT spacers ...")
    gen = build_generator(spacers)
    gen = [g for g in gen if set(g) <= set(NUC)][:N_GEN]
    print(f"[a4] generated {len(gen):,} candidates")

    # --- genome enumeration ---
    fa = DS / "hg38.fa.gz"
    print(f"[a4] streaming {CHROM} from {fa.name} ...")
    codes = read_chrom(fa, CHROM)
    print(f"[a4] {CHROM}: {len(codes):,} bases ({np.mean(codes < 4):.1%} ACGT)")
    packed, site_codes = enumerate_sites(codes)
    print(f"[a4] enumerated {len(packed):,} NGG-adjacent protospacers (both strands)")

    clf, n_pairs, n_pos, circ = train_site_oracle()
    print(f"[a4] per-site oracle trained on {n_pairs:,} mismatch-only CIRCLE pairs "
          f"({n_pos:,} validated)")

    gen_codes = _CODE[np.frombuffer("".join(gen).encode(), np.uint8)].reshape(len(gen), PROTO)
    print(f"[a4] aligning {len(gen):,} generated guides ...")
    gen_burden, gen_hits = aligned_burden(clf, gen_codes, packed, site_codes, tag="generated")

    # the 10 real CIRCLE guides, aligned and scored identically -> same scale as the candidates
    real_g = (circ["sgRNA_type"].str.strip().str.replace("N", "", regex=False)
              .str.upper().str[:PROTO].unique())
    real_codes = _CODE[np.frombuffer("".join(real_g.tolist()).encode(), np.uint8)
                       ].reshape(len(real_g), PROTO)
    real_burden, real_hits = aligned_burden(clf, real_codes, packed, site_codes)
    validated = (circ.assign(r=circ["sgRNA_type"].str.strip())
                 .groupby("r")["label"].apply(lambda s: float((s > 0).sum()))
                 .reindex([f"{g}NGG" for g in real_g]).to_numpy())

    # --- certification step: calibrate on the 10 validated guides, certify the generated pool ---
    s = float(np.log1p(validated).std()) or 1.0
    certif = {}
    for tau in (100.0, 250.0):
        ztau = np.log1p(tau)
        T_true_cal = (ztau - np.log1p(validated)) / s
        T_hat_cal = (ztau - np.log1p(np.maximum(real_burden, 0))) / s
        T_hat_gen = (ztau - np.log1p(np.maximum(gen_burden, 0))) / s
        V_cal = T_true_cal - T_hat_cal
        p = conformal_pvalues(V_cal, -T_hat_gen)
        row = {"tau_absolute_validated_sites": tau,
               "n_cal": int(len(V_cal)), "m_candidates": int(len(p)),
               "min_pvalue_observed": round(float(p.min()), 6),
               "n_candidates_at_minimal_p": int((p <= 1.0 / (len(V_cal) + 1) + 1e-12).sum())}
        for q in QS:
            sel = bh_select(p, q)
            row[f"q={q}"] = {"n_selected": int(sel.sum()),
                             "accept_rate": round(float(sel.mean()), 4),
                             **feasibility(len(V_cal), len(p), q)}
        row["scale_diagnostic"] = {
            "cal_validated_burden_genomewide_median": float(np.median(validated)),
            "cal_aligned_burden_chr22_median": round(float(np.median(real_burden)), 4),
            "candidate_aligned_burden_chr22_median": round(float(np.median(gen_burden)), 4),
            "footprint_mismatch": ("calibration TRUTH is genome-wide validated cleavage while both "
                                   "T_hat values are chr22-restricted predictions; the public file "
                                   "carries no site coordinates, so the truth cannot be restricted "
                                   "to the scanned footprint"),
            "consequence": (
                f"observed minimum p-value {float(p.min()):.3f} vs the attainable minimum "
                f"{1.0/(len(V_cal)+1):.3f}; {int((p <= 1.0/(len(V_cal)+1) + 1e-12).sum())} of "
                f"{len(p)} candidates attain it, against the k_min required for any selection. "
                "The two failure modes of this loop are symmetric: too few calibration guides "
                "relative to the pool size makes the accepted set empty, while a calibration "
                "cohort whose truth is measured on a different footprint than the candidates' "
                "scores can make it everything. Both are exchangeability failures, not oracle "
                "failures.")}
        certif[f"tau={int(tau)}"] = row

    out = {
        "alignment_spec": ALIGNMENT_SPEC,
        "generator": {"corpus": "CRISPR_HNN WT spacers", "n_corpus": int(len(spacers)),
                      "n_generated": int(len(gen)),
                      "novel_frac": round(float(np.mean([g not in set(spacers) for g in gen])), 3)},
        "chromosome": {"name": CHROM, "n_bases": int(len(codes)),
                       "n_pam_sites_enumerated": int(len(packed))},
        "site_oracle": {"training_pairs_mismatch_only": n_pairs, "validated_positives": n_pos,
                        "note": "bulge-annotated rows are excluded rather than silently stripped"},
        "generated_guides_aligned": {
            "mm_le_3_sites_per_guide": {
                "median": float(np.median(gen_hits)), "min": int(gen_hits.min()),
                "max": int(gen_hits.max())},
            "aligned_burden_chr22": {
                "median": round(float(np.median(gen_burden)), 4),
                "p90": round(float(np.percentile(gen_burden, 90)), 4),
                "max": round(float(gen_burden.max()), 4)},
            "note": f"burden is restricted to {CHROM}; a full-genome figure is ~50x larger and "
                    "requires the same scan over all chromosomes"},
        "circle_cohort_aligned_same_scale": {
            "n_guides": int(len(real_g)),
            "mm_le_3_sites_per_guide_median": float(np.median(real_hits)),
            "aligned_burden_median": round(float(np.median(real_burden)), 4),
            "validated_burden_median": float(np.median(validated))},
        "certification": certif,
        "feasibility_bound": {
            "rule": "a non-empty BH selection requires ceil(m/(q*(n_cal+1))) candidates at the "
                    "minimal conformal p-value 1/(n_cal+1)",
            "published_setup_n_cal80_m2000_q0.10": feasibility(80, 2000, 0.10),
            "honest_cohort_n_cal10_m2000_q0.10": feasibility(10, 2000, 0.10),
            "shortlist_m10_q0.10": feasibility(10, 10, 0.10),
            "shortlist_m50_q0.10": feasibility(10, 50, 0.10),
        },
        "verdict": (
            "The loop now runs end to end on real alignment output: generated guides are enumerated "
            f"against {CHROM} under an explicit mm<=3/NGG spec and scored per site by the "
            "CIRCLE-trained oracle. The published accept rate of 0.0 is nevertheless not an oracle "
            "property: with a calibration cohort of 10 real guides (or even 80 pseudo-guides), BH "
            "over a 2,000-candidate pool cannot reject at q=0.10 unless a large fraction of the "
            "pool attains the minimal conformal p-value. Certified selection from generative pools "
            "requires n_cal >= m/q - 1 validated calibration guides, so the deployable regime is a "
            "SHORT candidate shortlist, not a large generated pool. Conversely, when the "
            "calibration cohort's truth and the candidates' scores are measured on different "
            "footprints (genome-wide validated cleavage vs a chr22-restricted scan), every "
            "candidate attains the minimal p-value and BH accepts the entire pool: the certificate "
            "is only as meaningful as the exchangeability of the calibration and candidate pairs, "
            "and a deployable loop therefore needs validated truth WITH SITE COORDINATES so that "
            "both sides share one measurement footprint."),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "a4_generator_loop.json").write_text(json.dumps(out, indent=2))
    print("\n[a4] feasibility: n_cal=10, m=2000, q=0.10 -> need "
          f"{out['feasibility_bound']['honest_cohort_n_cal10_m2000_q0.10']['k_min_candidates_at_minimal_p_for_any_selection']}"
          " candidates at the minimal p-value")
    for k, v in certif.items():
        print(f"     {k}: selected {v['q=0.1']['n_selected']}/{v['m_candidates']} at q=0.10 "
              f"(min p {v['min_pvalue_observed']}, {v['n_candidates_at_minimal_p']} at minimum)")
    print("[a4] -> audit/results/a4_generator_loop.json")


if __name__ == "__main__":
    main()
