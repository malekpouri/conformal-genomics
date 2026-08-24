#!/usr/bin/env bash
# =====================================================================
# ConformalGen — one-line end-to-end reproduction driver.
# Runs Stages 01-06 sequentially and the full test suite.
# Deterministic; CPU-only (0 GB VRAM; GPU optional, <= 6 GB).
#
#   ./reproduce.sh            # full pipeline + tests
#   PY=python3 ./reproduce.sh # override the Python interpreter
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
echo ">> Python: $($PY --version 2>&1)  |  interpreter: $PY"

run() { echo; echo "========== $1 =========="; shift; "$@"; }

# --- Stage 01-06: data -> predictors -> calibration -> guided gen -> RQ -> figures
# Stage 01 rebuilds data/ from the raw genomic sources, which are large and NOT redistributed.
# The prepared data/ is committed, so a fresh clone reproduces from Stage 02 onward. Stage 01 runs
# only when its raw inputs are present next to the parent notebook directory.
if ls ../output_result_final*crispron.csv >/dev/null 2>&1 && [ -f ../report/whole_genome_hits_raw.csv ]; then
  run "01  Prepare data & guide-level splits"    "$PY" scripts/01_prepare_data.py
else
  echo; echo "========== 01  Prepare data (SKIPPED) =========="
  echo "raw genomic sources not found — using the committed data/ (splits, pool, per-chromosome table)."
fi
run "02  Fit predictors & non-conformity scores" "$PY" scripts/02_fit_predictors.py
run "03  Calibrate & test coverage"              "$PY" scripts/03_calibrate_and_test.py
run "04  Conformal-guided generation"            "$PY" scripts/04_run_guided_generation.py
run "05  RQ1/RQ2/RQ3 benchmark suite"            "$PY" scripts/05_run_rq_benchmarks.py
run "06  Publication figures (300 DPI)"          "$PY" scripts/06_plot_figures.py
run "07  Comprehensive ablation + FAR control"   "$PY" scripts/07_ablation.py
# Stage 08 audits CIRCLE-seq / Tsai (large, non-redistributed). Runs when those files are present;
# otherwise the committed results/json/phase3_biological_audit.json is kept.
have_bio=""
for p in ../../I_1_CIRCLE_seq*.csv ../I_1_CIRCLE_seq*.csv ../../tsai_validated_offtargets.tsv ../tsai_validated_offtargets.tsv; do
  ls $p >/dev/null 2>&1 && have_bio=1
done
if [ -n "$have_bio" ]; then
  run "08  Biological audit (CIRCLE-seq / Tsai)"  "$PY" scripts/08_biological_audit.py
else
  echo; echo "========== 08  Biological audit (SKIPPED) =========="
  echo "CIRCLE-seq / Tsai sources not found — keeping committed phase3_biological_audit.json."
fi

run "09  cfBH selection sensitivity"             "$PY" scripts/09_selection_sensitivity.py

# --- Test suite (data, scores, conformal, guided gen, benchmarks, selection, density-ratio) ---
echo; echo "========== TESTS =========="
for t in test_data_splits test_scores test_conformal_synthetic \
         test_guided_generation test_benchmark_repro test_selection test_density_ratio; do
  "$PY" "tests/${t}.py"
done

echo; echo ">> DONE. Metrics in results/json/, figures in figures/."
