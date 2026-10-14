# Note on these result files

`results/json/` and `remediation/results/` are machine-readable outputs of the pipeline stages.
**They are not all current.** `remediation/results/phase1.json`, `phase2.json` and
`phase2_burdens.csv` were regenerated after the grouping defect described in
[`../CORRECTION.md`](../CORRECTION.md) was fixed; the remaining files are the outputs of stages
that have not been re-run and still contain figures computed under the defective grouping key —
notably `external_replication.json` (`roc_auc` 0.925, `burden_rho` 0.944, `power_q0.10` 0.584,
`power_q0.20` 0.796) and `deployment_benchmarks.json`.

The authoritative corrected numbers are in [`../audit/results/`](../audit/results/), produced by
the scripts in [`../audit/`](../audit/). Where the two disagree, `audit/results/` is correct.
