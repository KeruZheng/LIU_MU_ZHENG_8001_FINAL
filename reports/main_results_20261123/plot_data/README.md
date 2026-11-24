# Plotting Tables

All CSVs are exported from the frozen November 23 `evidence.json` and
`summary.json`. No training or live results are read. See the
[Chinese data map](../DATA_MAP_ZH.md) for figure-to-file mappings, local paths,
GitHub coverage, metric definitions, and replotting instructions.

Start with `rq1_rq2.csv`, `rq3_state.csv`, and `convergence.csv` for the three
existing figures. `rq3_image.csv` and `c8.csv` support additional figures.
`test_seeds.csv` supplies individual training-seed points. `episodes.csv`
contains all 34,200 checkpoint-episode rows; these are repeated measurements
from 114 unique training runs, not independent training samples.

Coverage and success rate are different metrics, both in [0,1]. `seed_std`
is sample standard deviation across three training seeds, not standard error.
Validation curves are independent of the primary test runs. E4 is excluded.
Use `run_id` to deduplicate shared RQ1/RQ2 and reused RQ3 image baselines.

`manifest.json` records input hashes, CSV hashes and row counts. Recreate with
`python3 reports/main_results_20261123/export_plot_data.py` from the repo root.
