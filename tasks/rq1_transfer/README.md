# RQ1: transfer and data efficiency

Question: does the EquiDiff adaptation transfer to planar Push-T, especially at
small demonstration counts? Compare DP and EquiDiff-C4 across 25/50/100/200 demos,
with three training seeds. The main matrix also includes DP + exact C4 augmentation
for the shared RQ2 comparison: 36 unique runs, not two independently trained tables.

- Scientific code/configurations/protocol:
  [C4 main release](../../experiments/releases/c4_main_20261118_v1/PROTOCOL.md).
- Independent convergence checks: four `conv_dp_n*_20261119_v1` releases,
  seeds 0/1/2, 20k/40k/60k updates, validation only. Their 20k results must not be
  pooled with the main matrix's 20k test results.
- Exploratory C4 convergence: `conv_c4_n50_20261119_v2` and
  `conv_c4_n200_20261119_v2`, one seed each, run independently at the same
  20k/40k/60k milestones. These two diagnostics do not establish three-seed
  convergence or extend the primary matrix.
- Raw main evidence: `results/c4_main_20261118_v1/runs/<run>/`.
- Raw convergence evidence: `results/conv_dp_n*_20261119_v1/runs/<run>/`.
- [Published result snapshot](reports/submission_20261119.json); local provenance:
  `results/run_index.json`; release membership: [task.json](task.json).

Primary score is the mean of the 16k/18k/20k test checkpoint scores, with final
20k also available. Coverage is intersection/goal area, not IoU. No partial run
should be presented as a finished three-checkpoint result.

Official DP Push-T reference seed 42 (3050 epochs, published 90-demo recipe) was
submitted as H800 job 28232 on November 20 at 21:13 CST, following successful
preflight 28225. It is a separate reference, not a primary-matrix run. Its source
is in `experiments/releases/official_dp_s42_20261120_v1` and operational details
are in [the reference runner notes](../../operations/official_reference_20261120/README.md).
Full reference results and the EquiDiff Stack-D1 check remain outstanding.
