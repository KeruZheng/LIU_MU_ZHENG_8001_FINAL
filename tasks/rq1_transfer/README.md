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
- Raw main evidence: `results/c4_main_20261118_v1/runs/<run>/`.
- Raw convergence evidence: `results/conv_dp_n*_20261119_v1/runs/<run>/`.
- [Published result snapshot](reports/submission_20261119.json); local provenance:
  `results/run_index.json`; release membership: [task.json](task.json).

Primary score is the mean of the 16k/18k/20k test checkpoint scores, with final
20k also available. Coverage is intersection/goal area, not IoU. No partial run
should be presented as a finished three-checkpoint result. Full official DP
Push-T and EquiDiff Stack-D1 reference checks remain outstanding.
