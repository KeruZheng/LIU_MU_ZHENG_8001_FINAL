# RQ2: equivariance versus exact augmentation

Question: once physically exact C4 augmentation is available, does the
equivariant architecture still improve performance? Use the same frozen
36-run main matrix as RQ1; compare DP, DP + C4 augmentation, and EquiDiff-C4.

- [Shared C4 scientific protocol](../../experiments/releases/c4_main_20261118_v1/PROTOCOL.md).
- [C8 architecture ablation](../../experiments/releases/c8_ablation_20261118_v1/PROTOCOL.md):
  50/200 demos times three seeds, six additional runs.
- Shared main raw evidence stays at
  `../rq1_transfer/results/c4_main_20261118_v1/runs/`.
- C8 raw evidence is owned here: `results/c8_ablation_20261118_v1/runs/`.
- [Published result snapshot](reports/submission_20261119.json); local provenance:
  `results/run_index.json`; release membership: [task.json](task.json).

C8 is an architectural ablation on the validated C4 environment. A 45-degree
rotation is not an exact symmetry of square walls, rasterized images, or the
DDPM component-wise clipping rule. Model size and GPU/runtime differences are
recorded and must be considered when interpreting C4/C8 comparisons.

The original disk-quota-failure C8 attempt is quarantined. Its historical backup
is excluded from the current checkpoint views and is not a usable trained model.
