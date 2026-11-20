# RQ3: incorrect symmetry and fixes

Question: how does a fixed, unobserved goal affect equivariant policies, and do
explicit goal input or a learnable symmetry-breaking gate recover performance?

- [Frozen state protocol](../../experiments/releases/rq3_state_20261118_v1/PROTOCOL.md).
- Fixed/rotated goal times four methods times three seeds: 24 runs, 50 demos.
- Methods: state DP, state EquiDiff, EquiDiff + goal, EquiDiff + gate.
- Raw evidence: `results/rq3_state_20261118_v1/runs/<run>/`.
- [Published result snapshot](reports/submission_20261119.json); local provenance:
  `results/run_index.json`; release membership: [task.json](task.json).

The state study is separate from the image-policy matrix. Ordinary state input
does not contain goal metadata. The gate starts at zero and its learned value
is retained in training curves.

The November 20 [image extension](../../experiments/releases/rq3_image_fixed_20261120_v1/PROTOCOL.md)
adds fixed/rotated goals x {DP, C4, C4+gate} x three seeds at 50 demos. There are
18 comparison cells: six completed rotated DP/C4 baselines are explicitly reused
from RQ1, and 12 new trainings are submitted as jobs 28269 and 28270. This revises
the proposal's 18-additional-run budget to avoid duplicate work. The global shared
run IDs are in task.json. These image inputs show the target, so the state study's
hidden-goal hypothesis cannot be applied unchanged. GPU preflight gates formal
training; submitted does not mean running or complete. New raw evidence lives in
`results/rq3_image_fixed_20261120_v1/` and `results/rq3_image_rotated_20261120_v1/`.
