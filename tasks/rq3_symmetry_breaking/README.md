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
is retained in training curves. The proposal's additional 18 image-input RQ3
runs are not implemented by this state matrix and remain outstanding.
