# E4 continuation after retired Slurm job IDs

The November 26 continuation jobs 30475 and 30486 failed before training:
`squeue -j` returned an error after source jobs 29427/29428 left the live
queue. Both source jobs are recorded as TIMEOUT; their checkpoints survive.

This revision creates independent `e4_continue_s{1,2}_20261127_v1` releases.
It queries all live jobs for the user, then requires an exact terminal `sacct`
record for the source ID. Missing accounting, active jobs and scheduler failures
stop dispatch. Four regression tests cover these cases. Submission and runtime
both use the guard. Retired source IDs are not used as new Slurm dependencies.

The builder copies unchanged training, resume, export and v3 evaluation code
from `../e4_recovery_20261126/`. Original scientific configurations and source
manifests are verified. No old release or environment is modified. The new
releases are registered with the existing verified backup worker.

Submitted jobs: seed 1 = 30759; seed 2 = 30760. Both initially waited for
Priority on A100/H800 partitions. Submission is not evidence of training.
Each lane resumes EquiDiff, evaluates all three domains with v3, waits for
verified local archival, then trains/evaluates/archives DP. Epoch 500 remains
the final budget. RNG/loader states were absent from the source checkpoints,
so continuation is not bitwise identical to uninterrupted training.

Current monitor: `../../tasks/rq1_transfer/results/e4_recovery_20261127/`.
The November 26 observer remains a historical view of failed jobs and must not
be used as the current continuation dashboard. The new observer runs a frozen
copy outside this Git repository and checks once per minute. Receipts prevent
duplicate submission; a runtime failure is reported, not blindly resubmitted.

Validation: `python3 -m unittest discover -s operations/e4_recovery_20261127
-p 'test_*.py'` (run as one line from the repository root).
