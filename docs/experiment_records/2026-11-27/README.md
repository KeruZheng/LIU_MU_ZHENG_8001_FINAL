# 2026-11-27: E4 continuation repair

RQ1/RQ2/RQ3 evidence remains frozen. E4 seed 0 corrected paired evaluation is
complete: DP D0/D1/D2 = 77/0/0 successes out of 100 per domain; EquiDiff =
85/0/0. All 300 initial scenes pair exactly between methods. These are
single-training-seed results, not cross-seed superiority evidence.

Original EquiDiff seed 1/2 jobs 29427/29428 reached the 72-hour time limit.
Continuation jobs 30475/30486 then failed before training because `squeue -j`
rejected retired source IDs. Original checkpoints and failure evidence remain.

The new terminal-state guard checks the live user queue and exact historical
accounting. Four regression tests passed. New immutable releases were built
from verified source manifests, keeping the scientific training/evaluation
configuration unchanged. Jobs **30759** (seed 1) and **30760** (seed 2) were
submitted on November 27 at approximately 12:38 CST and initially queued for
Priority. They have 72-hour allocations requested on A100/H800 partitions.

Read-only inspection of the actual latest checkpoints confirms seed 1 saved
epoch index 470 / 105504 completed updates (29 epochs remaining), and seed 2
saved epoch index 460 / 103264 completed updates (39 epochs remaining).
Training logs went beyond the last saves; that unsaved work will be repeated.

The planned sequence is EquiDiff resume to epoch 500, corrected v3 evaluation,
SHA256-verified local backup and remote duplicate cleanup, then DP training
and evaluation. Original partial checkpoints are retained. Source checkpoints
lack RNG/loader state, so restarted randomness is a documented limitation.

- [Recovery code and procedure](../../../operations/e4_recovery_20261127/README.md)
- Local live monitor: `tasks/rq1_transfer/results/e4_recovery_20261127/STATUS.md`
- Local paired seed 0 evidence:
  `tasks/rq1_transfer/results/e4_corrected_eval_20261126_v3/attempt2/paired_results.json`

Live result directories are local runtime artifacts, not promised GitHub assets.
