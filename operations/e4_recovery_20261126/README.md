# E4 continuation and independent evaluation recovery

This operation preserves the running November 23 training releases and their
environments. It adds two new continuation releases and an independently
versioned corrected evaluator. Main Push-T evidence is unchanged.

## Training continuation

`build_remote.py` verifies the original manifest and copies its sources and
scientific YAML to `e4_continue_s{1,2}_20261126_v1`. The new trainer restores
the latest completed epoch boundary after source jobs 29427/29428 terminate.
No signals are sent to those jobs. `submit_continuations.py` requests 72-hour
A100/H800 allocations with `afterany` dependencies. Per-account submission
limits are recorded as deferral, not successful submission. The persistent
`watch_recovery.py` retries once per minute, using receipts and job-name checks
to prevent duplicates. An accepted job may still wait for scheduling.

The old checkpoints store model weights, EMA weights, optimizer state, epoch
and global step. Periodic checkpoints are written before the two end-of-epoch
counter increments. `resume_support.py` validates that relationship against
224 batches/epoch, restores optimizer state, regenerates only ESCNN's eval-mode
caches, verifies every loaded tensor, and resumes at the next epoch. It restores
EMA's optimization counter instead of restarting its warmup. The original
500-epoch cosine schedule is reconstructed at the completed update count.

The old checkpoint does NOT retain RNG or loader-worker state. This is a
checkpoint continuation with restarted seeded randomness, not a claim of
bitwise equality to uninterrupted training. Work since the last periodic
checkpoint may be repeated. The final budget remains epoch 500 / 112000 updates.

`check_resume.py` checks actual checkpoint tensors, optimizer step counts,
EMA decay and learning rate. A local epoch-490 checkpoint passed, including
restoration to epoch 491 / 109984 updates with unchanged LR. Remote validation
job 30414 checks actual snapshots from both active lanes inside a GPU allocation,
then runs the corrected D2 regression. Validation snapshots are read-only hard
links to atomically saved checkpoints; they do not change the running files.

Job 30414 completed successfully with exit code 0:0. Seed 1's test snapshot
restored to epoch 441 / 98784 completed updates; seed 2's restored to epoch
431 / 96544 updates. These are validation snapshots, not the future cutoff
checkpoints: continuation selects the latest saved boundary after termination.
The deterministic toy test also matches uninterrupted model and EMA weights
exactly and rejects a partial-epoch checkpoint.

Each continuation trains EquiDiff to the fixed final checkpoint, runs corrected
evaluation, then waits for the existing SHA256-verified local backup/remote
cleanup gate before starting DP. DP then follows the same training/evaluation
and archive sequence. Original partial-run checkpoints remain available; they
are not deleted by this new lane. Checkpoints and inference weights produced by
the new completed run are archived and verified before remote copies are removed.

## Corrected evaluation v3

The v2 job stopped progressing during the upstream runner's final extra reset
after a complete 400-step D2 rollout, and eventually timed out. Its assertion
did not preserve the second scene record, so the differing field is unknown.
Instrumented reruns of seeds 200012--200015 passed on local RTX 5090 and SICC
RTX 2080 Ti (job 30413). These results do not establish the unique root cause
of the historical failure.

Version 3 addresses reset-history dependence and unbounded failure handling:

- Each seeded reset closes the previous environment and reconstructs a fresh
  environment using the same serialized task metadata, before a full seeded
  reset. Python and NumPy RNGs are both seeded. No simulator/system package is
  edited and no geometric assertion is relaxed.
- Before each model's formal evaluation, all 100 seeds in each domain are
  audited in two processes with different original constructor seeds. Model,
  state, image and low-dimensional observation hashes must agree. Seeds 200014
  and 200000 are repeated after intervening resets as well.
- Each formal rollout's initial scene must exactly match its independent audit.
  D1/D2 must contain 100 distinct peg positions, and the D2 audit checks rotations.
- The exact formerly failing four-episode D2 shard is repeated for 400 steps,
  including the final reset, as a regression gate. It is engineering evidence
  and does not replace a formal result shard.
- Mismatches save the second record before failing. Worker cleanup is bounded,
  and each isolated child process has a 20-minute timeout with process-group
  cleanup. A failed check stops dispatch instead of silently accepting a score.
- Scene and diffusion random streams remain separate. The final epoch-500 EMA,
  100 episodes/domain, 400 steps, absolute actions and fixed checkpoint choice
  are unchanged. New protocol ID: `seeded_fresh_scene_v3`.

All 300 episodes per model are evaluated anew. The 112 records from v2 are
preserved and are NOT pooled with v3. `run_pair.py` runs DP seed 0 followed by
EquiDiff seed 0, then requires exact scene pairing for all 300 initial conditions
before publishing `paired_results.json`. This remains a single-training-seed
comparison, not a multi-seed superiority claim.

## Runtime locations

Paths below are relative to the workspace containing `codes/`.

- Frozen local evaluator: `development/proposal_v2/releases/e4_corrected_eval_20261126_v3/`.
- Active local pair output: `codes/tasks/rq1_transfer/results/e4_corrected_eval_20261126_v3/attempt2/`.
- Recovery observer: `codes/tasks/rq1_transfer/results/e4_recovery_20261126/latest.json` and `STATUS.md`.
- Diagnostics and checkpoint checks: `development/proposal_v2/operations/e4_recovery_20261126/`.
- New remote releases: `robotic/releases/e4_continue_s{1,2}_20261126_v1/`.
- Verified archive worker: existing `operations/parallel_20261119/transfer_guard.py`.

The first local pair launcher resolved the venv Python symlink to its base
interpreter, causing an import failure before any formal episode. Its output is
preserved in the v3 result root. Attempt 2 uses the venv path without resolving
the interpreter symlink. The environment itself was not modified.

The existing November 23 dashboard still describes the original lanes. Use the
new recovery observer for continuation receipts and the independent local pair
status. `preflight` and quota deferral are not completed evaluation or running
continuation training. Check runtime evidence for current state.
