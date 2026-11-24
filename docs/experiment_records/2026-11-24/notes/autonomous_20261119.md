> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/autonomous_20261119.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 2026-11-19 storage guard and cluster recovery

## 21:39 CST live update

C8 recovery job **27812 is RUNNING on um-a100-01**, passed CUDA/write preflight,
and its fresh seed-0 run reached about 251/20000 steps. RQ3 27809 is evaluating
the 20k checkpoint of fixed-state goal seed 2 (88/100 episodes). Local C4
200-demo seed 1 is at about 19423/20000 steps. Both recovered cluster queues
are now active; old job 27692 remains stuck COMPLETING on the excluded node.

The initial archive completed at 21:36 CST with **86 verified checkpoints,
81 remote duplicates deleted, 37,745,031,913 bytes freed (35.15 GiB), no errors**.
Its final summary is `results/remote_archive/20261119_quota/summary.json`.
The continuous archive worker remains active for future completed runs.

## 21:15 CST live update

C8's damaged checkpoint has been copied and SHA-256 verified at both ends.
The failed 4000-step attempt was preserved under its quarantine directory.
The deferred recovery submitted **27812 / a100_batch**, currently PENDING with
reason Priority. It will perform CUDA/storage preflight before starting the
same frozen seed-0 configuration from scratch, followed by the remaining C8
matrix with per-run archive gating. Initial archive: 83/86 checkpoints verified,
about 22.26 GiB freed. The historical details below describe earlier snapshots.

## Scope and authorization

The user authorized automatic progress, verified local offloading, and deletion
of remote duplicates after backup. Preserve existing changes, active training,
and frozen training source. This update implements the storage/recovery workflow
for the existing 66-run matrix; it does not add the missing 18 RQ3 image runs or
official long-recipe checks to the completed scope.

## Confirmed failures

- RQ3 job 27692 failed during the 18k evaluation of
  `fixed_state_equidiff_goal_n50_s2`, after 19 training runs and 18 full evaluation
  sets. CUDA initialization reported error 804.
- `scontrol show node um-gpu03` reports
  `IDLE+COMPLETING+DRAIN+INVALID_REG`, with reason
  `gres/gpu count reported lower than configured (0 < 7)`.
  This is a node GPU registration failure; the underlying driver/hardware cause
  is not established. The old job is stuck COMPLETING; do not cancel other jobs
  or attempt node/driver repair.
- Recovery attempt 27808 on `um-gpu04` failed the small CUDA preflight before
  touching model checkpoints. `nvidia-smi` worked but PyTorch CUDA initialization
  reported an unknown error. This node is also excluded from RQ3 recovery.
- C8 job 27693 failed with `Disk quota exceeded`. Its first formal run reached
  about 4000 steps, but `last.pt` is an incomplete ZIP checkpoint. It cannot be
  resumed as-is. Preserve the failed attempt and restart that one run using the
  exact same frozen configuration and seed after its backup is verified.

## Recovery currently deployed

- RQ3 recovery job **27809**: submitted to `gpu_batch`, then the pending job was
  allowed to use `gpu_batch,a100_batch`, excluding `um-gpu03,um-gpu04`.
  It started on **um-a100-02 / A100-SXM4-40GB** around 21:00 CST. CUDA matrix
  multiplication and a group-owned storage write probe passed. It skipped the
  completed training and 16k evaluation and resumed the remaining evaluation.
- Changing GPU model for subsequent training/evaluation is a runtime difference.
  Existing runtime records and result tables retain device/version labels; do
  not present all RQ3 seeds as having used the same GPU.
- C8 recovery is automated by `notes/recover_c8_20261119.py`, PID initially
  **2967779**. It waits for SHA-256-verified backup of the damaged checkpoint,
  verifies the local copy and source hashes, then moves the failed run to
  `releases/c8_ablation_20261118_v1/quarantine/20261119_diskquota/` and submits the
  frozen six-run queue with archive gating. It records a submission attempt
  before calling `sbatch`; inspect the receipt before retrying ambiguous submits.
- C8 recovery state and eventual job ID are recorded in
  `development/proposal_v2/results/storage_guard_20261119/c8_recovery_status.json`
  and `development/proposal_v2/operations/storage_guard_20261119/c8_submission.json`.

## Continuous archive gate

Local and remote operational scripts are isolated from the training releases:

- Local: `development/proposal_v2/operations/storage_guard_20261119/`
- Remote: `${SICC_PROJECT}/operations/storage_guard_20261119/`
- Main helper: `storage_guard.py`; batch entry: `resume.sbatch`.
- Local archive worker PID initially **2963847**; actual PID is in
  `results/storage_guard_20261119/worker.pid`.

For each run, the driver invokes the original frozen supervisor with an
operational manifest containing that run's original four commands. Frozen
source/hash checks, seed, training budget, checkpoint selection, and evaluation
protocol are unchanged. After all three evaluations are complete, it writes an
archive request and **does not dispatch the next run until a verified archive
receipt exists and the four remote checkpoint files have been removed**.

The local worker copies run artifacts, fsyncs and hashes local files, then asks
the remote helper to compare all hashes while holding the run/supervisor locks.
It records a durable transaction before deleting only the four `.pt` files.
Completion markers, metrics, runtime metadata, configurations, and evaluation
reports remain on the cluster. Partial cleanup is resumable; checksum mismatch,
incomplete evaluation, or active locks prevent deletion.

New archives: `development/proposal_v2/results/verified_archive_20261119/`.
Earlier 18 complete RQ3 runs reference their checkpoints in the initial archive
instead of creating extra copies. All 18 archive receipts were established by
21:00 CST. The local worker polls every 30 seconds. If offloading fails, it logs
the error and retries; the training driver waits, preserving the model. The
driver's archive wait limit is 12 hours, after which it exits without dispatching
the next run.

Tests: **7 passed** in `notes/test_storage_guard_20261119.py`: successful cleanup
and metadata preservation, idempotent receipt, changed-file rejection,
incomplete-evaluation rejection, active supervisor lock protection, missing-file
proof requirements, partial-transaction recovery, and queue waiting behavior
(some checks share one test). Batch script syntax also passed.
The deferred C8 quarantine helper additionally passed 3 isolated filesystem
tests: matching-backup preservation and retry, checksum-mismatch rejection, and
completed-run rejection. Total new operational safety checks: 10 passed.

## Storage ownership and initial archive

- Corrected 624 remote project artifact ownership entries to `group_${CLUSTER_ACCOUNT}`
  and enabled setgid inheritance on 62 directories under project runs/releases.
  No model/source content changed. RQ3 78 frozen files and C8 58 frozen files
  retained their recorded SHA-256 hashes.
- The initial 39.18 GiB archive is still running under PID **2942022** at this
  update. It had verified 80 checkpoints and freed about 13.67 GiB by 20:57 CST.
  Latest counts and final summary are under
  `results/remote_archive/20261119_quota/`.
- Exact account/group quota remains unavailable through the exposed NFS tools.
  Official documentation gives 1 TB group `/data` and 200 GB account `/home`,
  but these are not measured remaining quotas. See `storage_archive_20261119.md`.

## Monitoring and constraints

The old read-only monitor was replaced, not a training process. New monitor PID
initially **2964835**, interval 120 seconds; actual PID is in the operations
directory's `monitor.pid`. It reads `jobs.json` to track recovery job IDs and
uses Slurm failure states to avoid displaying a dead queue as running. It keeps
the same `notes/live_status.md`, result mirrors, and scientific summary outputs.
Local C4 supervisor PID **2538616** continues unchanged.

Do not edit deployed operational scripts while their workers or drivers run.
Create another version if a source fix is needed. `jobs.json` is runtime data,
not frozen source. Keep all initial and new archive receipts for future restores.

## Next checks

1. Confirm job 27809 finishes its remaining evaluation and produces the first
   newly completed run/archive receipt under the live gate.
2. Confirm the initial damaged C8 checkpoint backup completes, quarantine is
   recorded, and the deferred C8 submission passes its CUDA preflight.
3. Check the initial archive final summary and freed-byte total.
4. Maintain the current 66-run experiment scope honestly; the additional 18 image
   RQ3 experiments, official sanity checks, and final presentation remain work.
