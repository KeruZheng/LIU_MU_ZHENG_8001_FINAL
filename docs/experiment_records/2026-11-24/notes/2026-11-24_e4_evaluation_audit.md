> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/2026-11-24_e4_evaluation_audit.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# E4 D1/D2 evaluation audit, 2026-11-24

## Finding

The current E4 runner inherits upstream `RobomimicImageRunner.env_fn`, which
sets `robomimic_env.env.hard_reset = False`. In pinned MimicGen Square_D1/D2,
peg position (and D2 peg rotation) is sampled in `_load_model`, not in the
ordinary soft reset. Test seeds are set only after environment creation.
Therefore the 100 test episodes do not resample the full intended target
distribution: nuts are resampled, but each vector worker retains its peg.
The same episode seed alone does not guarantee the same complete scene across
methods when constructor randomness is not paired.

Confirmed using a CPU-only, rendering-disabled diagnostic in the project
simulation environment, with seeds 200000, 200001, 200002. For each of D1/D2,
hard_reset=False produces one unique peg pose across the three resets;
hard_reset=True produces three unique peg poses. Detailed measurements:
`development/proposal_v2/results/e4_eval_audit_20261124/reset_audit.json`.
No running source, configuration, process or environment was changed.

## Interpretation of existing scores

EquiDiff seed 0's final epoch-500 EMA produced D0 85/100, D1 0/100, D2 0/100
under the existing evaluator. The zero scores are actual rollout outputs,
not missing data, but D1/D2 must be labeled provisional evaluator outputs,
not validated full-distribution transfer estimates. This audit does not
establish that the reset defect explains all failures or that corrected
evaluation will improve the score. D2 also changes the agentview camera,
in addition to object/peg ranges and peg rotation.

Main Push-T RQ1/RQ2/RQ3 use a different evaluator and are unaffected by this
specific finding. The frozen main report excludes E4 results already.

## Required follow-up

Prepare a separate, versioned E4 evaluation runner. Validate per-episode peg
and nut resampling, reproducible full-scene pairing across methods, camera
and action semantics, rendering/memory behavior, and complete model rollout.
Re-evaluate saved final checkpoints, retaining original results as history.
Do not edit the active local/SICC frozen releases or interrupt training.
Blindly changing hard_reset=True in active code is not an approved solution:
the upstream comment notes rendering memory costs and cached-state restoration
also needs to preserve XML/model geometry, not just dynamic states.

The currently running DP/SICC lanes still use the old evaluator; completion
counts in their dashboards describe pipeline completion, not resolution of
this audit. New final D1/D2 outputs require the same qualification until
independent corrected evaluation has completed.

## Corrected evaluation deployment (16:24 CST)

User authorized corrected evaluation on available cluster resources. New frozen
release `e4_corrected_eval_20261124_v1` uses full seeded model resets, bypasses
states-only caching, and records per-episode peg/nut poses plus model, state and
observation hashes. Four-episode shards use independent processes to bound EGL
memory. The override is installed in spawned vector workers, not just the parent.
Source: `codes/operations/e4_corrected_eval_20261124/`.

- GPU reset preflight **29923** passed on SICC `um-gpu01`, RTX 2080 Ti,
  PyTorch 2.7.1+cu128. D1 three seeds produce three distinct peg/nut layouts;
  repeating seed 200000 after intervening scenes reproduces exact state/model
  and initial image hashes. Peak process RSS stabilized around 1.4 GiB.
- A100 account submission limit is two and both slots hold existing training.
  H800 was pending with a late scheduler estimate, so the pending diagnostic
  was moved to the idle `gpu_batch` partition, excluding um-gpu03/04.
- Full suite **29929** is RUNNING on `um-gpu01`. At this timestamp, it is
  performing paired-constructor audits and real-policy smoke tests, not yet
  producing the 300 formal episode scores. Sequence: D1, D2, D0, 100 each.
- Seed-0 final epoch-500 / 112000-update EMA was exported losslessly from
  checkpoint SHA256 `7ffe554fccc0ad5eb4a54a07421eeb15c599d94c132317651ed4e88aff57a219`.
  Inference package SHA256 `a798eb704a33c2b8a88c761ed3934ed8c774311f8f6944785dee7699ea500da8`.
- Local watcher PID **691392** pulls results every minute. After completion it
  verifies all local result hashes and the retained local EMA, then removes
  only the redundant remote inference package. Original checkpoints remain.
- Status: `codes/tasks/rq1_transfer/results/e4_corrected_eval_20261124/STATUS.md`.
- Existing 5090 DP and SICC EquiDiff seeds 1/2 continue unchanged. Legacy queue
  is empty but has no prepared E4 simulator environment; this deployment uses
  the already verified SICC project environment.

The corrected diffusion RNG restarts per shard and cluster PyTorch differs
from local training. Before/after scores therefore cannot isolate reset as
the sole changed factor. The new full-scene protocol is applied to D0 too.
Other future final outputs from old training lanes still need corrected
evaluation; their automatic old transfer scores are not scientifically cleared
by the launch of this seed-0 evaluation.

### v2 retry (16:30 CST)

Job 29929 passed both constructor-independent D1 resets with identical scene
and image hashes, but failed before formal rollout while loading ESCNN cached
expanded filters. New v2 calls `policy.train()` before strict EMA loading and
`eval()` afterwards to regenerate caches. CPU verification matched every saved
EMA tensor exactly. Job **29935** runs the same suite from
`e4_corrected_eval_20261124_v2`; watcher PID **694068** uses separate `v2/` results.
The previous watcher exited on failure; previous evidence remains intact.
Failed v1 remote inference weights were removed only after verifying local
weights and every archived result/log hash; receipts are in
`development/proposal_v2/operations/e4_corrected_eval_20261124/`.
Fleet's auxiliary registry includes 29935. At this timestamp GPU model smoke
tests are still pending/in progress; no corrected success rate is available.

### Formal evaluation started (16:38 CST)

Job **29935** passed all D0/D1/D2 paired-constructor scene/model/state/image
checks and saved-policy GPU smoke rollouts, then entered formal D1 seed200000
shard. Public evidence: `codes/operations/e4_corrected_eval_20261124/preflight_evidence.json`.
Smoke rollouts have eight steps and must never be used as scientific scores.
Pinned robosuite emits ignored EGL destructor errors on worker exit after
completed rollouts; subprocesses exit zero and each shard has a fresh process.
Existing training source hashes rechecked unchanged (local139files, remote144
each). Local DP epoch320; SICC EquiDiff seed1 epoch157 and seed2 epoch150 at
16:34 check. No corrected final rate is available yet. GitHub code pushed at
6523240; subsequent evidence-only documentation commit records these passed gates.

### First formal shard completed (16:42 CST)

Job 29935 completed D1 seeds 200000--200003, all 400 steps; four failures
are only an interim four-episode result, not a 100-episode domain estimate.
Rollout took132.45seconds; total shard wall time including initialization was
about226seconds. Suite advanced automatically to seeds200004--200007.
Rough full-suite estimate at this throughput:4--6hours, subject to later shards.
Code and passed preflight evidence pushed to GitHub main at8eb6295.
