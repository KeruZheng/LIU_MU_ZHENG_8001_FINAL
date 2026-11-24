> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/local_followup_20261122.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# November 22 local follow-up and remaining proposal scope

At 01:19 CST the primary local 36-run matrix, state RQ3 24 runs, image RQ3
12 new runs, ordinary DP convergence 12 runs and augmentation convergence
12 runs were complete. C8 was 4/6; C4 50/200 convergence was 4/6. All image
and augmentation runs have verified local archive receipts.

Official DP 28232 completed 3050 epochs, global_step 512400, Slurm 20:34:21,
exit 0:0. Verified local archive receipt exists. Native scheduled rollout scores:
last at epoch 3000 = 0.7991124191, best at epoch 550 = 0.8519331253. Do not call
the last scheduled rollout a final epoch-3050 evaluation. This is a single seed.

The local primary queue had finished and no follow-up was running. An unrelated
GPU service uses about 3.4 GiB and was left unchanged. The project dev Python
successfully ran CUDA matrix multiplication on 5090, torch 2.12.0.dev20260407+cu128.
Stack-D1 is not ready: robomimic, robosuite, mimicgen, mujoco, hydra and h5py are
absent from this project environment. No dependencies were installed into it.

User approved filling the missing 25/100-demo C4 learning curves while the
cluster independently owns 50/200-demo curves. New frozen local release:
`development/proposal_v2/releases/conv_c4_n25_n100_local_20261122_v1`.
Six fresh runs (seeds 0/1/2), alternating 25/100 by seed, validation at 20k/40k/60k.
These are additional exploratory diagnostics, not a proposal-required rerun or
primary test replacement. Original checkpoints are retained locally with hashes
and fsync; no second-copy backup is claimed and no files are deleted.

Supervisor 79096 uses the frozen scripts/supervise.py; initial training child
79266. Actual PIDs/phase live in results/supervisor_status.json. Full native replay
passed and first run equidiff_c4_n25_s0 entered training. A 70% GPU allocation cap
leaves space for existing workstation services. Source, data split, optimizer and
architecture match the existing C4 convergence recipes; only demos/seeds differ.
Fleet runtime configuration includes the new local lane. Existing monitor and
archive processes were left running. Snapshot tests (2) and fleet tests (8) pass.

Remaining proposal scope is not all scheduled: Stack-D1 reference, optional E4,
and full max-over-training for the primary study remain outstanding. Original
primary checkpoints only cover 16k/18k/20k and cannot supply a complete training
maximum retrospectively. Official DP reference is one seed, not a three-seed
reference. Newly added local curves must not be used to claim these gaps closed.
