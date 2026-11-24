> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `development/proposal_v2/releases/e4_square_d0_local_20261122_v1/PROTOCOL.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# E4: Square D0 to D1/D2 transfer

Frozen November 22, before any E4 model or transfer scores. This is the optional
proposal extension on one MimicGen task family, not a reproduction across all
MimicGen tasks. Square has official D0, D1 and D2 distributions; Stack has no D2.

Two methods: official EquiDiff C8 image policy and official DP image policy,
using the pinned EquiDiff repository's configurations and respective observation
encoders/normalizers. Both use absolute 6-DoF actions plus gripper, two RGB cameras,
100 D0 demonstrations (demo_0..99), dataset split seed 42 (98/2), training seeds
0/1/2, 500 epochs, batch64 with drop_last, full [512,1024,2048] U-Net, 100 diffusion
steps, EMA, AdamW and cosine learning-rate schedule. EquiDiff's upstream batch128
is changed to batch64 to match DP's batch and optimizer-update budget. Runtime
worker counts are two for loaders and four for simulation. TF32 is disabled.
This is a matched-budget E4 comparison, not an unmodified official recipe claim.

The first 100 raw D0 demos are an exact verified subset of the official source.
Absolute conversion uses the unchanged official converter. Every non-action array
and metadata value is verified unchanged; gripper actions are preserved and all
converted actions finite. D1/D2 raw files are hash-checked, then only environment
metadata is exposed to the evaluation runner. Target demos never enter training,
normalization, model selection or hyperparameter tuning.

D0's official scheduled 6-train/50-test evaluation runs every ten epochs for
diagnostics. It does not choose the transfer checkpoint. The primary checkpoint
is fixed in advance: final epoch-500 EMA. Each final policy is evaluated on D0,
D1 and D2, 100 seeds 200000..200099 per domain, up to 400 control steps, no
fine-tuning. Policy random seed follows the method's training seed; simulator
seed IDs are paired across methods. Matching numeric IDs across distributions
does not imply identical scene states. Report per-training-seed success rates,
paired method differences within domain, and D0-to-D1/D2 drops separately.
No best-target-checkpoint selection. No performance claim from resource checks.

Before formal work, both full-size models must pass batch64 updates, 100-step
sampling, checkpoint serialization, and short two-episode D0/D1/D2 rollouts on
engineering seeds 900000/900001. Both data splits and update counts must agree.
The local sequential queue runs EquiDiff then DP for seed0, then seeds1/2.
Any failed stage stops dispatch and records the failure; no blind retries or
checkpoint overwrites. All model files stay locally with SHA256 inventory;
there is no remote-cleanup action for these local originals.

Environment: reuse the completed Stack-D1 reference's independent local venv
(Python3.10, PyTorch nightly/CUDA12.8, MuJoCo2.3.2 and pinned MimicGen/Robomimic).
Do not alter shared dependencies, drivers, active cluster jobs or frozen sources.
