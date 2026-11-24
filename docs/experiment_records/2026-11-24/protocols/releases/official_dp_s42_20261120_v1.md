> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `development/proposal_v2/releases/official_dp_s42_20261120_v1/PROTOCOL.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Official DP reference

One fresh seed (42) uses the published Push-T image configuration unchanged in
scientific settings: 90 demonstrations, 3050 epochs, batch 64, 84-pixel crops,
U-Net widths [512, 1024, 2048], EMA, cosine learning-rate schedule, 100 diffusion
steps, and native 6-train/50-test rollout every 50 epochs. This reference is
separate from the fixed-budget primary experiment and its data splits.

The frozen release verifies the pinned upstream files and its own source hashes.
Runtime changes are the dataset/output paths, disabled W&B logging, fresh-start
mode, synchronous atomic checkpoint writes, and an extra terminal checkpoint.
TF32 is disabled as in the successful preflight. Upstream source is unchanged.
Top-5 checkpoints plus latest and final remain available until completion;
the existing archive worker then copies all run artifacts locally, verifies
hashes, and deletes only the verified remote checkpoints.

Request one H800, 16 CPU cores, 96 GiB host memory, and the partition's four-day
limit. This is an allocation limit, not a runtime prediction. Estimate completion
after observing full epochs and native evaluation. A scheduler timeout must be
handled explicitly: upstream's resume loop must not silently add 3050 more
epochs. The runner refuses an existing run directory instead of auto-resuming.

Preflight 28225 passed on A100 with Gym 0.21 in the independent official_dp
environment. Each reference allocation repeats GPU and dependency checks.
