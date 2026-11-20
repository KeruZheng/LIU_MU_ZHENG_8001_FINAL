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

Request one H800, 16 CPU cores, and 96 GiB host memory. The frozen batch script
initially requested the partition's four-day limit, but H800's partition QoS
caps jobs at three days. Job 28232's pending allocation was corrected with
`scontrol update JobId=28232 TimeLimit=3-00:00:00`; the frozen script was preserved.
Future submissions must pass `--time=3-00:00:00`. This is an allocation limit,
not a runtime prediction. Estimate completion
after observing full epochs and native evaluation. A scheduler timeout must be
handled explicitly: upstream's resume loop must not silently add 3050 more
epochs. The runner refuses an existing run directory instead of auto-resuming.

Preflight 28225 passed on A100 with Gym 0.21 in the independent official_dp
environment. Each reference allocation repeats GPU and dependency checks.
