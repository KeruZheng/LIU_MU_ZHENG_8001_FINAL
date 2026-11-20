# Resource optimization and device compatibility

The primary local C4 queue and the SICC A100 RQ3/C8 queues keep their frozen
source, configurations, environments, and output paths. This operations version
adds independent work and measurements. It never cancels/restarts primary jobs.

## Allocation strategy

- Local 5090: retain the sequential 36-run primary image matrix.
- Legacy cluster: retain the two two-GPU DP convergence jobs, 232570 and 232571.
  Both were running on November 20 evening, ahead of the initial scheduler
  estimate. Extra idle CPU cores do not imply free GPUs.
- SICC A100: RQ3 job 27809 completed; preserve active C8 job 27812.
- SICC V100: separate 50-demo and 200-demo exploratory C4 convergence lanes,
  one GPU and one training seed per lane. Each trains from scratch and validates
  at 20k, 40k, and 60k, with verified archive gating.
- SICC H800: C8 throughput calibration 27855 completed. Official DP preflight
  27862 failed; A100 retry 28225 passed. Full official reference 28232 was
  submitted to H800 on November 20; see ../official_reference_20261120/README.md.

The live account audit additionally found per-account partition submission caps
of two for A100, H800, and V100, despite higher per-user limits. Normal QoS permits
seven submitted user jobs. The historical stuck COMPLETING job was absent from
the November 20 evening live queue. Do not change cluster policy or other users' jobs.

## V100 compatibility finding

Job 27854 failed its CUDA matrix-multiplication preflight before training. The
allocated device was a Tesla V100 PCIe 16GB (compute capability 7.0); the existing
torch 2.7.1+cu128 build supports capability 7.5 and newer. `nvidia-smi` alone is
not sufficient validation. This is a userspace compatibility issue, not evidence
that the V100 node itself is faulty.

A separate project venv `envs/v100` uses torch 2.6.0+cu124 and torchvision
0.21.0+cu124. Pure Python/scientific dependencies are read from the project's
existing environments; installations are confined to the new venv. Do not
upgrade or reinstall packages in an active training environment.

The C8 batch64 recipe already measured roughly 17 GiB peak allocation on A100,
so a 16GB V100 is not eligible for that unchanged recipe. Do not silently reduce
batch size or model width to fill that card. Use it for the smaller C4 diagnostic;
actual memory capacity and forward/backward execution still require Slurm checks.

The unused `conv_c4_n50_n200_20261119_v1` source snapshot records the original
sequential plan. Neither of its runs started in job 27854. It is superseded by
the two `conv_c4_n*_20261119_v2` lanes; never schedule both plans.

## Scripts and interpretation

`benchmark_then_converge.py` verifies CUDA and the frozen calibration source,
then measures the original C8 batch64/FP32/TF32-off 200-update configuration only
when device capacity permits. Separate Slurm-ID output directories prevent
cross-device overwrites. Finished calibration checkpoints enter the existing
verified archive service; calibration scores are not added to primary tables.

`v100_convergence.sbatch` selects the new V100 environment and the previously
frozen staged-convergence runner. `official_preflight.py` verifies 231 pinned
upstream Python/YAML hashes, uses the published full-width/84-pixel-crop model
in upstream debug mode, runs training and native rollout/video paths, and saves
checkpoints synchronously via atomic replacement. It uses two engineering rollout
seeds and six training batches; it is not the 3050-epoch reference experiment.

Do not launch copies of future C8 seeds while the original serial driver still
owns them. It automatically advances through all six runs. Parallelizing that
queue requires an explicit dispatch handoff at a safe boundary, not source edits,
speculative duplicate training, or killing the current experiment. The H800
measurement establishes whether such a handoff would save meaningful time.

## Observed outcomes on November 20

The two V100 jobs, 27871 and 27872, passed CUDA, full native replay, training,
and all three validation milestones through 60k. They completed in approximately
14.6 and 14.8 hours, including verified offloading. Distinct GPU UUIDs confirmed
that they used separate devices. The existing main queues were unchanged.

H800 calibration 27855 completed: median C8 update 0.8664 seconds; 200-update
loop 175.24 seconds. The ongoing A100 C8 run was approximately 1.82 seconds per
update. This roughly 2.1x comparison is a short runtime measurement, not a full
training/evaluation makespan estimate or a model-performance comparison.

Official DP preflight 27862 built the model but failed before training because
the pinned upstream asynchronous vector environment expects Gym 0.21's shared
memory API, while the project base supplies Gym 0.26. The dedicated
`envs/official_dp` environment now installs Gym 0.21.0; the active RQ3/C8
environment still reads Gym 0.26.2 from the original base. A source-distribution
SHA-256 check preceded the Gym build. The only Gym source-distribution edit was
its invalid packaging requirement `opencv-python>=3.` -> `opencv-python>=3.0`;
no Gym runtime files were changed. Build provenance is retained in the private
project environment directory. Dependency checks pass for official_dp.

Retry 28225 used the same frozen preflight program and model configuration and
COMPLETED on A100 in 5 minutes 6 seconds, with Slurm exit code 0:0. The result
records two debug epochs (six training batches), two rollout seeds, and about
6.14 GiB peak GPU allocation. Training, native rollout/video, validation, and
checkpoint saving passed. This completes the engineering preflight; the official
3050-epoch reference training has not been launched.

The V100 environment's dependency checker reports missing `py3nj`, as does the
project's C4 escnn setup. This optional SO(3) acceleration is not used by C4;
the completed C4 training/evaluation runs provide the relevant execution proof.
