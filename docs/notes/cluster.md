# Cluster Tasks

Updated: 2026-11-16. SICC code/data deployment, isolated environment installation
and A100 preflight job 27159 succeeded. No cluster training or formal comparison
jobs have been submitted.

## Layout

All project code and data must share one remote root named `robotic`:

```text
robotic/
  src/ upstream/ scripts/ slurm/ configs/ manifests/
  data/ envs/ runs/ results/ logs/
```

`configs/clusters.json` contains examples for SICC and a legacy Slurm cluster.
Actual endpoints, account identifiers and root paths belong in the ignored
`configs/clusters.local.json`. Follow each cluster's documented limits and
query current quotas before scheduling parallel jobs.

## Deployment checklist

- [x] Prepare source/data checksums, upload helper and Slurm templates.
- [x] Verify SSH connectivity, account permissions and current queues.
- [ ] Verify remaining group storage quota; filesystem free space is not this quota.
- [x] Create the shared `robotic` root and upload code plus data.
- [x] Install an independent project environment; do not copy the workstation venv.
- [x] Complete the CUDA/model preflight through Slurm (27159, exit 0:0).
- [ ] Run a DP pilot to measure memory, throughput and evaluation cost.
- [ ] Freeze a validated protocol and submit formal experiments in quota-sized batches.

GPU tests and model execution must run on allocated compute nodes, never the
login node. Do not override Slurm's `CUDA_VISIBLE_DEVICES`. Keep source/input
versions fixed while jobs run. Record cluster, job ID, config, source/data hash
and output path for each submission.

## Access status

GPU and SSH access were verified from the agent session after 13:45 CST on
2026-11-16. Both clusters accepted authenticated SSH with strict host-key
checking. The preferred cluster's account/QOS match the planned configuration.
SICC now contains the deployed robotic root and an isolated Python 3.10.19
interpreter under envs/python; envs/robotic does not inherit system packages.
The final stack uses PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128. pip check
passed. Storage quotas remain unverified. The backup cluster is not deployed.

Job 27159 completed on an NVIDIA A100-SXM4-40GB in 51 seconds. Full pilot model
CUDA forward/backward passed, maximum single-step C4 error was 1.490e-8, and
all seven tests passed. Logs and environment manifests were copied back under
results/sicc_deployment_20261116/. The user's queue was empty after completion.
The original EquiDiff dependency stack and architecture port remain unfinished.
