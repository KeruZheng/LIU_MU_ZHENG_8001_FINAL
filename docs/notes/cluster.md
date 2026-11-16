# Cluster Tasks

Updated: 2026-11-16. The project is authorized to use the workstation and
Slurm clusters. No remote deployment or job submission has succeeded yet.

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
- [ ] Verify SSH connectivity, account permissions, storage and current queues.
- [ ] Create the shared `robotic` root and upload code plus data.
- [ ] Install an independent project environment; do not copy the workstation venv.
- [ ] Submit the CUDA/model preflight through Slurm.
- [ ] Run a DP pilot to measure memory, throughput and evaluation cost.
- [ ] Freeze a validated protocol and submit formal experiments in quota-sized batches.

GPU tests and model execution must run on allocated compute nodes, never the
login node. Do not override Slurm's `CUDA_VISIBLE_DEVICES`. Keep source/input
versions fixed while jobs run. Record cluster, job ID, config, source/data hash
and output path for each submission.

## Access status

The agent session currently cannot access the workstation GPU or open the
cluster SSH connections. The workstation owner supplied a healthy RTX 5090
`nvidia-smi` report at 2026-11-16 13:12 CST, confirming that the host driver
works. Agent-side access restrictions do not establish a host hardware,
driver or cluster account fault.

No Slurm job IDs are available for this project yet.
