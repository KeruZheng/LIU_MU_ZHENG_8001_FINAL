# Move the paused C4 follow-up to the legacy cluster

The user's November 22 instruction transfers ownership of all six 25/100-demo
C4 convergence curves from the local 5090 to the legacy cluster. Stack-D1 keeps
the workstation GPU. The local C4 auto-resume watcher has been stopped and its
fleet entry removed; it must not be restarted after Stack-D1 completes.

## Job and run ownership

| Slurm job | Frozen release | Runs | Initial state |
| --- | --- | --- | --- |
| 232837 | `conv_c4_n25_s1_legacy_20261122_v1` | 25 demos, seed 1 | Fresh training, A40 on fstsvr10 |
| 232837 | `conv_c4_n100_s02_legacy_20261122_v1` | 100 demos, seeds 0 then 2 | Fresh training, A40 on fstsvr04 |
| 232838 | `conv_c4_n25_s02_legacy_20261122_v1` | 25 demos, seeds 0 then 2 | Seed 0 resumes the preserved 12k checkpoint; seed 2 starts fresh |
| 232838 | `conv_c4_n100_s1_legacy_20261122_v1` | 100 demos, seed 1 | Fresh training |

Job 232837 received two A40s across two nodes and both curves passed full-model
batch-64 GPU preflight and native C4 replay validation before training. Job
232838 was submitted held until the 4.13 GB resume checkpoint finishes upload
and remote SHA256 verification. Releasing its hold only makes it eligible for
scheduling; it may still wait for GPUs. Four GPUs are requested in total, and
two were actually allocated at initial deployment. The source snapshots are
under `experiments/releases/`; the exact lane split is in [plan.json](plan.json).

Both runs retain the original frozen model, split, optimization and configs:
20k/40k/60k milestones with 100 validation episodes per milestone. These are
exploratory convergence checks, separate from primary held-out test results.
The old local six-run lane is retired from live fleet counts to prevent double
counting; its checkpoint, logs and source remain available for provenance.

## Environment and continuation boundary

The project-private `envs/c4_legacy_20261122` uses Python 3.11 and reads existing
legacy convergence/PyTorch dependencies without changing them. It adds the same
escnn fork (`fc4714cb6dc0d2a32f9fcea35771968b89911109`), lie-learn 0.0.2,
pymanopt 2.2.1 and autograd 1.8.0. PyTorch is 2.6.0+cu124; the actual allocations
are NVIDIA A40s. Optional SO(3) py3nj acceleration is absent, as on the local
C4 environment. GPU preflight executes inside Slurm, never on the login node.

The migrated seed 0 retains model, EMA, optimizer and all saved random states
at step 12,000. The runtime changes from RTX 5090 / Python 3.10 / PyTorch 2.12 /
CUDA 12.8 to the legacy runtime; this is not a claim of bitwise hardware-equivalent
continuation. Its original `runtime.json` is retained as
`runtime_before_migration.json`, and the new runtime is recorded separately.
Its exact scientific configuration must match the checkpoint before resume.

`check_gpu.py` tests a disposable full batch, including optimizer/RNG restoration
when an imported checkpoint exists. `run_lane.py` rejects a missing, partial,
or wrong imported checkpoint rather than silently starting seed 0 from scratch.
It then invokes the unchanged archive-gated `parallel_20261119/convergence.py`.

The initial cross-node submission serialized its unpinned steps on the first
node. A supplementary Slurm step, 232837.2, explicitly dispatched the 25-demo
lane onto the already allocated fstsvr10 GPU. The original production lane lock
and archive receipts prevent a later queued step from repeating scientific
training. The original submitted batch file is preserved in `submitted/`;
the current `two_lanes.sbatch` pins nodes explicitly for future submissions.
No running training source or remote submitted batch file was edited.

## Input staging and archive safeguards

`stage_resume.py` runs independently on the workstation. It continues an
interrupted upload into `import_last.pt.part`, checks size and SHA256 remotely,
renames the fully verified file to `last.pt`, and releases only the named held
job 232838. Failures leave the job held and generate a fleet service alert.
Four tests cover correct release, corruption rejection, wrong-job rejection,
and an already-running job. The importer checksum is:

```text
b93aa2922a156cf8023bbe6395171f16dacd342ca94f491d3f0c04d4fa978f17
```

The preserved original remains on the workstation. No original checkpoint was
deleted. Shared dataset/upstream hashes (1,316 files) and all new release source
hashes were verified before submission. Legacy quota was 242/500 GB at deployment,
with enough headroom for the four active lanes' temporary and retained checkpoints.

All four releases are registered with the existing archive worker and fleet
monitor. Each finished curve waits for local copying, complete hash verification,
and removal of remote duplicate checkpoint files before the next seed starts.
Raw evidence belongs to `tasks/rq1_transfer/results/<release>/`; verified model
archives use the existing `parallel_20261119/archive/legacy/` storage location.

Local private deployment receipts and staging status are under
`development/proposal_v2/operations/migrate_c4_legacy_20261122/` in the workspace.

```bash
python3 -m unittest discover -s operations/migrate_c4_legacy_20261122 -p 'test_*.py' -v
```
