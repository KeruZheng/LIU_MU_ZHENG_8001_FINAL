# E4 parallel placement

The six-run Square D0-to-D1/D2 study is split by training seed. Local RTX 5090
owns EquiDiff and DP seed 0; two SICC lanes own the paired methods for seeds 1
and 2. Each lane runs EquiDiff followed by DP. No target demonstrations enter
training or selection. The original 500 epochs, 112000 optimizer updates,
batch 64, final EMA checkpoint and 100 episodes per evaluation domain remain.

## Source and environment

`build.py` verifies the original local E4 frozen manifest, copies its scientific
configuration byte for byte, and creates new immutable releases:

- `e4_square_d0_s1_sicc_20261123_v1`
- `e4_square_d0_s2_sicc_20261123_v1`

The trainer changes only GPU admission: CUDA compute capability >= 8 replaces
the RTX 5090 name restriction. Both methods and all three domains must pass
full-size GPU/sampling/checkpoint/rollout preflight before a lane trains.
The runtime input hash, split IDs, batch size and update budget must match the
local preflight. Each job records `nvidia-smi`, interpreter, and package versions.

`setup_env.py` installs into a new project-only venv. Existing official-DP,
RQ3 and base project environments are read-only dependencies. Simulation source
archives match the local pinned commits and SHA256 values. PyTorch3D uses its
official no-extension packaging option, as locally. Run `finish_env.py` next to
install Open3D, which the pinned Robomimic environment imports, with NumPy/SciPy
constraints and `--ignore-installed` to write only into this new venv. The SICC runtime uses
PyTorch 2.7.1/CUDA 12.8, while the RTX 5090 uses 2.12 nightly/CUDA 12.8; this
hardware/runtime difference must be reported, not described as identical
historical software. No system package, driver or cluster configuration changes.
OpenCV uses the matching 4.9 headless wheel because the cluster login environment
has no desktop `libGL.so.1`; MuJoCo rendering still uses EGL on allocated GPUs.
`private_libs.py` downloads Ubuntu graphics/C++ runtime packages and extracts
them under the project without installing system packages. `submit.py` exports
only these private library paths to the two jobs. `check_imports.py` checks package
imports on the login node without creating models, simulations or CUDA tensors.
The actual CUDA and simulator preflight runs only within Slurm allocations.

## Local handoff

`local_owner.py` first checks the live trainer identity, original supervisor,
phase, frozen hashes, and absence of seed-1/2 runs. It replaces only the old
queue-management process, adopts the still-running EquiDiff seed-0 trainer,
verifies all its final artifacts, and launches DP seed 0 using unchanged original
source. It never signals the training PID or its process group. A real-process
integration test confirms the child survives and only the two seed-0 runs finish.
The receipt and deployed owner/observer copies live outside the frozen release.

## Storage and observation

Each remote completed run waits for the existing verified archive worker.
The worker copies all results and models locally, verifies SHA256, then removes
only the corresponding remote checkpoints. A receipt gates the next method.
Failed stages stop dispatch; no automatic training retry or checkpoint overwrite.

The fleet collector mirrors small SICC result files approximately every two
minutes. The independent E4 observer combines them with the local run every
30 seconds at `tasks/rq1_transfer/results/e4_monitor/status.html` and `STATUS.md`.
Scheduler waiting, GPU preflight, training, evaluation and archive waits are
separate states. There is no external messaging service.

```bash
python -m unittest discover -s operations/e4_parallel_20261123 -p 'test_*.py' -v
python operations/e4_parallel_20261123/build.py --project /path/to/proposal_v2
```

Environment setup and release builders refuse reuse. Deploy into fresh paths,
verify remote hashes and data receipts, register the archive releases, then
submit one `train.sbatch` per seed through Slurm. Private endpoints, job receipts
and runtime paths are kept in ignored operational records and task evidence.
The submission overrides the frozen script's four-day default with three days:
H800's partition permits four days but its effective QOS permits only three.
Both jobs may be scheduled on A100 or H800; pending is not a running GPU preflight.

## EGL recovery, v2

The first two SICC attempts failed at their first native image rollout after
CUDA training updates and diffusion sampling. An allocated A100 diagnostic
confirmed successful CUDA backward, but only Mesa EGL vendor registration and
no NVIDIA EGL library in the job environment. It could not initialize rendering.

`private_nvidia_egl.py` downloads NVIDIA's exact 580.173.02 graphics and GPU
compiler packages, verifies their repository SHA256 values, and extracts them
under the project. It does not install packages or change the kernel driver.
Jobs explicitly select the private NVIDIA vendor JSON and library path.

`retry_egl.py` creates fresh `_v2` lanes with byte-identical trainers, upstream
source and YAML configurations. Old failed evidence is preserved. New lanes
have an `afterok` dependency on the small EGL diagnostic, then repeat both full
model preflights. Failed diagnostic dependencies cancel the gated jobs.
An account-wide submission cap initially admitted only the diagnostic and seed-1
retry. `recovery_dispatch.py` waits for proven rendering, then submits seed 2
when a slot is available and records its job in ignored fleet configuration.
It retries connection/submission-cap waits, not failed training. Unit tests
verify failed diagnostics never dispatch and account-cap waits preserve other
services. The E4 observer reads current release assignments from `task.json`.

The NVIDIA repair diagnostic subsequently generated a valid 84x84 RGB frame on
an A100 (identical SHA256 to the same simple scene locally), but its cleanup
incorrectly called `Renderer.close()`, absent in MuJoCo 2.3.2. That diagnostic
failure cancelled the dependent seed-1 job before it ran. `egl_diagnose_v3.py`
releases the two owned contexts explicitly and passes a real local CUDA/render
check with the same MuJoCo version. `submit_inline.py` now runs that corrected
check and the unchanged v2 lane in the same GPU allocation. Both seeds have
separate submissions, so no separate diagnostic job or submission dispatcher is
needed. The cancelled lane had no preflight or training outputs; its frozen
source and original cancelled-job receipt remain unchanged. Remote full-model
preflight must still pass before formal training.
