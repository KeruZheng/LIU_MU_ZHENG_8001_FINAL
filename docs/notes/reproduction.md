# Reproduction Status

Updated: 2026-11-16. **There are no formal reproduction results yet.**

## Sources and data

- DP: `5ba07ac6661db573af695b419a7947ecb704690f`.
- EquiDiff: `e40abb003b25071d4e5b01bfa9933dc16cd32c67`.
- Official Push-T: 206 trajectories, 25,650 frames, 96x96 RGB observations.
- Split: 200 training / 6 validation trajectories, with a fixed nested 50-demo subset.
- Upstream license files are retained in restored source directories.
- Archives, extracted data and source checksums are recorded in `manifests/`.

## What is implemented

The pilot reuses the official DP temporal U-Net, with ResNet18/GroupNorm and
an explicit C4 Frame Averaging option. It supports EMA, checkpoint/RNG recovery
and original Push-T coverage evaluation. The native rotated environment is a
prototype. Formal and rotated training are blocked pending validation.

This is not yet the official DP recipe or original EquiDiff architecture.
Differences include pooled image features, no random crop, reduced U-Net
widths, constant learning rate and replacement sequence sampling. Tiny
models are engineering tests only.

## Verified

Seven tests passed: pixel/coordinate rotation agreement, C4 composition and
normalization, split integrity, episode boundaries, FrameAvg equivariance
and gradients, uninterrupted/resumed training equality, and coupled-noise
DDPM sampling equivariance. Both tiny and full pilot CPU preflights passed.

A two-step tiny-model run completed training, checkpoint reload and two short
simulator episodes. Success was zero; this has no performance significance.
Uninterrupted four-step training and two-plus-two resumed training produced
bit-identical model and EMA weights.

Native symmetry diagnostic: five initial-state seeds, 20 control steps each:

| Quantity | Maximum error |
|---|---:|
| Block position | 3.41e-13 |
| Coverage | 6.94e-16 |
| Mean absolute pixel error, 0-1 scale | 0.001399 |
| Individual pixel error, 0-1 scale | 0.72549 |

This short diagnostic is not proof of exact simulator symmetry. Rendering
still differs, and original demonstrations target (256,256) while the
prototype targets (255.5,255.5). Data alignment and demo replay remain open.

## Environment isolation

Workstation development used a project-local venv that reads the existing
PyTorch installation through `--system-site-packages`; new dependencies were
installed only in the project venv. Existing project environments were not
upgraded or replaced. Fresh-clone and cluster recipes create independent
environments. The environment/data/checkpoint directories are not in Git.

On 2026-11-16 after 13:45 CST, the agent verified RTX 5090 access, actual CUDA
forward/backward computation and authenticated SSH to both clusters. No
dependencies were installed and existing training processes were preserved.
SICC deployment and environment validation are now complete; no formal GPU
run has been completed.

## GPU engineering pilot

Full pilot model FrameAvg preflight passed with maximum single-step C4 error
2.235e-8. The existing DP-style configuration completed 100 updates on 50 demos
(seed 0, batch 16). Mean loss fell from 1.168895 over the first ten updates to
0.117876 over the last ten. Peak PyTorch allocated memory was 2.093 GiB;
updates 11-49 ran at about 15.96 updates/s. These short measurements do not
establish convergence or a formal training budget.

Checkpoint loading and two 30-step rollouts on validation seeds completed;
mean maximum coverage was 0.069664 and success rate was zero. This is an
engineering check, not a method performance comparison. Local artifacts are
in runs/dp_gpu_pilot_20261116/ and results/preflight_full_gpu_20261116.json.
The full DDPM sampler GPU equivariance check remains pending.

## SICC environment validation

All 522 source and 819 data files passed remote checksums. An independent
Python 3.10.19 interpreter and venv were installed inside robotic/envs, using
PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128 and NumPy 1.26.4. pip check passed.
Only project-owned environments were modified; existing jobs were preserved.

A100 preflight job 27159 completed with exit 0:0 in 51 seconds. The full pilot
model passed CUDA forward/backward (2.315 seconds, single-step C4 error 1.490e-8)
on NVIDIA A100-SXM4-40GB; all seven tests also passed. The existing unit tests
still run on CPU. Artifacts are in results/sicc_deployment_20261116/.

This validates the DP-style/FrameAvg environment. Original EquiDiff dependencies,
its architecture port, cluster pilot training and formal comparison remain open.
