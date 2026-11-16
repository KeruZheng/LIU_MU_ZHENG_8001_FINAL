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

The host RTX 5090 and driver were independently confirmed healthy by the
owner. The agent session still lacks GPU/SSH access. No cluster upload or
formal GPU run has been completed.
