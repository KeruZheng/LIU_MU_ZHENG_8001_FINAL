# Project Task List

Presentation: **2026-11-26**. Updated: 2026-11-16.

## Completed

- [x] Preserve official DP and EquiDiff sources at pinned commits.
- [x] Download and validate the official 206-trajectory Push-T dataset.
- [x] Freeze a 200/6 trajectory train/validation split and nested 50-demo subset.
- [x] Implement C4 transforms, isotropic normalization and sequence loading.
- [x] Implement a DP-style pilot, explicit Frame Averaging backup and checkpoint recovery.
- [x] Run real-data CPU training, checkpoint loading and simulator evaluation.
- [x] Pass seven tests, including exact resume consistency and coupled-noise sampling equivariance.
- [x] Check both tiny and full pilot model forward/backward on CPU.
- [x] Prepare cluster configuration examples, upload and Slurm preflight scripts.
- [x] Verify workstation CUDA computation and SSH access to both clusters.
- [x] Pass full pilot model GPU preflight, 100 DP-style updates and two short rollouts.
- [x] Deploy SICC code/data and an isolated Python 3.10 / PyTorch 2.7.1 environment.
- [x] Complete A100 preflight job 27159: full model CUDA check and all seven tests pass.

## Remaining

- [ ] Run cluster pilot cost calibration and verify remaining group storage quota.
- [ ] Reproduce an official DP GPU pilot and measure learning/compute cost.
- [ ] Port original EquiDiff to Push-T; the current FrameAvg backup is not that port.
- [ ] Align native rendering and demonstrations with the corrected rotation center.
- [ ] Validate demo replay and define the exact augmentation protocol.
- [ ] Freeze the formal architecture, step budget, evaluation and checkpoint rules.
- [ ] Train 2 data sizes x 3 methods x 3 seeds (18 runs), prioritizing the 9 low-data runs.
- [ ] Aggregate per-seed/per-orientation metrics and success/failure rollouts.
- [ ] If budget permits, add the state-input goal-conditioning repair experiment.
- [ ] Produce slides and rehearse.

## Schedule

| Dates | Target |
|---|---|
| Nov 16-17 | Source/data/environment setup, correctness checks, GPU/cluster preflight |
| Nov 18-19 | Native data alignment, original architecture port decision, cost calibration |
| Nov 20 | First valid three-method low-data comparison |
| Nov 21-22 | Repeat seeds/high-data runs; slide draft |
| Nov 23-24 | Complete core matrix and freeze presentation results |
| Nov 25 | Final slides and rehearsal |
| Nov 26 | Presentation |

Schedule remains contingent on architecture porting and native symmetry validation.
