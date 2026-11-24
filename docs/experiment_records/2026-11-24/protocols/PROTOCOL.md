> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `development/proposal_v2/PROTOCOL.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# C4 Main Protocol, Frozen Before Test Evaluation

Date: 2026-11-18. This is a Push-T adaptation study, not a claim to reproduce
the published MimicGen scores or the full official DP optimization recipe.

## Main Matrix

- 25, 50, 100, 200 independent training demonstrations, nested by the existing split.
- Three training seeds (0, 1, 2), shared across methods.
- Methods: DP, DP + exact C4 augmentation, EquiDiff-C4 Push-T adaptation.
- The existing DP-C4-FrameAvg runs remain a separately named fallback comparison.
- 20,000 optimizer updates, batch 64, U-Net widths [512, 1024, 2048], AdamW,
  learning rate 1e-4, existing EMA schedule, 100 DDPM training/inference steps.
- Fixed balanced base orientations for all arms; only the augmentation arm gets
  additional online C4 transformations. The original 200/6 split is unchanged.
- FP32 with TF32 disabled in all new training and evaluation processes.
- Caching and batched loading preserve sample values and RNG draws. Existing
  uncached runs are not silently substituted for new-protocol main runs.

## EquiDiff Adaptation

The encoder reuses the pinned upstream EquiResBlock. Its spatial schedule is
adapted from 76-pixel crops to full 96-pixel inputs without asymmetric pooling:
96 -> 48 -> 24 -> 12 -> 4 -> 1. There are 128 regular fields at the output.
Agent position and 2-D action are rho1 vectors, converted between image y-down
and escnn y-up conventions. The action embedding has 64 regular fields. The
temporal U-Net and group-axis sharing reuse the upstream implementation, with
the action output type changed from manipulation features to one rho1 vector.
The resulting model has 257,559,008 parameters; DP has 288,378,562 parameters.

The original escnn fork is pinned to fc4714cb6dc0d2a32f9fcea35771968b89911109.
Its optional SO(3) py3nj acceleration is absent; C4 does not use SO(3).
EMA coefficients are saved in train mode; escnn inference filters are rebuilt
after checkpoint loading to prevent stale expanded weights.

## Environment And Documented Proposal Corrections

Native pymunk physics runs in each rotated frame. Rendering inverse-transforms
the physical state to a shadow canonical scene, uses the unmodified upstream
rasterizer, then applies lossless rot90. The shadow scene is not simulated.
This is a deliberate renderer construction, not evidence that the original
pygame rasterizer itself is exactly rotation equivariant.

The walls and pixel lattice have center (255.5, 255.5). To preserve canonical
demonstration images, the original goal at (256, 256) is rotated in position
as well as heading about that center. Its four positions are (256,256),
(256,255), (255,255), (255,256). This differs from the proposal's fixed goal
position. Fixing all goals at 256 would not match exact image augmentation.
The original integer initial-state sampler is shifted by six pixels so its
position mean is 255.5. Its angle distribution and legacy initialization are
preserved, then the resulting complete state is rotated.

All six held-out demonstration action sequences are replayed in all four native
orientations, under both legacy initialization settings. Fixed tolerances are
recorded before results in scripts/validate_c4.py. This checks conjugacy during
contact; it does not claim that replay exactly reconstructs human recordings.
Sampled canonical images match the unmodified original renderer exactly. Stored
recordings show small residual re-render differences even with the original
renderer; results/dataset_render_alignment_v2.json reports that separately.

## Evaluation And Statistics

- Reserved test: 25 existing base seeds x 4 paired initial orientations, 300
  control steps maximum, native C4 environment, identical grid for all methods.
- Policy randomness: base_seed + 1000003 * orientation, paired across methods;
  orientations have independent stochastic streams, not coupled diffusion noise.
- Coverage means intersection area divided by goal area (not IoU).
- Success follows the upstream environment: coverage > 0.95 (proposal wrote >=).
- Retain EMA at updates 16,000, 18,000, 20,000, in addition to resumable last.pt.
- Primary aggregate: mean of the three retained-checkpoint test scores. Also
  report final 20,000 separately. The best of these three is only a limited
  checkpoint-grid maximum, not the paper's full max-over-training metric.
- Bootstrap samples paired training seeds and paired base initial states, keeping
  their four orientations together. Report seed values and uncertainty explicitly;
  three training seeds do not support precise generalization claims.
- Validation seeds remain separate; test results are not used to change this
  frozen model/optimizer/step protocol. Revisions require a new experiment version.

## Scheduling And Remaining Scope

Start with a 50-demo EquiDiff-C4 run, then interleave the three methods and seeds
for 50 and 200 demos before 25 and 100. Store source and configuration hashes.
The local queue uses an immutable release, never codes/src. Cluster cache
calibration is a separate immutable release. No running queue source is edited.

C8, RQ3 and optional E4 are not part of this C4 matrix and remain separate work.
The full DP reference recipe, full max-over-training metric, and MimicGen sanity
checks remain outstanding; completing this matrix alone is not the whole proposal.
