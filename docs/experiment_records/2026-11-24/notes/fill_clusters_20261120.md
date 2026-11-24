> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/fill_clusters_20261120.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Cluster queues filled on November 20

Audit and submission window: 22:47-23:12 CST. User explicitly requested filling
both clusters. Active local primary and SICC C8 source/config/env unchanged.

## Submitted jobs

| Cluster | Job | Resources | Scope | Observed at 23:08 CST |
|---|---|---|---|---|
| Legacy | 232668 | 2 GPUs | Aug convergence 25/50 demos, 3 seeds each | PENDING Resources |
| Legacy | 232669 | 2 GPUs | Aug convergence 100/200 demos, 3 seeds each | PENDING Priority |
| SICC | 28265 | V100 | C4 convergence 50 demos, seeds 1/2 | RUNNING |
| SICC | 28266 | V100 | C4 convergence 200 demos, seeds 1/2 | RUNNING |
| SICC | 28269 | H800 | Image RQ3 fixed: DP/C4/gate, 3 seeds each | PENDING Priority |
| SICC | 28270 | A100 | Image RQ3 rotated gate, 3 seeds | PENDING Resources |

Existing jobs 27812 (C8, A100) and 28232 (official DP, H800) remain. SICC now has
two submissions in each of A100/H800/V100, filling each partition's MaxSubmitPA=2.
Normal MaxSubmitPU=7 does not override those partition caps. Legacy submits two
two-GPU jobs, filling MaxJobsPU=2 and MaxTRESPU=gpu=4 when allocated. Its four-job
submission cap permits more waiting jobs, but no speculative or duplicate filler
was submitted. First legacy start estimate: November 21 05:36 CST, not guaranteed.

## Scientific scope

- 12 new augmentation convergence runs compare the missing method against the
  completed DP 20k/40k/60k curves. Validation only, fresh initialization, not a
  primary-result replacement. Designed after seeing primary results.
- Four additional C4 curves supplement completed seed 0 with seeds 1/2 at 50/200
  demos. V100 GPU UUIDs differ; full replay and CUDA checks passed and both lanes
  entered training at about 22:55. Torch 2.6.0+cu124, peak allocation 10481303552 B.
- Image RQ3 is now specified as fixed/rotated x DP/C4/gate x three seeds at 50 demos.
  Six rotated DP/C4 baselines are reused from the primary matrix by explicit run ID.
  Nine fixed runs and three rotated-gate runs are new. This is 18 comparison cells,
  12 distinct new trainings: a documented deduplication of the proposal's stated
  18 additional image runs. Unique primary+image target is now 78, not 84.
- RGB contains the goal, so image experiments must not claim the state's hidden-goal
  ambiguity. Protocol is an exploratory extension frozen after state results.
- Image tests passed locally: zero-gate equivariance, gate gradient and nonzero
  symmetry breaking, visible target with unchanged physics; seven image/state tests.
  Slurm image lanes still require full native replay and full-model batch64 backward
  and rollout checks before formal training. They have not passed GPU preflight yet
  while waiting for allocation. Failures prevent further dispatch.
- Stack-D1 reference and optional E4 are not deployed by this action.

## Provenance and monitoring

New source authoring directory: `development/proposal_v2/image_rq3_20261120/`.
Frozen copies: `development/proposal_v2/releases/rq3_image_*_20261120_v1/`.
Other new releases: `conv_aug_n*_20261120_v1`, `conv_c4_n*_s12_20261120_v1`.
Build/deployment documentation: `codes/operations/fill_20261120/`.
Submission receipts: `development/proposal_v2/operations/fill_20261120/`.

All eight new releases were hash-verified remotely and registered with the existing
archive worker before submission. Registry/config changes are data, not running
source edits. Every complete run gates its next seed on verified local offloading
and remote model cleanup. Fleet v2 configuration now includes all releases/jobs;
the dashboard shows all eight new lanes. No monitor daemon was stopped/replaced.
Fleet eight tests and snapshot layout two tests pass. New source/staged artifacts
were checked for private machine paths and model/environment files.

Published commit: `0c4f4b3` on GitHub main. Final 23:11 CST audit: SICC has three
running jobs (C8 plus two V100 convergence lanes) and three pending jobs (official
DP and two image RQ3 lanes); legacy has two pending two-GPU jobs. Local 5090
utilization was 100%. Fleet reports no current alerts; all archive/monitor
services are alive. Queue occupancy is not physical GPU occupancy.
