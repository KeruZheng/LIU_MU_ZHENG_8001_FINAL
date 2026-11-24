> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `development/proposal_v2/releases/rq3_image_rotated_20261120_v1/PROTOCOL.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Image RQ3: visible-goal extension

Frozen November 20, after inspecting state and primary image results. This is
an exploratory implementation of the proposal's image extension, not a new
pre-registered confirmation of the state-input hidden-goal hypothesis.

Matrix: fixed/rotated goal x {DP, EquiDiff-C4, EquiDiff-C4 + gate} x three seeds,
50 demos, 20k updates, batch64, [512,1024,2048] widths, unchanged EMA/optimizer,
100-step DDPM, FP32 with TF32 disabled. The fixed condition uses canonical demos;
the rotated condition uses the existing balanced C4 labels. Goal remains visible
in RGB for every method. An extra goal-input arm is not added because the image
already carries that information. A fixed-goal image does not have the state's
unobserved-goal ambiguity, so any deficit cannot be attributed to that mechanism
without further evidence.

The gate adds a fixed world vector (1,0) through a zero-initialized learnable
scalar into the equivariant observation projection. Zero gate preserves C4
equivariance; nonzero gate can break it. Unit tests verify both, gradient flow,
and visible goal differences with unchanged physics. Each allocation runs these
tests, full native replay, full-width batch64 backward, and short native rollout
checks before any formal training. Preflight failures prevent dispatch.

Evaluation: the existing 25 paired test base states x four initial orientations,
300 control steps, retain 16k/18k/20k checkpoints and report their mean. Fixed
condition keeps world goal fixed while initial states rotate. Coverage is
intersection/goal area; success is >0.95. State and image results stay separate.

There are 18 comparison cells but only 12 new trainings: the six completed
rotated DP/C4 50-demo primary runs are reused explicitly by global run ID,
not rerun or counted twice. Nine new fixed-goal runs and three rotated gate runs
use independent releases. This resolves the proposal's unspecified 18-image-run
layout without wasting six identical baseline trainings. The original proposal
described 18 additional runs; this is a documented deduplication of that budget.

Every complete run is copied locally and hash-verified before remote checkpoints
are deleted and the next run starts. No active primary source is modified.
