# Local C4 convergence follow-up

Update November 22: ownership of all six curves has moved to the legacy cluster.
The local 12k checkpoint is preserved and imported for seed-0 continuation.
Do not restart this local supervisor; see the
[migration record](../migrate_c4_legacy_20261122/README.md). The text below records
the original local arrangement.

The completed primary matrix frees the local 5090 for six new validation curves:
25 and 100 demos, three seeds each, 20k/40k/60k milestones. These complement the
50/200-demo C4 curves already owned by the cluster queues; no run is duplicated.
This is an exploratory follow-up motivated by primary results, not an original
proposal requirement or a replacement for the reserved test results.

The model, split, optimizer and validation grid match existing C4 curves. Runs
start fresh, alternate 25/100 demos by seed, and use the validated local project
environment. The supervisor checks CUDA and full native replay before training,
refuses concurrent ownership, verifies frozen source before each subprocess,
and records live process/phase metadata for the fleet monitor. A 70% per-process
GPU allocation cap leaves room for existing workstation applications.

All checkpoints remain in their original local run directories. On completion,
the supervisor records their hashes and fsyncs them; this is not a second backup
copy and authorizes no deletion. No remote environment or active queue changes.
