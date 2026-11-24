> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `development/proposal_v2/releases/conv_c4_n25_s1_legacy_20261122_v1/PROTOCOL.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Migrated C4 convergence

Same original configurations and training/evaluation code, continued on legacy GPUs.
25-demo seed 0 resumes the verified local 12,000-step model, optimizer and RNG state.
Other seeds start fresh. All runs train to 20k/40k/60k and use 100-episode validation at each milestone.
Hardware and PyTorch versions change; do not claim bitwise continuation equivalence.
Completed runs must be copied locally, hash-verified, and remote checkpoints cleaned before the next seed.
