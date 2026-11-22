# Migrated C4 convergence

Same original configurations and training/evaluation code, continued on legacy GPUs.
25-demo seed 0 resumes the verified local 12,000-step model, optimizer and RNG state.
Other seeds start fresh. All runs train to 20k/40k/60k and use 100-episode validation at each milestone.
Hardware and PyTorch versions change; do not claim bitwise continuation equivalence.
Completed runs must be copied locally, hash-verified, and remote checkpoints cleaned before the next seed.
