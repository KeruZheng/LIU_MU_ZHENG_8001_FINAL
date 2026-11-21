# Local Stack-D1 priority run

On November 22, the user assigned the local RTX 5090 to Stack-D1 as the highest
priority. The independent C4 25/100-demo follow-up lane was paused at its atomic
12,000-update checkpoint after CPU deserialization, optimizer/RNG presence
checks, SHA256 verification, and fsync. Its frozen source was not modified.
Other applications and both clusters continue independently.

## Running reference

- Frozen release: `stack_d1_s0_local_20261122_v1` under the workspace's
  `development/proposal_v2/releases/` directory.
- Task evidence entry: `tasks/rq1_transfer/results/stack_d1_s0_local_20261122_v1`
  (a local symlink to the frozen release and its outputs).
- Run: `runs/stack_d1_n100_s0`, seed 0, 100 demos, 500 epochs, batch 128,
  full C8 image model, default optimizer/scheduler/EMA and 100 diffusion steps.
- The official 98/2 train/validation split has 9,876 training sequences,
  77 full batches per epoch, and 38,500 total optimizer updates.
- Scheduled evaluation remains every 10 epochs, six training initial states
  plus 50 test initial states, at most 400 steps each. A separate final-checkpoint
  evaluation runs after epoch 500.
- Operational overrides: four rollout workers instead of 28, two training and
  two validation loader workers, disabled external W&B logging, and an 80%
  PyTorch allocator cap. Episode counts, batch size, model size, epochs, and
  diffusion steps are unchanged. Runtime dependency differences from the
  historical environment are documented in the
  [environment preparation](../../tasks/rq1_transfer/stack_d1/README.md).

`run.py` verifies frozen file hashes and the converted input hash before use.
Checkpoint writes are synchronous and atomic, with fsync before rename.
The upstream algorithm is copied unchanged into the release. A heartbeat
thread and rollout timing wrapper only observe progress. The release has its
own `config.yaml` and `frozen_files.json`; do not edit it while running.

`supervise.py` holds a singleton lock and requires a successful batch-128
preflight before starting a fresh full run. It refuses to overwrite prior
preflight/training logs or outputs and does not blindly retry failed training.

## Actual resource preflight

Ten complete batch-128 updates passed, followed by batch-128 100-step inference
and a short rollout through four spawned simulators. Steady update time was
0.4369 seconds (about 2.29 updates/s), and peak allocated GPU memory was
15,202,375,680 bytes (14.16 GiB). These checks are not performance results.
The first formal epoch also ran at approximately 2.3 updates/s.

The measured update time projects about 4.7 hours of optimizer updates alone.
Total time includes 50 scheduled simulation evaluations, the final evaluation,
validation, sample diagnostics, checkpoints, and final hash verification.
Early rollout groups took roughly 35 seconds per four-episode group; a full
56-episode evaluation has 14 groups. Initial planning is approximately 12-16
hours from startup, subject to full-round and sustained-throughput measurements.

## Monitoring and C4 continuation

Read `results/supervisor_status.json`, `runs/stack_d1_n100_s0/status.json`, and
`logs/train.log` in the frozen release. `logs.json.txt` is the upstream per-epoch
scientific log, and `checkpoints/` retains latest/top-five/final states.
The fleet configuration monitors the supervisor and the continuation watcher
as services, separately from the Push-T primary-matrix counts.

The independent `resume_c4.py` watcher waits for successful completion of the
entire Stack run. It checks the final evaluation, all checkpoint inventory
hashes, the preserved C4 checkpoint, and original C4 frozen source hashes before
launching the original C4 supervisor from its recorded resume command. Failure
of Stack-D1 leaves C4 paused for investigation. No checkpoint is deleted.
The original C4 supervisor restores model, EMA, optimizer, and random states;
the pause receipt records the exact checkpoint and resume entry point.

Private handoff receipts and the frozen continuation watcher are under
`development/proposal_v2/operations/stack_d1_priority_20261122/`.
The resume guard tests cover success/idempotency, corrupt Stack checkpoints,
and failed Stack runs:

```bash
python3 -m unittest discover -s operations/stack_d1_local_20261122 -p 'test_*.py' -v
```
