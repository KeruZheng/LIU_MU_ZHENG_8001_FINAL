# Local E4 launch

E4 runs on the local RTX 5090 in the already tested Stack-D1 project environment.
The independent frozen release is `e4_square_d0_local_20261122_v1`; the task result
entry is `tasks/rq1_transfer/results/e4_square_d0_local_20261122_v1`.
See [the protocol](PROTOCOL.md) for the matched-budget comparison and its limits.

Queue stages: download pinned official Square D0/D1/D2 files, SHA256 verification,
exact first-100 D0 subset, official absolute-action conversion and array validation,
both-model full-size GPU preflight in all domains, then six sequential trainings
(EquiDiff/DP for seeds 0/1/2), each followed by final-checkpoint D0/D1/D2 evaluation.
Target demonstration arrays never enter training. Download/preparation is not GPU
training. Any failure stops dispatch and records an error; inspect before recovery.

Runtime entries: `results/supervisor_status.json` for phase and download progress,
`results/data_ready.json`, `results/preflight_passed.json`, and
`runs/<method>_square_d0_s<seed>/status.json`. Each run stores final domain scores,
completed status and local checkpoint hash inventory. The fleet monitors the
queue owner as a service, separate from the Push-T main matrix. Completed Stack-D1
was removed from the active-process service list after its completion was verified.

The data download receipts and private launch parameters are under workspace
`development/proposal_v2/operations/e4_20261122/`. The host's configured proxy was
unavailable; downloads use direct HTTPS, pinned HF revision and LFS checksums.
No system/network configuration or active training environment was changed.

Build using the project's Stack interpreter and a fresh release name/directory:

```bash
python operations/e4_20261122/build.py --project ../development/proposal_v2 --codes .
python -m unittest discover -s operations/e4_20261122 -p 'test_*.py' -v
```

The builder refuses existing releases. Published experiments/releases snapshots
include the pinned upstream files, launch source and all six YAML configurations.
Local GPU preflight must pass for both methods before any formal run begins.

Independent live observer (30-second refresh):

```bash
python operations/e4_20261122/monitor.py --root /path/to/frozen/release \
  --codes /path/to/codes --output tasks/rq1_transfer/results/e4_monitor
```

The output contains `STATUS.md`, auto-refreshing `status.html`, `latest.json`,
and `history.jsonl`. It checks queue identity/heartbeat and reports stalled
progress, failure, download throughput, preflight and per-run/domain results.
It never modifies the training queue. Run a frozen copy for persistent monitoring.
The existing fleet collector observes the monitor process and snapshot freshness.
