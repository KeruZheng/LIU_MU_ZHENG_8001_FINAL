> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/live_status.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Autonomous Run Status

Generated: 2026-11-22T18:44:53.361937+08:00

This monitor reads current Slurm job IDs and copies results without checkpoints. A separate storage guard verifies checkpoint backups and gates the remote queues.

| Queue | State | Current Job | Step |
|---|---|---|---|
| c4_main_20261118_v1 | complete | - | - |
| rq3_state_20261118_v1 | complete | - | - |
| c8_ablation_20261118_v1 | complete | - | - |

## Slurm

```text
27690|robotic_cache_calib|COMPLETED|00:09:47|0:0
27690.batch|batch|COMPLETED|00:09:47|0:0
27690.extern|extern|COMPLETED|00:09:47|0:0
27691|robotic_cache_general|COMPLETED|00:02:19|0:0
27691.batch|batch|COMPLETED|00:02:19|0:0
27691.extern|extern|COMPLETED|00:02:19|0:0
27809|robotic_rq3_guard|COMPLETED|08:57:37|0:0
27809.batch|batch|COMPLETED|08:57:37|0:0
27809.extern|extern|COMPLETED|08:57:38|0:0
27812|robotic_c8_guard|COMPLETED|2-21:12:10|0:0
27812.batch|batch|COMPLETED|2-21:12:10|0:0
27812.extern|extern|COMPLETED|2-21:12:10|0:0

```

Summary: `development/proposal_v2/results/summary/matrix_summary.json`

Full monitor record: `development/proposal_v2/results/monitor/latest.json`

Storage guard: `development/proposal_v2/results/storage_guard_20261119/status.json`
