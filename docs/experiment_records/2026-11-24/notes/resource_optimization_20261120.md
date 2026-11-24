> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/resource_optimization_20261120.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Resource optimization handoff

Updated 2026-11-20 evening CST. Preserve all active primary training.

## Current evidence

- Main matrix: 25/36 fully evaluated on the local 5090 around 20:50.
- RQ3 state: 24/24 complete; job 27809 COMPLETED. All 24 local archive receipts exist.
- C8: 2/6 complete and archived; job 27812 trains `equidiff_c8_n50_s2` on A100.
- Legacy DP convergence: jobs 232570 and 232571 are RUNNING on fstsvr04,
  using four allocated GPUs across the four data-size lanes. Nine of 12 runs
  complete around 20:50; remaining seed-2 runs are finishing training/evaluation.
- New V100 C4 convergence: 27871 and 27872 completed their separate 50/200-demo,
  seed-0, 20k/40k/60k curves. Both verified archive receipts exist.
- H800 C8 benchmark 27855 completed and archived. Median update 0.86638 s,
  compared with about 1.82 s on the ongoing A100 C8 run. Do not extrapolate
  this ratio to evaluation or queue time.
- Original failed V100 attempt 27854 is resolved by the completed V100 jobs.
- Official DP preflight 27862 failed on Gym API compatibility. Retry 28225
  COMPLETED in 00:05:06 with exit 0:0 on A100 after installing Gym 0.21 only
  in the dedicated official_dp environment. The result records two debug
  epochs (six training batches), two rollout seeds, and 6588692480 peak GPU bytes.

Remote preflight.json and Slurm accounting both confirm the retry passed.
The preflight itself does not launch full training. A separate official seed-42
3050-epoch job, 28232, was submitted to H800 at 21:13 CST on November 20.
Release: `official_dp_s42_20261120_v1`, frozen original config (90 demos).
The initial four-day request exceeded H800 QoS's three-day cap despite the
partition allowing four days. Corrected only the pending job TimeLimit with
`scontrol update JobId=28232 TimeLimit=3-00:00:00`; frozen files unchanged.
At 21:18 CST the job was PENDING (Priority); scheduler tentative start was
2026-11-20 16:56:28 UTC, or November 21 00:56:28 CST. This is not a reservation.
No precise completion ETA before full epochs.
Fleet auxiliary monitoring and the verified archive registry include this job.

## Deployment changes

- `codes/operations/resource_20261119/`: C8 benchmark, V100 launch, official DP preflight.
- New runtime-compatible SICC environment `envs/v100`: torch 2.6.0+cu124,
  torchvision 0.21.0+cu124, NumPy 1.26.4; imports project dependencies from rq3,
  official_dp, robotic through a new environment-local .pth. Current V100 jobs
  completed before official_dp Gym was changed; future V100 jobs must revalidate
  their resolved dependency versions rather than assuming the previous runtime.
- V100 original preflight failed because torch 2.7.1+cu128 omitted sm_70. No
  cluster driver or configuration was changed. V100 is 16GB and cannot fit the
  unchanged C8 batch64 recipe, so it ran C4 diagnostics instead.
- SICC partition MaxSubmitPA is two for A100/H800/V100; MaxSubmitPU alone was
  insufficient for admission planning. An A100/H800 mixed submission failed
  MaxSubmitJobsPerAccount while both A100 jobs were present; the H800-only retry
  was accepted. All failed submission receipts remain in parallel_20261119 ops.
- Unused `conv_c4_n50_n200_20261119_v1` is superseded by the two v2 lanes.
  Do not schedule it again and duplicate the completed curves.

## Monitoring and storage

Fleet observer uses `codes/operations/fleet/runtime/v2/`. Initial new PIDs:
collector 3060240, watchdog 3060241; actual IDs are in fleet/results.
Only v1 observer processes were replaced; training and archival processes were
not stopped. V2 adds throughput estimates and separate auxiliary-job status.
Runtime config includes the new job IDs and remains Git ignored.

Generic archive worker 3030236 now serves both clusters. Its registry is
`development/proposal_v2/operations/parallel_20261119/active_servers.json`.
Remote `registry.json` includes C4 v2, C8 benchmark, and official preflight releases.
Completed main runs still use the original dedicated storage guard.

Verified receipts at 20:51: 24 RQ3, two C8 primary, nine legacy convergence,
two C4 convergence, one C8 benchmark. The legacy archive worker may spend several
minutes copying one checkpoint set; `active_archive` explains such heartbeat ages.

## Scientific interpretation

Single-seed C4 validation coverage at 20k/40k/60k:

| Demos | 20k | 40k | 60k |
|---|---|---|---|
| 50 | 0.339965 | 0.392075 | 0.412119 |
| 200 | 0.677349 | 0.747728 | 0.804519 |

These show continuing improvement in those two trajectories, not a three-seed
convergence conclusion. The primary 20k test protocol remains a fixed-budget
comparison. Never pool these validation curves with primary test scores.

## Still pending

- Resource-optimization code, frozen snapshots, official reference launcher and
  task definition were committed and pushed as db80ea028ecc6c84ddee287fd36c956736951049.
  Remote main was verified equal to local HEAD; working tree clean. Fleet eight
  tests and layout two tests passed. Fixed result reports remain dated November 19;
  live newer evidence remains in ignored task results, not in the fixed reports.
- C8's serial driver still owns its six original runs. Do not launch duplicate
  future seeds. A real parallel handoff needs a verified dispatch boundary;
  no such handoff has occurred, and the current C8 training must continue.
- Additional image RQ3, Stack-D1 reference, optional E4, and official long DP
  recipe results remain uncompleted; long DP seed 42 is now submitted.
  No scheduled LLM agent fills new work automatically;
  the existing queues and monitoring/archival daemons continue autonomously.
