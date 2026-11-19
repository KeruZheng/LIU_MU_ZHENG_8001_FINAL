# Research tasks and evidence ownership

Organize by research question; record the machine as run metadata. Machines can
change after node failure, but a run's scientific identity must remain stable.

```text
codes/
  tasks/
    rq1_transfer/           # Main data-efficiency matrix; DP convergence
      task.json            # Release membership and outstanding scope
      reports/             # Small, shareable result snapshots
      results/             # Local raw evidence and provenance (Git ignored)
      runs/                # New isolated reproductions (Git ignored)
    rq2_exact_aug/          # Main comparison references; C8 evidence
    rq3_symmetry_breaking/  # State fixed/rotated goal and fixes
  experiments/
    releases/              # Exact source, configurations, protocols, hashes
    prepare_run.py         # Prepare a new isolated run; never starts training
  operations/fleet/        # Three-platform collection and independent watchdog
```

RQ1 and RQ2 share the same 36 main runs. Raw main evidence is owned by RQ1; RQ2
references it through the same global `release/run` identifiers. Do not add the
RQ1 and RQ2 report row counts to obtain an experiment total. C8 has six independent
runs, and RQ3 state has 24, giving 66 unique primary runs. Twelve new convergence
runs are a separate validation study with a different runtime and fresh seeds.

Every task has a `task.json`, a live ignored `reports/latest.json`, and an ignored
`results/run_index.json`. The evidence index identifies the code snapshot,
configuration, training/evaluation state, device, Slurm job, runtime versions,
JSON evidence hashes, checkpoint locations, and validation/test split.
`reports/submission_20261119.json` is the fixed snapshot published to GitHub;
it does not claim to update when later experiments finish. Live files are
ignored so background monitoring does not continuously dirty the Git checkout.
Use the source runtime records when training and evaluation ran on different
machines. A recovery job ID is not necessarily the original training job ID.

Raw metric files and training curves are copied into task-owned `results/`
directories while runs continue. These are observed snapshots, not atomic
snapshots of all files at the same training step. Each file is independently
identified; only complete evaluation reports count toward completion. Large
checkpoints remain in their existing archive locations to avoid moving files
used by active queues or duplicating tens of gigabytes. Archive receipts, not
mere file existence, establish verified backup and authorize remote cleanup.

The top-level fallback implementation remains in place with its existing edits.
Do not mix its scores with the current main protocol. Do not move, rewrite, or
repoint active `development/proposal_v2/releases/` directories.

Submission: include task definitions, reports, scientific snapshots, and monitor
source. Do not include environments, datasets, model files, private SSH settings,
or live logs. No commit or push is performed by the monitoring programs.
