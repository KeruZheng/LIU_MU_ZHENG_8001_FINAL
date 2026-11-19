# Three-platform monitoring

This service observes the local 5090, SICC, and the legacy Slurm cluster. It copies
small result artifacts into the research-task layout. It does not train models,
change cluster configuration, submit/cancel jobs, delete models, or alter active
training releases. Existing hash-verified archive workers retain cleanup ownership.

Local entry points:

- Live dashboard: [results/STATUS.md](results/STATUS.md).
- Machine-readable observations: `results/latest.json`, append-only `history.jsonl`.
- Collector health: `results/heartbeat.json`, `collector.log`, `collector_error.json`.
- Independent watchdog health: `results/watchdog.json` and `watchdog.log`.
- Per-task evidence indices: `../../tasks/<task>/results/run_index.json`.

The collector queries both schedulers every 120 seconds, checks live process
identity and archive-worker heartbeats, copies JSON/JSONL results without models,
and reports failed jobs, unreachable clusters, stale metadata, low local disk,
and 30 minutes without observed step/episode/phase progress. Native replay,
checkpoint saves, and expensive evaluation can cause a progress warning; it is
an investigation signal, not permission to kill a job. Slurm PENDING is always
displayed as waiting for resources. Filesystem free space is not user/group quota.

The independent watchdog checks the collector every 60 seconds. It reports a
missing process or 10 minutes without a successful snapshot. With
`--restart-collector`, it may restart only a dead collector, using a singleton
lock and a five-minute retry interval. It never kills an existing process and
never restarts training or archive services. The collector also checks the
watchdog, so one failed monitor is visible to the other.

Both processes run on the local workstation. They are not independent of a
workstation power/network failure; neither can report while the entire machine
is down. Remote archive gates preserve checkpoints and wait if local transfer
fails. True off-machine alerting needs a separately configured observer/channel.
No external messages or agent/API calls are enabled by this service.

`config.local.json` contains private endpoints and absolute paths and is Git
ignored. It records releases, owning/shared research tasks, jobs, checkpoints,
and monitored worker processes. `config.example.json` documents the schema.

```bash
python3 operations/fleet/collect.py --config operations/fleet/config.local.json \
  --output operations/fleet/results --once
python3 -m unittest discover -s operations/fleet -p 'test_*.py' -v
```

The deployed processes use a frozen copy under ignored `runtime/v1/` so editing
reviewable monitor source does not silently change running code. Update by making
a new runtime version. JSON configuration is runtime data and can be updated
separately when job IDs change. The original primary monitor and archive workers
continue unchanged; this new collector checks them independently.
