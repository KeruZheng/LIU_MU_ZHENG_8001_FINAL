# Exact scientific snapshots

Each release contains byte-identical source, configurations, selected scientific
manifests, protocols, and tests from the corresponding frozen experiment.
`SNAPSHOT.json` maps every included file to SHA-256 and records the original
freeze manifest hash. Machine-specific queue scripts, private deployment history,
environments, datasets, checkpoints and logs are omitted. These are curated
scientific snapshots, not copies of the full historical deployment directories.

Keep the separate source versions: C4 image, C8 image, and RQ3 state releases
had different implementations at launch. Refactoring them into one new trainer
would lose the direct link to already-running results. Identical RQ1/RQ2 source
is shared through release references rather than copied into both task folders.

To prepare a new run from the repository root (`codes/`):

```bash
python3 experiments/prepare_run.py \
  --task rq1_transfer --release c4_main_20261118_v1 \
  --run dp_n50_s0 --label my_dp_n50_s0
```

The command verifies the snapshot and creates a new working directory under
`tasks/rq1_transfer/runs/my_dp_n50_s0/`, with data/upstream links to this checkout.
It prints validation and training commands, but launches neither. It refuses to
reuse a label. Use the environment appropriate to the release, and allocate a GPU
through Slurm on clusters. This is a preparation helper, not an installation or
full training/evaluation pipeline.

Restore pinned upstream sources/data using the existing `scripts/fetch_artifacts.py`.
EquiDiff additionally needs the pinned escnn fork described in the main protocol;
the older fallback requirements alone do not establish that environment.
Runtime details for existing results are in task evidence, not assumed from the
machine currently doing evaluation. The convenience helper does not implement
the multi-stage convergence runner; use its recorded lane protocol and the
separately frozen operations version for that study.
