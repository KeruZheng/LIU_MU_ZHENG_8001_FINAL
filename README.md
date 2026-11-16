# Robotic: EquiDiff / Push-T study

Repository: <https://github.com/KeruZheng/LIU_MU_ZHENG_8001_FINAL>.

This is the implementation workspace for the 2026-11-26 presentation. Official
DP and EquiDiff snapshots are in `upstream/`; exact commits and archive hashes
are recorded in `manifests/upstream.json`. Upstream files are unmodified.

## Current scope

- Official Push-T data downloaded and checked: 206 trajectories, 25,650 frames.
- Frozen split: 200 training / 6 validation trajectories; nested 50-demo subset.
- C4 transforms, isotropic coordinate normalization, and sequence dataset.
- DP-style pilot with upstream temporal U-Net, ResNet18 + GroupNorm, EMA,
  atomic checkpoints and explicit resume, original Push-T evaluation.
- C4 Frame Averaging backup, including noise/action/observation transformations.
- Native rotated environment prototype and separate physics/pixel diagnostics.
- Cluster upload, environment setup, and Slurm preflight/pilot scripts.

Original EquiDiff has NOT been ported to Push-T yet. The pilot is NOT the full
official DP training recipe: it uses pooled ResNet features, no random crop,
reduced U-Net widths, constant learning rate, and replacement sequence sampling.
`tiny=true` is exclusively for engineering tests, never a reported baseline.
Formal training and rotated-dataset training intentionally reject execution
until GPU calibration and native rendering/data alignment are implemented.

## Local commands

For a fresh clone, restore the pinned upstream archives and official data first:

```bash
git clone https://github.com/KeruZheng/LIU_MU_ZHENG_8001_FINAL.git
cd LIU_MU_ZHENG_8001_FINAL
python3 scripts/fetch_artifacts.py --data
python3.10 -m venv envs/local
envs/local/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
envs/local/bin/python -m pip install -r configs/requirements.txt
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/envs/huggingface"
envs/local/bin/python scripts/prepare_data.py
sha256sum --check manifests/code.sha256 manifests/data.sha256
```

The fresh-clone recipe uses a fully isolated environment. PyTorch 2.7.1 is the
proposed portable baseline, pending GPU verification; the development checks
recorded so far used the installed workstation PyTorch version below. GPU,
cluster connectivity and formal reproduction results are not yet verified
from the agent execution environment. The workstation owner independently
confirmed a healthy RTX 5090 driver on 2026-11-16.

Run from this directory. The local virtual environment inherits PyTorch from
the existing `modelscope_5090` environment, with new dependencies installed only
inside this project. Its absolute paths are workstation-specific.

```bash
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/envs/huggingface"
export SDL_VIDEODRIVER=dummy
envs/local/bin/python -m unittest discover -s tests -v
envs/local/bin/python scripts/preflight.py --output results/preflight_cpu.json
envs/local/bin/python scripts/check_symmetry.py

# Use a fresh output directory for every independent run.
envs/local/bin/python -m pusht_study.train --config configs/smoke_cpu.json \
  --output runs/my_cpu_smoke --device cpu
envs/local/bin/python -m pusht_study.evaluate \
  --checkpoint runs/my_cpu_smoke/last.pt \
  --output runs/my_cpu_smoke/evaluation.json --device cpu --episodes 2 --max-steps 8

# Only after CUDA is actually available:
envs/local/bin/python scripts/preflight.py --require-cuda --full-model
envs/local/bin/python -m pusht_study.train --config configs/pilot_dp.json \
  --output runs/dp_gpu_pilot --device cuda
```

Use `--resume` for a run with a complete `last.pt`. Only the target number of
steps may change on resume; method, seed, split and other configuration must
match. Checkpoints contain model, EMA, optimizer, RNG states, step and config.
Metrics include `attempt_start_step`; after an interrupted run, rows beyond
the restored checkpoint may repeat and must be deduplicated before plotting.

The original task reports coverage as intersection area divided by goal area,
not IoU or the clipped environment reward. Pilot evaluation is for engineering
validation only. The 100-episode rotated test protocol is reserved separately.

## Artifacts and reproducibility

`manifests/data.json` identifies the official archive and episode structure.
`scripts/prepare_data.py` verifies/extracts the archive and creates splits; it
refuses to replace a different existing split. `scripts/source_manifest.py`
hashes project and upstream source files. Run it after source changes before
upload. Environments, raw data and checkpoints are excluded from Git.

Native symmetry diagnostics currently show negligible short-rollout physics
error but nonzero raster error. In addition, original demonstrations target
(256, 256), while the prototype uses (255.5, 255.5). Image re-rendering and demo
replay must be validated before treating rotation augmentation as exact.

## Cluster

The entire contents of this directory map to one remote `robotic/` directory.
Thus code, upstream sources, data, environments and outputs stay together:

```text
robotic/
  src/  upstream/  scripts/  slurm/  configs/  manifests/
  data/  envs/  runs/  results/  logs/
```

Copy `configs/clusters.json` to `configs/clusters.local.json` and enter your
own cluster account, trusted SSH endpoint and absolute root ending in
`robotic`. The local configuration is Git-ignored. The example configuration
cannot execute actions until configured. Actual workstation deployment paths
are recorded separately in the local project notes and are not credentials
or defaults for other users.

```bash
python3 scripts/source_manifest.py
python3 scripts/cluster.py check --cluster sicc
python3 scripts/cluster.py upload --cluster sicc
python3 scripts/cluster.py preflight --cluster sicc
```

These print commands without running them. `--execute` performs the selected
action and records intents/results in `manifests/cluster_actions.jsonl`.
After upload, build the remote environment with `scripts/setup_cluster_env.sh`.
Do not copy the workstation virtual environment or run model tests on login
nodes. Source upload refuses to run while any `robotic_` job is active. Source
and input versions must remain immutable while jobs run. Both cluster recipes
need actual queue, quota and permission checks before use; no Slurm jobs have
been submitted by this setup.

Progress snapshots: [task list](docs/notes/list.md),
[cluster tasks](docs/notes/cluster.md), and [reproduction status](docs/notes/reproduction.md).
The original workspace maintains its detailed notes and path map outside the
code directory; published snapshots omit private cluster connection settings.
