# EquiDiff Stack-D1 reference preparation

This is the proposal's W1-2 reference sanity check, separate from the Push-T
primary matrix and optional E4 (D0 training followed by D1/D2 transfer tests).
Preparation began on November 22, 2026. The full seed-0 reference subsequently
started on the local RTX 5090 after passing batch-128 resource preflight; see
[the priority-run record](../../../operations/stack_d1_local_20261122/README.md).
The evidence below documents the earlier environment-preparation checks.

## Environment and provenance

The workstation environment is `development/proposal_v2/envs/stack_d1_20261122`
relative to the parent workspace. It is a separate Python 3.10 venv. Its
`project_dependencies.pth` reads the existing `proposal_v2/envs/dev`,
`codes/envs/local`, and `modelscope_5090` dependency directories. New packages
are installed only into the new venv; the active training environments and
frozen releases are unchanged. This environment depends on those local paths
and must not be copied directly to a cluster.

- [sources.json](sources.json): fixed upstream commits, archive SHA256 values,
  dataset URL and verified SHA256, and packaging adaptations.
- [environment.json](environment.json): actual versions used by preflight.
- [preflight_report.json](preflight_report.json): shareable verification results
  and converted-data receipt, without workstation-specific absolute paths.
- [requirements.txt](requirements.txt): additional dependencies installed in
  the new environment; not a complete standalone environment lock.
- Local full dependency freeze and interpreter paths:
  `../results/stack_d1_20261122/environment.freeze.txt` and `interpreter.json`.

Robosuite, Robomimic, task-zoo, and MimicGen match the commits in the official
EquiDiff installation instructions. MimicGen and task-zoo use editable installs
from those pinned source directories so their simulation assets are available.
MuJoCo is the required version 2.3.2. Headless rendering uses the existing NVIDIA
EGL libraries (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`). No system packages,
drivers, cluster settings, or shell startup files were changed.

Compatibility differences from the original Python 3.9 / PyTorch 2.1 / CUDA 11.8
recipe are explicit: Python 3.10, PyTorch 2.12 nightly / CUDA 12.8, torchvision
0.27 nightly, and diffusers 0.30.3 support this workstation's RTX 5090. Other
actual dependency versions are recorded in `environment.json`. This is a tested
runtime adaptation, not an identical historical environment or a performance
reproduction result.

Gym 0.21 required one packaging-only fix, `opencv-python>=3.` to
`opencv-python>=3.0`, to build with modern tooling. Its runtime source is
unchanged. PyTorch3D 0.7.5 was built with `PYTORCH3D_NO_EXTENSION=1`; its pure
PyTorch rotation transforms are tested on CPU and CUDA. Compiled PyTorch3D
rendering/point-cloud operations are unavailable and are not used by this image
recipe. `pip check` reports one remaining metadata requirement: escnn's optional
`py3nj` SO(3) accelerator. The tested Stack image architecture uses C8 and does
not need that accelerator. Do not describe `pip check` as fully passing.

To reconstruct the local additions, create a fresh venv, append the three
read-only dependency paths in a `.pth` file, install `requirements.txt`, and
install the four pinned source packages with `--no-deps --no-build-isolation`.
Install `cmake==3.29.6` first and put the new environment's `bin/` on `PATH`
before installing the rest of `requirements.txt`; `egl_probe` needs CMake
while building its wheel.
Use editable installation for MimicGen/task-zoo. Build Gym with the packaging
fix above and PyTorch3D with the documented no-extension switch. The fixed source
archives and receipts are kept under
`development/proposal_v2/envs/stack_d1_sources`. The new environment's `bin/`
must be on `PATH` when compiling `egl_probe` so its private CMake is found.

## Verified scope

The local evidence is under `../results/stack_d1_20261122/`:

| Check | Result |
| --- | --- |
| Actual CUDA forward/backward | Passed on RTX 5090 |
| Official raw dataset | 1,000 demos / 108,233 frames; source SHA256 verified |
| Complete absolute-action dataset | 1,000 demos converted with the official script; all unchanged arrays and metadata checked; new SHA256 receipt recorded |
| Official 100-demo input and cache | Loader, train/validation split, sequence shapes, and finite normalizer checked |
| Stack_D1 reset/step | Passed with both 84x84 RGB cameras |
| H.264 video output | Passed |
| Two-demo absolute action conversion and replay | Passed; max position error about 0.8 mm, comparable to original relative-action replay |
| CPU/CUDA rotation round trip | Max absolute error about 3e-8 |
| Full C8 model, 271,175,734 parameters | Batch-1 forward/backward, AdamW update, EMA update, 100-step sampling passed |
| Spawned vector environment | One worker, eight simulation steps passed |
| Official training workspace | One training batch, one validation batch, one short rollout, checkpoint save/deserialization passed |
| Peak allocated GPU memory | 5.65 GiB model check; 6.58 GiB workspace check |
| Formal batch 128 / rollout concurrency | Subsequent batch-128 and four-worker preflight passed; default 28 workers not tested |

These checks use two demos and deliberately short execution. Their success score
is not scientifically meaningful. The checkpoint check verifies serialization
and deserialization, not full optimizer-resume equivalence.

Run from the `codes/` directory, using the separate interpreter. Set
`PYTHONPATH="$PWD/upstream/equidiff"`, `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl`, `OMP_NUM_THREADS=2`, and `OPENBLAS_NUM_THREADS=1`.

```bash
python tasks/rq1_transfer/stack_d1/preflight.py simulation \
  --dataset data/robomimic/datasets/stack_d1/stack_d1.hdf5 \
  --output tasks/rq1_transfer/results/stack_d1_new_check/preflight
python tasks/rq1_transfer/stack_d1/preflight.py model \
  --dataset tasks/rq1_transfer/results/stack_d1_new_check/preflight/smoke_two_demos_abs.hdf5 \
  --output tasks/rq1_transfer/results/stack_d1_new_check/preflight
python tasks/rq1_transfer/stack_d1/preflight.py vector \
  --dataset tasks/rq1_transfer/results/stack_d1_new_check/preflight/smoke_two_demos_abs.hdf5 \
  --output tasks/rq1_transfer/results/stack_d1_new_check/preflight
python tasks/rq1_transfer/stack_d1/preflight.py workspace \
  --dataset tasks/rq1_transfer/results/stack_d1_new_check/preflight/smoke_two_demos_abs.hdf5 \
  --output tasks/rq1_transfer/results/stack_d1_new_check/workspace
python tasks/rq1_transfer/stack_d1/preflight.py data \
  --dataset data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5 \
  --output tasks/rq1_transfer/results/stack_d1_new_check/data
```

Use a fresh output directory. GPU tests require at least 10 GiB free and cap
their PyTorch allocator at 25% of the GPU. Run stages sequentially. The workspace
smoke writes a roughly 4.8 GB checkpoint. Active jobs continue independently.

## Formal run boundary

The intended reference is the unmodified upstream
`train_equi_diffusion_unet_abs` recipe with `task_name=stack_d1 n_demo=100`:
500 epochs, batch 128, C8 image model, full U-Net widths, EMA, 100 diffusion
steps, and scheduled evaluation every 10 epochs. Default evaluation is six
training initial states and 50 test seeds, up to 400 steps each. Changing rollout
worker count for memory management does not change these episode counts.

The raw dataset must be converted with the official
`equi_diffpo/scripts/robomimic_dataset_conversion.py` before training with
absolute actions. Conversion output remains `.part` until all episodes,
unchanged state/observation arrays, finite actions, and gripper preservation
are verified and a new hash receipt is recorded. Formal training uses the
converted complete file, never the two-demo preflight file.

The subsequent priority run passed formal-batch preflight, froze a new release,
and started with four rollout workers. On the user's priority instruction, the
25/100-demo C4 queue was paused at a verified 12,000-step checkpoint, with an
automatic continuation guard after Stack-D1 completes. No frozen C4 source was
changed. The original small-batch checks are not a full-run ETA; see the linked
run record for actual batch-128 measurements.
