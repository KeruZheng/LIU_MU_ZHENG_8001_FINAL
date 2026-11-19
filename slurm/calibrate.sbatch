#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROBOTIC_ROOT=${1:?Absolute robotic project root required}
ROBOTIC_CONFIG=${2:?Relative calibration configuration required}
ROBOTIC_RUN=${3:?Unique run name required}
[[ "$ROBOTIC_ROOT" == /* && "${ROBOTIC_ROOT##*/}" == robotic ]]
[[ "$ROBOTIC_CONFIG" != /* && "$ROBOTIC_CONFIG" != *..* ]]
[[ "$ROBOTIC_RUN" =~ ^[a-zA-Z0-9_-]+$ ]]
cd "$ROBOTIC_ROOT"
export PYTHONPATH="$ROBOTIC_ROOT/src"
export HF_HOME="$ROBOTIC_ROOT/envs/huggingface"
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
sha256sum --check manifests/code.sha256 manifests/data.sha256
TRAIN_START=$(date +%s)
envs/robotic/bin/python -m pusht_study.train --config "$ROBOTIC_CONFIG" \
  --output "runs/$ROBOTIC_RUN" --device cuda
TRAIN_END=$(date +%s)
envs/robotic/bin/python -m pusht_study.evaluate --checkpoint "runs/$ROBOTIC_RUN/last.pt" \
  --output "runs/$ROBOTIC_RUN/evaluation_300.json" --device cuda --episodes 2 --max-steps 300
EVAL_END=$(date +%s)
CKPT_BYTES=$(stat -c %s "runs/$ROBOTIC_RUN/last.pt")
envs/robotic/bin/python - "$ROBOTIC_RUN" "$TRAIN_START" "$TRAIN_END" "$EVAL_END" "$CKPT_BYTES" <<'PY'
import json
import sys
from pathlib import Path
run, start, train_end, eval_end, ckpt = sys.argv[1:]
out = {
    "run": run,
    "train_seconds": int(train_end) - int(start),
    "eval_seconds_2x300": int(eval_end) - int(train_end),
    "checkpoint_bytes": int(ckpt),
}
Path(f"runs/{run}/calibration_timing.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
PY
