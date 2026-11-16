#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROBOTIC_ROOT=${1:?Absolute robotic project root required}
ROBOTIC_CONFIG=${2:?Relative pilot configuration required}
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
envs/robotic/bin/python -m pusht_study.train --config "$ROBOTIC_CONFIG" \
  --output "runs/$ROBOTIC_RUN" --device cuda
envs/robotic/bin/python -m pusht_study.evaluate --checkpoint "runs/$ROBOTIC_RUN/last.pt" \
  --output "runs/$ROBOTIC_RUN/evaluation.json" --device cuda
