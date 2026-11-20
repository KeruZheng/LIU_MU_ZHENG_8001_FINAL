#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
set -euo pipefail
umask 0002
ROBOTIC_ROOT=${1:?Project root required}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 WANDB_MODE=disabled
exec "$ROBOTIC_ROOT/envs/official_dp/bin/python" \
  "$ROBOTIC_ROOT/operations/resource_20261119/official_preflight.py" --project "$ROBOTIC_ROOT"
