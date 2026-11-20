#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
set -euo pipefail
umask 0002
ROBOTIC_ROOT=${1:?Project root required}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 WANDB_MODE=disabled
exec "$ROBOTIC_ROOT/envs/official_dp/bin/python" \
  "$ROBOTIC_ROOT/releases/official_dp_s42_20261120_v1/runner.py" --project "$ROBOTIC_ROOT"
