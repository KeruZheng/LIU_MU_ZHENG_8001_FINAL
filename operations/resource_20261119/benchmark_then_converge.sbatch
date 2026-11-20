#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
set -euo pipefail
umask 0002
ROBOTIC_ROOT=${1:?Project root required}
export PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
exec "$ROBOTIC_ROOT/envs/rq3/bin/python" \
  "$ROBOTIC_ROOT/operations/resource_20261119/benchmark_then_converge.py" \
  --project "$ROBOTIC_ROOT" "${@:2}"
